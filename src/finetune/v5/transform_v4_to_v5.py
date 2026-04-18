"""Transform existing V4 SFT/DPO corpora into V5-format training rows.

The V4 corpus (``finetune_artifacts/teacher_data/master_v4.jsonl``,
31K rows) was generated for a model that relied on a 200-line system
prompt at inference. Every V4 example ships with:

    <|im_start|>system
    You are DocWain, an enterprise document intelligence assistant. ...
    <|im_end|>
    <|im_start|>user
    {task}
    <|im_end|>
    <|im_start|>assistant
    {target response}
    <|im_end|>

For V5 we want identity **baked into the weights**, which means the
system field must be empty at training time. The user/assistant portions
are still high-quality signal — we reuse them by stripping the system
segment and re-tagging each row with one capability from the charter
(``src.finetune.v5.capability_charter``).

Capability inference is deterministic based on the V4 row's ``area`` +
``category`` + ``difficulty`` metadata. Anything ambiguous falls into
``entity_extraction`` (the safest default for the V4 corpus, which is
70% extraction-heavy).

Output: ``finetune_artifacts/v5/sft_reused.jsonl``
        ``finetune_artifacts/v5/dpo_reused.jsonl``

Each output row:
    {
        "capability": str,  # one of the 12 charter keys
        "source": "v4_reused",
        "difficulty": str,
        "system": "",       # always empty — identity in weights
        "user": str,
        "assistant": str,
        "original_v4_area": str,
        "original_v4_source": str,
    }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finetune.v5.capability_charter import CHARTER  # noqa: E402


# ChatML parser — ``<|im_start|>role\ncontent<|im_end|>`` sections.
_CHATML_RE = re.compile(
    r"<\|im_start\|>(\w+)\s*\n(.*?)(?=<\|im_end\|>)",
    re.DOTALL,
)


def _parse_chatml(text: str) -> Dict[str, str]:
    """Return {system, user, assistant} strings from a ChatML-wrapped blob.

    When a role appears multiple times, the last instance wins — matches
    how the trainer consumes it. Empty string for missing roles.
    """
    out: Dict[str, str] = {"system": "", "user": "", "assistant": ""}
    for m in _CHATML_RE.finditer(text or ""):
        role = m.group(1).strip().lower()
        body = m.group(2).strip()
        if role in out:
            out[role] = body
    return out


# V4 area / category → V5 capability mapping.
# Keys are lowercased for fuzzy match; the mapper checks both area and
# category and falls through to a difficulty-based default.
_AREA_MAP: Dict[str, str] = {
    # Direct matches from V4 categorisation
    "long_context": "cross_doc_reasoning",
    "long_context_reasoning": "cross_doc_reasoning",
    "numerical": "entity_extraction",
    "numerical_reasoning": "entity_extraction",
    "temporal": "entity_extraction",
    "temporal_reasoning": "entity_extraction",
    "tabular": "entity_extraction",
    "table": "entity_extraction",
    "table_extraction": "entity_extraction",
    "boundary_detection": "entity_extraction",
    "boundary": "entity_extraction",
    "document_extraction": "entity_extraction",
    "extraction": "entity_extraction",
    "classification": "doctype_classification",
    "domain": "domain_recognition",
    "domain_classification": "domain_recognition",
    "legal": "cross_doc_reasoning",
    "legal_reasoning": "cross_doc_reasoning",
    "comparison": "cross_doc_reasoning",
    "reasoning": "intent_understanding",
    "intent": "intent_understanding",
    "intent_understanding": "intent_understanding",
    "layout": "layout_understanding",
    "layout_ocr": "layout_understanding",
    "ocr": "layout_understanding",
    "tool": "tool_calling",
    "tool_calling": "tool_calling",
    "tool_use": "tool_calling",
    "identity": "identity_in_weights",
    "schema": "schema_adherence",
    "refusal": "grounded_refusal",
    "context": "context_dependence",
    "citation": "citation_discipline",
}


def _infer_capability(area: str, category: str, difficulty: str) -> str:
    """Map V4 (area, category, difficulty) → V5 capability_id.

    Always returns a valid charter key; defaults to ``entity_extraction``
    when the V4 metadata is too generic to classify.
    """
    area_l = (area or "").strip().lower()
    category_l = (category or "").strip().lower()

    if area_l in _AREA_MAP:
        return _AREA_MAP[area_l]
    if category_l in _AREA_MAP:
        return _AREA_MAP[category_l]

    # Look for substring hits — V4 uses freeform strings sometimes
    for needle, cap in _AREA_MAP.items():
        if needle in area_l or needle in category_l:
            return cap

    return "entity_extraction"


@dataclass
class TransformStats:
    rows_read: int = 0
    rows_written: int = 0
    rows_skipped_empty: int = 0
    rows_skipped_system_only: int = 0
    rows_skipped_no_assistant: int = 0
    per_capability: Dict[str, int] = None

    def __post_init__(self):
        if self.per_capability is None:
            self.per_capability = {k: 0 for k in CHARTER}


def transform_sft(
    src_path: str,
    dst_path: str,
    *,
    limit: Optional[int] = None,
    drop_system: bool = True,
) -> TransformStats:
    """Stream-transform an SFT JSONL file from V4 shape to V5 shape."""
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    stats = TransformStats()
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            stats.rows_read += 1
            if limit and stats.rows_read > limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = row.get("text") or ""
            if not text:
                stats.rows_skipped_empty += 1
                continue

            parts = _parse_chatml(text)
            user = parts.get("user", "").strip()
            assistant = parts.get("assistant", "").strip()
            system = parts.get("system", "").strip()

            if not assistant:
                stats.rows_skipped_no_assistant += 1
                continue
            if not user:
                # System-only rows aren't useful for V5 where system is empty
                stats.rows_skipped_system_only += 1
                continue

            # Strip <think> blocks inside the assistant response — Qwen3 format
            assistant_clean = re.sub(
                r"<think>.*?</think>", "", assistant, flags=re.DOTALL,
            ).strip() or assistant

            capability = _infer_capability(
                row.get("area", ""),
                row.get("category", ""),
                row.get("difficulty", ""),
            )

            new_row = {
                "capability": capability,
                "source": "v4_reused",
                "difficulty": row.get("difficulty", "medium"),
                "system": "" if drop_system else system,
                "user": user,
                "assistant": assistant_clean,
                "original_v4_area": row.get("area", ""),
                "original_v4_source": row.get("source", ""),
            }
            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            stats.rows_written += 1
            stats.per_capability[capability] += 1

    return stats


def transform_dpo(
    src_path: str,
    dst_path: str,
    *,
    limit: Optional[int] = None,
    drop_system: bool = True,
) -> TransformStats:
    """Transform a DPO JSONL — expects V4 DPO shape: {prompt, chosen, rejected}.

    Falls back gracefully on other shapes (e.g. {chosen_text, rejected_text}).
    """
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    stats = TransformStats()
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            stats.rows_read += 1
            if limit and stats.rows_read > limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            # V4 DPO variants:
            #   {prompt, chosen, rejected, area, category, difficulty}
            #   {text_chosen, text_rejected, ...}
            prompt = row.get("prompt") or ""
            chosen = row.get("chosen") or row.get("text_chosen") or ""
            rejected = row.get("rejected") or row.get("text_rejected") or ""

            # If prompt is a ChatML blob, strip system/user out of it
            if "<|im_start|>" in prompt:
                parts = _parse_chatml(prompt)
                user = parts.get("user", "").strip()
            else:
                user = prompt.strip()

            # chosen / rejected may also carry ChatML wrapping
            for field_name, value in (("chosen", chosen), ("rejected", rejected)):
                if "<|im_start|>" in value:
                    value = _parse_chatml(value).get("assistant", value).strip()
                if field_name == "chosen":
                    chosen_clean = value
                else:
                    rejected_clean = value

            if not (user and chosen_clean and rejected_clean):
                stats.rows_skipped_empty += 1
                continue

            capability = _infer_capability(
                row.get("area", ""),
                row.get("category", ""),
                row.get("difficulty", ""),
            )

            new_row = {
                "capability": capability,
                "source": "v4_reused_dpo",
                "system": "" if drop_system else "",
                "user": user,
                "chosen": chosen_clean,
                "rejected": rejected_clean,
                "difficulty": row.get("difficulty", "medium"),
            }
            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            stats.rows_written += 1
            stats.per_capability[capability] += 1

    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--v4-sft",
        default="finetune_artifacts/teacher_data/master_v4.jsonl",
    )
    ap.add_argument(
        "--v4-dpo",
        default="finetune_artifacts/teacher_data/master_dpo_v4.jsonl",
    )
    ap.add_argument(
        "--out-sft", default="finetune_artifacts/v5/sft_reused.jsonl",
    )
    ap.add_argument(
        "--out-dpo", default="finetune_artifacts/v5/dpo_reused.jsonl",
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("transforming V4 SFT → V5 ...")
    if Path(args.v4_sft).exists():
        sft_stats = transform_sft(args.v4_sft, args.out_sft, limit=args.limit)
        print(f"  read: {sft_stats.rows_read}, written: {sft_stats.rows_written}, "
              f"skipped_empty: {sft_stats.rows_skipped_empty}, "
              f"skipped_no_assistant: {sft_stats.rows_skipped_no_assistant}")
        print("  by capability:")
        for cap, count in sorted(sft_stats.per_capability.items(), key=lambda kv: -kv[1]):
            target = CHARTER[cap].sft_target_rows
            pct = (count / target * 100) if target else 0
            print(f"    {cap:25s} {count:>6}  (target {target:>6}, {pct:>5.1f}%)")
    else:
        print(f"  [skip] {args.v4_sft} not found")

    print()
    print("transforming V4 DPO → V5 ...")
    if Path(args.v4_dpo).exists():
        dpo_stats = transform_dpo(args.v4_dpo, args.out_dpo, limit=args.limit)
        print(f"  read: {dpo_stats.rows_read}, written: {dpo_stats.rows_written}, "
              f"skipped_empty: {dpo_stats.rows_skipped_empty}")
        print("  by capability:")
        for cap, count in sorted(dpo_stats.per_capability.items(), key=lambda kv: -kv[1]):
            target = CHARTER[cap].dpo_target_pairs
            pct = (count / target * 100) if target else 0
            print(f"    {cap:25s} {count:>6}  (target {target:>6}, {pct:>5.1f}%)")
    else:
        print(f"  [skip] {args.v4_dpo} not found")


if __name__ == "__main__":
    main()
