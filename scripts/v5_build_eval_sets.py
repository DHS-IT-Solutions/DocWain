#!/usr/bin/env python3
"""Sample eval sets from the training corpus.

evaluate.py's capability charter expects an eval JSONL per capability
at ``finetune_artifacts/v5/eval_sets/<cap.eval_set>.jsonl``. We don't
hold data out during training, so we sample ~30 rows per capability
from the training corpus as a "did the model learn the pattern"
verification — not a generalisation benchmark.

Each row emits ``{prompt, expected}`` where ``expected`` is shaped to
match the capability's scorer in ``evaluate.py``.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "finetune_artifacts/v5/sft_combined.jsonl"
EVAL_DIR = ROOT / "finetune_artifacts/v5/eval_sets"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# charter.eval_set → filename mapping (pulled from capability_charter.py)
EVAL_FILE = {
    "layout_understanding":     "layout_regions_v5.jsonl",
    "domain_recognition":       "domain_classification_v5.jsonl",
    "doctype_classification":   "doctype_v5.jsonl",
    "entity_extraction":        "extraction_v5.jsonl",
    "intent_understanding":     "intent_narrative_v5.jsonl",
    "context_dependence":       "contrastive_v5.jsonl",
    "cross_doc_reasoning":      "cross_doc_v5.jsonl",
    "grounded_refusal":         "hard_negatives_v5.jsonl",
    "schema_adherence":         "schema_v5.jsonl",
    "tool_calling":             "tool_traces_v5.jsonl",
    "identity_in_weights":      "identity_probes_v5.jsonl",
    "citation_discipline":      "citation_v5.jsonl",
}

PER_CAP = 30  # rows per eval set


def expected_for(cap: str, assistant: str):
    """Return the ``expected`` field shape evaluate.py's scorer needs."""
    if cap in ("domain_recognition", "doctype_classification"):
        # Scorer does lowercase startswith / substring match on the first word-ish token
        return assistant.strip().split()[0].lower().strip(".:!,")
    if cap == "schema_adherence":
        # Scorer wants key-set match; pass parsed object if valid JSON else str
        try:
            obj = json.loads(assistant)
            if isinstance(obj, dict):
                return {k: "" for k in obj.keys()}  # key-set only
        except json.JSONDecodeError:
            pass
        return assistant
    if cap in ("grounded_refusal", "identity_in_weights", "tool_calling"):
        # Custom scorers ignore `expected`; still include the assistant for reference
        return assistant
    # For generic scorers (intent, layout, context, cross_doc, citation): pass
    # a list of keywords pulled from the assistant. Loose match.
    words = [w.strip(".,:;!?()[]\"'").lower() for w in assistant.split()]
    keywords = [w for w in words if len(w) > 4 and w.isalpha()][:5]
    return keywords or [assistant[:30].lower()]


def main() -> None:
    rng = random.Random(42)
    buckets: dict[str, list[dict]] = defaultdict(list)
    with CORPUS.open("r") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cap = row.get("capability")
            if cap in EVAL_FILE:
                buckets[cap].append(row)

    print("Sampling eval sets:")
    for cap, filename in EVAL_FILE.items():
        rows = buckets.get(cap, [])
        if not rows:
            print(f"  SKIP {cap:28s} (no training rows)")
            continue
        sample = rng.sample(rows, k=min(PER_CAP, len(rows)))
        out = EVAL_DIR / filename
        with out.open("w") as f:
            for r in sample:
                f.write(json.dumps({
                    "prompt": r["user"],
                    "expected": expected_for(cap, r["assistant"]),
                }, ensure_ascii=False) + "\n")
        print(f"  wrote {out.name:40s}  n={len(sample)}")


if __name__ == "__main__":
    main()
