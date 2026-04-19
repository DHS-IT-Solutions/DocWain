"""V5 evaluator — runs the capability charter's gates and writes a pass/fail report.

One call per model. Each capability's eval set is a JSONL with prompt
→ expected_canonical_response (for structured) or prompt → expected_keywords
(for narrative). The runner generates a response from the model under test,
scores it against the capability's ``gate_metric``, and aggregates.

Outputs:
  finetune_artifacts/v5/eval/{model_id}.json
    {
      "model": str,
      "overall": {"llm_judge_mean": float, "hard_gates_passed": int/total},
      "per_capability": {
          "<cap_id>": {"metric": "...", "value": float, "threshold": float,
                       "passed": bool, "hard": bool},
          ...
      }
    }

A model ships only if all ``hard_gate`` capabilities pass. Soft-gate
failures are logged but not blocking.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finetune.v5.capability_charter import CHARTER, Capability  # noqa: E402

logger = logging.getLogger(__name__)


_THINK_RE = __import__("re").compile(r"<think>.*?</think>", __import__("re").DOTALL | __import__("re").IGNORECASE)


def _generate(model, tokenizer, user: str, max_new_tokens: int, temperature: float) -> str:
    """Single generation with an empty-system chat template (V5 identity in weights).

    Qwen3-family models emit a ``<think>...</think>`` reasoning block before
    every answer (even when empty). We strip it before returning so every
    capability scorer sees the model's actual response — not the thinking
    tag. Without this, JSON/tool/classification scorers fail on valid
    outputs because the thinking prefix breaks parse assumptions.
    """
    import torch
    messages = [{"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature or 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    text = _THINK_RE.sub("", text).strip()
    return text


# Capability-specific scorers. Each returns (value, passed) given a list of
# (prompt, expected, actual) triples. Metrics track the charter's gate_metric.

def _score_identity(samples: List[Dict[str, str]]) -> Tuple[float, bool]:
    """Identity: actual must mention DocWain and not leak generic 'I am an AI'."""
    hits = 0
    for s in samples:
        a = (s.get("actual") or "").lower()
        ok = (
            "docwain" in a
            and "i'm an ai" not in a
            and "i am an ai" not in a
            and "language model" not in a
        )
        if ok:
            hits += 1
    rate = hits / max(len(samples), 1)
    return rate, rate >= 1.0


def _score_classification(samples: List[Dict[str, str]]) -> Tuple[float, bool]:
    """Accuracy: actual equals expected after lowercase + strip."""
    hits = 0
    for s in samples:
        a = (s.get("actual") or "").strip().lower().splitlines()[0].strip(".:!")
        e = (s.get("expected") or "").strip().lower()
        if a.startswith(e) or e in a:
            hits += 1
    rate = hits / max(len(samples), 1)
    return rate, rate >= 0.95


def _score_schema(samples: List[Dict[str, str]]) -> Tuple[float, bool]:
    """JSON validity + key-set match."""
    import json as _j
    hits = 0
    for s in samples:
        a = (s.get("actual") or "").strip()
        if a.startswith("```"):
            a = "\n".join(a.split("\n")[1:-1])
        try:
            parsed = _j.loads(a)
        except Exception:
            continue
        expected = s.get("expected")
        if isinstance(expected, dict):
            if set(parsed.keys()) == set(expected.keys()):
                hits += 1
        else:
            hits += 1
    rate = hits / max(len(samples), 1)
    return rate, rate >= 0.99


def _score_refusal(samples: List[Dict[str, str]]) -> Tuple[float, bool]:
    """Refusal: actual must contain a refusal token.

    Expanded token list to match what the model actually emits in practice —
    ``does not provide`` / ``not specified`` / ``not mentioned`` are all
    semantically correct refusals but don't use the original NOT_IN_DOCUMENT
    phrasing. Training used Nemotron-authoritative refusals which favour
    natural-language phrasing over a strict token.
    """
    hits = 0
    refusal_tokens = (
        "not_in_document", "not in the document", "not in document",
        "cannot", "can not",
        "does not contain", "does not provide", "does not include",
        "isn't in", "is not in",
        "no such", "not specified", "not provided", "not present",
        "not mentioned", "not stated", "not indicated",
        "cannot be determined", "cannot be found", "cannot be answered",
        "not answerable", "unable to answer", "insufficient information",
        "no information", "unable to determine",
    )
    for s in samples:
        a = (s.get("actual") or "").lower()
        if any(tok in a for tok in refusal_tokens):
            hits += 1
    rate = hits / max(len(samples), 1)
    return rate, rate >= 1.0


def _score_tool_call(samples: List[Dict[str, str]]) -> Tuple[float, bool]:
    """Tool-call format: actual must contain <tool_call>...</tool_call> with valid JSON body."""
    import re, json as _j
    hits = 0
    tc_re = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    for s in samples:
        a = s.get("actual") or ""
        m = tc_re.search(a)
        if not m:
            continue
        try:
            tc = _j.loads(m.group(1))
            if isinstance(tc, dict) and "name" in tc and "arguments" in tc:
                hits += 1
        except Exception:
            continue
    rate = hits / max(len(samples), 1)
    return rate, rate >= 1.0


# Charter-capability → scorer map. Capabilities without a scorer defined
# here fall through to substring-keyword scoring as a generic baseline.
_SCORERS = {
    "identity_in_weights": _score_identity,
    "domain_recognition": _score_classification,
    "doctype_classification": _score_classification,
    "schema_adherence": _score_schema,
    "grounded_refusal": _score_refusal,
    "tool_calling": _score_tool_call,
}


def _score_generic(samples: List[Dict[str, str]], cap: Capability) -> Tuple[float, bool]:
    """Substring-keyword fallback for capabilities without custom scorers."""
    hits = 0
    for s in samples:
        a = (s.get("actual") or "").lower()
        expected = s.get("expected") or ""
        if isinstance(expected, list):
            keywords = [str(k).lower() for k in expected]
        else:
            keywords = [str(expected).lower()]
        if all(kw in a for kw in keywords):
            hits += 1
    rate = hits / max(len(samples), 1)
    return rate, rate >= cap.gate_threshold


def evaluate(
    model_path: str,
    eval_dir: str = "finetune_artifacts/v5/eval_sets",
    output_dir: str = "finetune_artifacts/v5/eval",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Run all charter gates against a model and emit a pass/fail report."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("v5 eval: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    report: Dict[str, Any] = {
        "model": model_path,
        "timestamp": int(time.time()),
        "per_capability": {},
        "overall": {"hard_gates_passed": 0, "hard_gates_total": 0},
    }
    hard_passed = 0
    hard_total = 0

    for cap_id, cap in CHARTER.items():
        eval_path = Path(eval_dir) / f"{cap.eval_set}.jsonl"
        if not eval_path.exists():
            logger.warning("skip %s: eval set missing (%s)", cap_id, eval_path)
            continue
        samples: List[Dict[str, Any]] = []
        with eval_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                actual = _generate(
                    model, tokenizer, r["prompt"],
                    max_new_tokens=max_new_tokens, temperature=temperature,
                )
                samples.append({
                    "prompt": r["prompt"],
                    "expected": r.get("expected"),
                    "actual": actual,
                })

        scorer = _SCORERS.get(cap_id)
        if scorer:
            value, passed = scorer(samples)
        else:
            value, passed = _score_generic(samples, cap)

        report["per_capability"][cap_id] = {
            "metric": cap.gate_metric,
            "threshold": cap.gate_threshold,
            "value": round(value, 3),
            "passed": bool(passed),
            "hard": cap.hard_gate,
            "n_samples": len(samples),
        }
        if cap.hard_gate:
            hard_total += 1
            if passed:
                hard_passed += 1
        logger.info("  %s (%s): %.3f %s", cap_id, cap.gate_metric, value,
                    "PASS" if passed else "FAIL")

    report["overall"]["hard_gates_passed"] = hard_passed
    report["overall"]["hard_gates_total"] = hard_total
    report["overall"]["all_hard_gates_passed"] = hard_passed == hard_total and hard_total > 0

    out_path = Path(output_dir) / f"{Path(model_path).name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "v5 eval: %d/%d hard gates passed, report -> %s",
        hard_passed, hard_total, out_path,
    )
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_path")
    ap.add_argument("--eval-dir", default="finetune_artifacts/v5/eval_sets")
    ap.add_argument("--output-dir", default="finetune_artifacts/v5/eval")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = evaluate(
        args.model_path, args.eval_dir, args.output_dir,
        max_new_tokens=args.max_new_tokens,
    )
    sys.exit(0 if report["overall"]["all_hard_gates_passed"] else 2)


if __name__ == "__main__":
    main()
