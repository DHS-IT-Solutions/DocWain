#!/usr/bin/env python3
"""Evaluate a MergeKit-produced DocWain seed against the V3 baseline.

Pipeline:
  1. Load the merged model via transformers (bfloat16, device_map="auto").
  2. For each of the first N rows in the golden-set JSONL, parse the
     ChatML-formatted ``text`` field into (system, user, assistant)
     segments, feed (system + user) into the model, capture the
     generated assistant turn, and score it with the existing LLM-judge
     at src/finetune/evaluation/llm_judge.py (score_response).
  3. Write a JSON report with per-row scores and summary stats.
  4. Exit 0 iff mean_score >= baseline_score, else exit 1 (CI gate).

Usage:
    python scripts/evaluate_merge_seed.py \
        --model-path /home/ubuntu/PycharmProjects/DocWain/models/DocWain-14B-v5-seed \
        --baseline-score 4.71 \
        --golden-set finetune_artifacts/teacher_data/master_v4.jsonl \
        --output finetune_artifacts/v5_seed_eval.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the project root importable so we can use the existing LLM-judge.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.finetune.evaluation.llm_judge import score_response  # noqa: E402

DEFAULT_GOLDEN_SET = "finetune_artifacts/teacher_data/master_v4.jsonl"
DEFAULT_NUM_ROWS = 200
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# ChatML parsing
# ---------------------------------------------------------------------------

CHATML_BLOCK_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>",
    re.DOTALL,
)


def parse_chatml(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (system, user, assistant) from a ChatML-formatted string.

    Missing roles become ``None``. When multiple turns exist, takes the
    first occurrence of each role — matches the master_v4 single-turn layout.
    """
    system_msg: Optional[str] = None
    user_msg: Optional[str] = None
    assistant_msg: Optional[str] = None
    for match in CHATML_BLOCK_RE.finditer(text):
        role, content = match.group(1), match.group(2).strip()
        if role == "system" and system_msg is None:
            system_msg = content
        elif role == "user" and user_msg is None:
            user_msg = content
        elif role == "assistant" and assistant_msg is None:
            assistant_msg = content
    return system_msg, user_msg, assistant_msg


def load_golden_set(path: str, limit: int) -> List[Dict[str, Any]]:
    """Load up to *limit* rows, parsed into {prompt, reference, system} dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] line {line_num}: json decode failed ({exc})", flush=True)
                continue

            text = obj.get("text", "")
            system_msg, user_msg, assistant_msg = parse_chatml(text)
            if user_msg is None or assistant_msg is None:
                print(f"[warn] line {line_num}: missing user/assistant turn, skipping",
                      flush=True)
                continue

            rows.append({
                "line_num": line_num,
                "system": system_msg,
                "prompt": user_msg,
                "reference": assistant_msg,
                "category": obj.get("category", "unknown"),
                "area": obj.get("area", "unknown"),
                "difficulty": obj.get("difficulty", "unknown"),
            })
            if len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------


def load_model(model_path: str):
    """Load the merged model + tokenizer. Tolerates MergeKit output layout."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] loading tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] loading model from {model_path} (bf16, device_map=auto)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def build_chatml_prompt(system: Optional[str], user: str) -> str:
    """Reconstruct the ChatML prefix the model was trained on, open the
    assistant turn so the model continues."""
    parts: List[str] = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def generate_response(
    model,
    tokenizer,
    system: Optional[str],
    user: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate an assistant response. Strips any <|im_end|> and trailing junk."""
    import torch

    prompt_text = build_chatml_prompt(system, user)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature <= 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    # Strip the prompt tokens; decode only the newly generated portion.
    input_len = inputs["input_ids"].shape[1]
    new_tokens = out[0, input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Cut at the first <|im_end|> if present.
    end_idx = decoded.find("<|im_end|>")
    if end_idx != -1:
        decoded = decoded[:end_idx]
    # Also strip any dangling special tokens.
    decoded = re.sub(r"<\|im_(start|end)\|>", "", decoded)
    return decoded.strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_row(prompt: str, response: str, reference: str) -> Tuple[float, Dict[str, float], str]:
    """Call the existing LLM-judge and return (row_mean, dims, judge_source)."""
    result = score_response(prompt, response, reference)
    dims = result.get("scores", {})
    judge_source = result.get("judge_source", "unknown")
    if not dims:
        return 0.0, {}, judge_source
    row_mean = sum(dims.values()) / len(dims)
    return row_mean, dims, judge_source


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a DocWain merge seed.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the merged model directory (MergeKit output).",
    )
    parser.add_argument(
        "--baseline-score",
        type=float,
        default=4.71,
        help="Minimum mean LLM-judge score required to pass. Default: 4.71 (V3).",
    )
    parser.add_argument(
        "--golden-set",
        default=DEFAULT_GOLDEN_SET,
        help=f"Path to the eval JSONL (default: {DEFAULT_GOLDEN_SET}).",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=DEFAULT_NUM_ROWS,
        help=f"Number of rows to evaluate from the golden set (default: {DEFAULT_NUM_ROWS}).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Generation cap per response (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (default: 0.0 = greedy).",
    )
    args = parser.parse_args()

    # Resolve relative paths against project root.
    golden_path = args.golden_set
    if not os.path.isabs(golden_path):
        golden_path = str(PROJECT_ROOT / golden_path)
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = str(PROJECT_ROOT / output_path)

    if not os.path.exists(args.model_path):
        print(f"[fatal] model path not found: {args.model_path}", file=sys.stderr)
        return 2
    if not os.path.exists(golden_path):
        print(f"[fatal] golden set not found: {golden_path}", file=sys.stderr)
        return 2

    print(f"[eval] model_path      = {args.model_path}", flush=True)
    print(f"[eval] baseline_score  = {args.baseline_score}", flush=True)
    print(f"[eval] golden_set      = {golden_path}", flush=True)
    print(f"[eval] num_rows        = {args.num_rows}", flush=True)
    print(f"[eval] output          = {output_path}", flush=True)

    rows = load_golden_set(golden_path, args.num_rows)
    print(f"[eval] loaded {len(rows)} rows", flush=True)
    if not rows:
        print("[fatal] no usable rows in golden set", file=sys.stderr)
        return 2

    model, tokenizer = load_model(args.model_path)

    per_row_scores: List[Dict[str, Any]] = []
    judge_source_counts: Dict[str, int] = {}
    t0 = time.time()

    try:
        for idx, row in enumerate(rows, start=1):
            elapsed = time.time() - t0
            print(f"[gen] {idx}/{len(rows)}  (elapsed {elapsed:.0f}s)", flush=True)
            try:
                response = generate_response(
                    model,
                    tokenizer,
                    row["system"],
                    row["prompt"],
                    args.max_new_tokens,
                    args.temperature,
                )
            except Exception as exc:  # noqa: BLE001 — tolerate any model error
                print(f"[warn] generation failed on row {idx}: {exc}", flush=True)
                response = ""

            row_mean, dims, judge_source = score_row(
                row["prompt"], response, row["reference"]
            )
            judge_source_counts[judge_source] = judge_source_counts.get(judge_source, 0) + 1

            per_row_scores.append({
                "index": idx,
                "line_num": row["line_num"],
                "category": row["category"],
                "area": row["area"],
                "difficulty": row["difficulty"],
                "prompt_chars": len(row["prompt"]),
                "response_chars": len(response),
                "response_preview": response[:400],
                "row_mean": round(row_mean, 4),
                "dimensions": {k: round(v, 4) for k, v in dims.items()},
                "judge_source": judge_source,
            })

            print(f"       mean={row_mean:.3f}  judge={judge_source}", flush=True)
    finally:
        # Free GPU memory promptly so the judge vLLM (if colocated) isn't starved.
        try:
            del model
        except NameError:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # Aggregate
    means = [r["row_mean"] for r in per_row_scores if r["row_mean"] > 0]
    if not means:
        print("[fatal] no scored rows — all generations or judge calls failed",
              file=sys.stderr)
        mean_score = 0.0
        min_score = 0.0
        max_score = 0.0
    else:
        mean_score = statistics.mean(means)
        min_score = min(means)
        max_score = max(means)

    passed_baseline = mean_score >= args.baseline_score

    report = {
        "model_path": args.model_path,
        "golden_set": golden_path,
        "num_rows_requested": args.num_rows,
        "num_rows_evaluated": len(per_row_scores),
        "num_rows_scored": len(means),
        "baseline_score": args.baseline_score,
        "mean_score": round(mean_score, 4),
        "min_score": round(min_score, 4),
        "max_score": round(max_score, 4),
        "passed_baseline": passed_baseline,
        "judge_sources": judge_source_counts,
        "wall_time_seconds": round(time.time() - t0, 2),
        "per_row_scores": per_row_scores,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[done] report written to {output_path}", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("DOCWAIN V5 SEED EVAL", flush=True)
    print("=" * 60, flush=True)
    print(f"  model_path     : {args.model_path}", flush=True)
    print(f"  rows evaluated : {len(per_row_scores)}  (scored: {len(means)})", flush=True)
    print(f"  mean_score     : {mean_score:.4f}", flush=True)
    print(f"  min_score      : {min_score:.4f}", flush=True)
    print(f"  max_score      : {max_score:.4f}", flush=True)
    print(f"  baseline       : {args.baseline_score:.4f}", flush=True)
    print(f"  judge_sources  : {judge_source_counts}", flush=True)
    verdict = "PASS" if passed_baseline else "FAIL"
    print(f"  verdict        : {verdict}", flush=True)
    print("=" * 60, flush=True)

    return 0 if passed_baseline else 1


if __name__ == "__main__":
    sys.exit(main())
