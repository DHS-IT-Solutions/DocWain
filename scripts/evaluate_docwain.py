#!/usr/bin/env python3
"""CLI entry point for running the DocWain LLM-judge evaluation pipeline.

Examples::

    PYTHONPATH=. python scripts/evaluate_docwain.py
    PYTHONPATH=. python scripts/evaluate_docwain.py --max-examples 20 --judge ollama
    PYTHONPATH=. python scripts/evaluate_docwain.py \\
        --model-url http://localhost:8100/v1/chat/completions \\
        --model-name docwain-smart \\
        --judge vllm \\
        --max-examples 50 \\
        --output-dir eval_results
"""

from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_docwain",
        description="Run the LLM-judge evaluation pipeline against DocWain.",
    )
    parser.add_argument(
        "--model-url",
        default="http://localhost:8100/v1/chat/completions",
        help="OpenAI-compatible chat completions URL for the model under test "
             "(default: http://localhost:8100/v1/chat/completions)",
    )
    parser.add_argument(
        "--model-name",
        default="docwain",
        help="Model alias/name to send in the request payload (default: docwain)",
    )
    parser.add_argument(
        "--judge",
        choices=["vllm", "ollama", "heuristic"],
        default="vllm",
        help="Judge backend to prefer (default: vllm; falls back automatically "
             "if unavailable)",
    )
    parser.add_argument(
        "--judge-url",
        default=None,
        help="Override the judge endpoint URL (default: same as --model-url for "
             "vllm, or http://localhost:11434/api/chat for ollama)",
    )
    parser.add_argument(
        "--judge-model",
        default="docwain",
        help="Judge model alias (default: docwain)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Maximum number of test bank examples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory for saving the detailed JSON report (default: eval_results)",
    )
    return parser


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from src.finetune.evaluation.llm_judge import (
        DEFAULT_OLLAMA_URL,
        DEFAULT_VLLM_MODEL,
        run_evaluation_suite,
    )

    # Resolve judge URL
    judge_url = args.judge_url
    judge_model = args.judge_model

    if judge_url is None:
        if args.judge == "ollama":
            judge_url = DEFAULT_OLLAMA_URL
            if judge_model == DEFAULT_VLLM_MODEL:
                judge_model = "qwen3:14b"
        else:
            judge_url = args.model_url  # same server judges its own responses

    print(f"Model  : {args.model_name} @ {args.model_url}", flush=True)
    print(f"Judge  : {judge_model} @ {judge_url} (preferred: {args.judge})", flush=True)
    print(f"Examples: {args.max_examples}", flush=True)
    print(f"Output : {args.output_dir}", flush=True)
    print(flush=True)

    results = run_evaluation_suite(
        output_dir=args.output_dir,
        model_url=args.model_url,
        model_name=args.model_name,
        judge_url=judge_url,
        judge_model=judge_model,
        max_examples=args.max_examples,
    )

    overall = results.get("overall_avg", 0.0)
    min_dim = results.get("min_dimension", 0.0)

    # Exit 1 if below basics gate (3.5 avg / 3.0 min)
    if overall < 3.5 or min_dim < 3.0:
        print(
            f"\nFAIL: overall_avg={overall:.3f} min_dimension={min_dim:.3f} "
            "(basics gate: avg>=3.5, min>=3.0)",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    else:
        prod_note = (
            " (production gate PASSED)" if overall >= 4.0 and min_dim >= 3.5 else ""
        )
        print(
            f"\nPASS: overall_avg={overall:.3f} min_dimension={min_dim:.3f}{prod_note}",
            flush=True,
        )


if __name__ == "__main__":
    main()
