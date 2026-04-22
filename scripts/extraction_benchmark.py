#!/usr/bin/env python3
"""Extraction benchmark — run ExtractionEngine against a document directory.

Exercises the full engine (deterministic + 4-parallel + merger) and reports
per-file accuracy (via the deterministic validator) and speed (per-stage
timing).

Usage:
    python scripts/extraction_benchmark.py <dir>
    python scripts/extraction_benchmark.py <dir> --no-ai  # skip Layer 2

``--no-ai`` points V2/semantic/vision to an unreachable Ollama host so
those engines fail fast; useful to isolate deterministic extraction cost
when the AI layer isn't available.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extraction.engine import ExtractionEngine  # noqa: E402


def _format_row(name: str, stats: Dict[str, Any]) -> str:
    pass_str = "PASS" if stats["validation_passed"] else "FAIL"
    return (
        f"[{pass_str}] {name:<55} "
        f"fmt={stats['format']:<5} "
        f"text={stats['text_chars']:>6}ch "
        f"tables={stats['tables']:>2} "
        f"entities={stats['entities']:>3} "
        f"det={stats['deterministic_s']:.3f}s "
        f"parallel={stats['parallel_s']:.3f}s "
        f"total={stats['total_s']:.3f}s"
    )


def run_benchmark(directory: Path, out_path: Path = None, no_ai: bool = False) -> int:
    kwargs: Dict[str, Any] = {}
    if no_ai:
        # Point legacy Ollama-backed extractors at an unreachable host so
        # they fail fast (V2 now uses vLLM but structural/semantic/vision
        # stubs still import the host); isolates deterministic-only cost.
        kwargs["ollama_host"] = "http://127.0.0.1:1"

    engine = ExtractionEngine(**kwargs)

    # Wire the vLLM manager into V2 for benchmarking outside the API,
    # using the same config AppState would use. Skip when --no-ai.
    if not no_ai:
        try:
            from src.serving.vllm_manager import VLLMManager
            from src.serving.config import GPU_MODE_FILE
            engine.v2_extractor._manager = VLLMManager(gpu_mode_file=GPU_MODE_FILE)
            print(f"[info] V2 using vLLM at {engine.v2_extractor._manager._url}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] vLLM manager init failed: {exc}")

    files = sorted(p for p in directory.iterdir() if p.is_file())
    results: Dict[str, Any] = {}
    per_doc_totals = []
    per_doc_determ = []
    per_doc_parallel = []
    pass_count = 0

    for p in files:
        try:
            content = p.read_bytes()
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {p.name}: read failed: {exc}")
            results[p.name] = {"error": str(exc)}
            continue

        t_start = time.monotonic()
        merged = engine.extract(
            document_id=f"bench::{p.name}",
            subscription_id="bench-sub",
            profile_id="bench-profile",
            document_bytes=content,
            file_type=p.suffix.lstrip("."),
        )
        total = time.monotonic() - t_start

        validation = merged.metadata.get("deterministic_validation", {})
        stats = {
            "format": merged.raw_extraction["file_format"] if merged.raw_extraction else "?",
            "text_chars": len(merged.clean_text or ""),
            "tables": len(merged.tables),
            "entities": len(merged.entities),
            "deterministic_s": merged.metadata.get("deterministic_elapsed_s", 0.0),
            "parallel_s": merged.metadata.get("parallel_engines_elapsed_s", 0.0),
            "total_s": round(total, 3),
            "validation_passed": validation.get("passed", False),
            "validation_failed_checks": validation.get("failed_checks", []),
            "validation_advisories": validation.get("advisories", []),
        }
        print(_format_row(p.name, stats))
        for f in stats["validation_failed_checks"]:
            print(f"    - {f}")
        for a in stats["validation_advisories"]:
            print(f"    · {a}")
        results[p.name] = stats
        per_doc_totals.append(total)
        per_doc_determ.append(stats["deterministic_s"])
        per_doc_parallel.append(stats["parallel_s"])
        if stats["validation_passed"]:
            pass_count += 1

    print()
    n = len(per_doc_totals)
    if n:
        def _stat(vals: List[float]) -> str:
            s = sorted(vals)
            return f"min={min(s):.3f}s  median={s[n//2]:.3f}s  max={max(s):.3f}s  total={sum(s):.3f}s"
        print(f"== Accuracy: {pass_count}/{n} files passed deterministic validation ==")
        print(f"== Timing (deterministic only): {_stat(per_doc_determ)} ==")
        print(f"== Timing (parallel AI engines): {_stat(per_doc_parallel)} ==")
        print(f"== Timing (end-to-end per doc): {_stat(per_doc_totals)} ==")

    if out_path:
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        print(f"wrote {out_path}")

    return 0 if pass_count == n else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-ai", action="store_true",
                    help="Make AI engines fail fast (isolates deterministic cost)")
    args = ap.parse_args()
    sys.exit(run_benchmark(args.directory, out_path=args.out, no_ai=args.no_ai))


if __name__ == "__main__":
    main()
