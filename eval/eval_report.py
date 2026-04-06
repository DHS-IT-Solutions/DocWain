"""Evaluation report comparison for DocWain accuracy harness."""

import json
from pathlib import Path
from typing import Optional


def _load_results(results_dir: str = "eval/results"):
    """Load all result files sorted by name (timestamp order)."""
    p = Path(results_dir)
    if not p.exists():
        return []
    files = sorted(p.glob("eval_*.json"))
    results = []
    for f in files:
        with open(f, "r") as fh:
            data = json.load(fh)
            data["_file"] = str(f)
            results.append(data)
    return results


def compare_latest(results_dir: str = "eval/results") -> Optional[str]:
    """Load all results, print latest summary, and compare to previous run if available.

    Returns:
        Formatted report string, or None if no results found.
    """
    all_results = _load_results(results_dir)
    if not all_results:
        print("No evaluation results found.")
        return None

    latest = all_results[-1]
    s = latest["summary"]

    lines = [
        "=" * 60,
        "EVALUATION REPORT",
        "=" * 60,
        f"Timestamp:          {s['timestamp']}",
        f"Total cases:        {s['total_cases']}",
        f"Passed:             {s['passed']}",
        f"Errors:             {s['errors']}",
        f"Avg fact coverage:  {s['avg_fact_coverage']:.2%}",
        f"Avg latency (ms):   {s['avg_latency_ms']:.1f}",
        f"Total hallucinations: {s['total_hallucinations']}",
    ]

    if s.get("by_category"):
        lines.append("")
        lines.append("By category:")
        for cat, stats in s["by_category"].items():
            lines.append(
                f"  {cat}: coverage={stats['avg_fact_coverage']:.2%}, "
                f"latency={stats['avg_latency_ms']:.1f}ms, n={stats['count']}"
            )

    # Compare to previous
    if len(all_results) >= 2:
        prev = all_results[-2]
        ps = prev["summary"]
        lines.append("")
        lines.append("-" * 60)
        lines.append("COMPARISON TO PREVIOUS RUN")
        lines.append("-" * 60)
        lines.append(f"Previous timestamp: {ps['timestamp']}")

        cov_delta = s["avg_fact_coverage"] - ps["avg_fact_coverage"]
        lat_delta = s["avg_latency_ms"] - ps["avg_latency_ms"]
        hal_delta = s["total_hallucinations"] - ps["total_hallucinations"]

        cov_dir = "+" if cov_delta >= 0 else ""
        lat_dir = "+" if lat_delta >= 0 else ""
        hal_dir = "+" if hal_delta >= 0 else ""

        lines.append(f"Fact coverage:      {cov_dir}{cov_delta:.2%}")
        lines.append(f"Latency (ms):       {lat_dir}{lat_delta:.1f}")
        lines.append(f"Hallucinations:     {hal_dir}{hal_delta}")

    lines.append("=" * 60)
    report = "\n".join(lines)
    print(report)
    return report
