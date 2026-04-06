"""Evaluation runner for DocWain accuracy harness."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def load_test_bank(path: str = "eval/test_bank.json") -> List[Dict]:
    """Load test cases from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def run_single_eval(case: Dict, ask_fn: Callable) -> Dict:
    """Run a single evaluation case and collect metrics.

    Args:
        case: Test case dict with id, query, expected_facts, negative_facts, expected_doc_ids.
        ask_fn: Callable matching ask_fn(query, profile_id=None, subscription_id=None) -> dict.

    Returns:
        Dict with status, latency_ms, fact_coverage, hallucination_count, etc.
    """
    result: Dict[str, Any] = {
        "case_id": case["id"],
        "category": case.get("category", "unknown"),
        "query": case["query"],
    }

    try:
        start = time.perf_counter()
        response = ask_fn(
            case["query"],
            profile_id=case.get("profile_id"),
            subscription_id=case.get("subscription_id"),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        answer = response.get("answer", "")
        source_doc_ids = response.get("source_doc_ids", [])
        answer_lower = answer.lower()

        # Fact coverage
        expected_facts = case.get("expected_facts", [])
        if expected_facts:
            found = sum(1 for f in expected_facts if f.lower() in answer_lower)
            fact_coverage = found / len(expected_facts)
        else:
            fact_coverage = 1.0

        # Hallucination detection
        negative_facts = case.get("negative_facts", [])
        hallucinated = [f for f in negative_facts if f.lower() in answer_lower]

        # Retrieval recall
        expected_doc_ids = case.get("expected_doc_ids", [])
        if expected_doc_ids:
            retrieved_set = set(source_doc_ids)
            recall_hits = sum(1 for d in expected_doc_ids if d in retrieved_set)
            retrieval_recall = recall_hits / len(expected_doc_ids)
        else:
            retrieval_recall = None

        result.update({
            "status": "ok",
            "latency_ms": round(elapsed_ms, 2),
            "fact_coverage": fact_coverage,
            "hallucination_count": len(hallucinated),
            "hallucinated_facts": hallucinated,
            "retrieval_recall": retrieval_recall,
            "answer_length": len(answer),
        })

    except Exception as exc:
        result.update({
            "status": "error",
            "error": str(exc),
            "latency_ms": None,
            "fact_coverage": 0.0,
            "hallucination_count": 0,
            "hallucinated_facts": [],
            "retrieval_recall": None,
            "answer_length": 0,
        })

    return result


def run_eval(
    test_bank_path: str = "eval/test_bank.json",
    ask_fn: Optional[Callable] = None,
    output_dir: str = "eval/results",
) -> Dict:
    """Run all evaluation cases, aggregate results, and save to a timestamped JSON file.

    Args:
        test_bank_path: Path to the test bank JSON.
        ask_fn: Callable matching the DocWain ask API.
        output_dir: Directory to write result files.

    Returns:
        Dict with summary statistics and individual results.
    """
    if ask_fn is None:
        raise ValueError("ask_fn is required")

    cases = load_test_bank(test_bank_path)
    results = [run_single_eval(case, ask_fn) for case in cases]

    ok_results = [r for r in results if r["status"] == "ok"]
    error_count = len(results) - len(ok_results)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(results),
        "passed": len(ok_results),
        "errors": error_count,
        "avg_fact_coverage": (
            sum(r["fact_coverage"] for r in ok_results) / len(ok_results)
            if ok_results
            else 0.0
        ),
        "avg_latency_ms": (
            sum(r["latency_ms"] for r in ok_results) / len(ok_results)
            if ok_results
            else 0.0
        ),
        "total_hallucinations": sum(r["hallucination_count"] for r in results),
    }

    # Breakdown by category
    categories: Dict[str, List[Dict]] = {}
    for r in ok_results:
        categories.setdefault(r["category"], []).append(r)
    summary["by_category"] = {
        cat: {
            "count": len(items),
            "avg_fact_coverage": sum(i["fact_coverage"] for i in items) / len(items),
            "avg_latency_ms": sum(i["latency_ms"] for i in items) / len(items),
        }
        for cat, items in categories.items()
    }

    report = {"summary": summary, "results": results}

    # Persist
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = out_path / f"eval_{ts}.json"
    with open(file_path, "w") as f:
        json.dump(report, f, indent=2)

    report["output_file"] = str(file_path)
    return report
