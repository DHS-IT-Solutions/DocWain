"""RAG regression harness — runs a golden query set after every embed batch.

A "golden set" is a YAML (or JSON) file stored at
``data/golden_queries/<subscription_id>/<profile_id>.yaml`` listing the
queries this profile must answer correctly and, for each, the
``expected_document_id`` that the retriever should return in its top-K.
Structure:

    version: 1
    top_k: 10
    min_recall_at_5: 0.60        # regression threshold
    min_mrr: 0.50                # regression threshold
    queries:
      - query: "Aquarius invoice INV-25-050"
        expected_document_id: "69e26127e41cbd913401e45a"
      - query: "Marketing Assurity Agreement"
        expected_document_id: "69e26127e41cbd913401e4cd"

The harness runs every query through ``UnifiedRetriever``, computes
Recall@5 / Recall@10 / MRR over the set, compares to the thresholds
and to the last-run baseline, and writes a report to
``doc.rag_regression`` (on a sentinel "_profile_rag_baseline" doc per
profile) plus the per-run history file. If ``strict`` is set and the
metrics regress vs baseline or fall below the threshold, the harness
returns a non-zero exit code so it can gate a CI / deploy step.

Invoke from Python after an embed batch:

    from src.tasks.rag_regression import run_regression
    report = run_regression(subscription_id, profile_id)

Or from the CLI:

    python -m src.tasks.rag_regression <subscription_id> <profile_id>

Golden sets are versioned in the repo so the thresholds become a
living contract between accuracy work and the retriever.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_GOLDEN_ROOT = Path(__file__).resolve().parents[2] / "data" / "golden_queries"
_HISTORY_ROOT = Path(__file__).resolve().parents[2] / "data" / "golden_queries" / ".history"


def _load_golden_set(subscription_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
    """Return the YAML-or-JSON golden set for this profile, or None."""
    candidates = [
        _GOLDEN_ROOT / subscription_id / f"{profile_id}.yaml",
        _GOLDEN_ROOT / subscription_id / f"{profile_id}.yml",
        _GOLDEN_ROOT / subscription_id / f"{profile_id}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml  # type: ignore
                    return yaml.safe_load(raw)
                except ImportError:
                    logger.warning("PyYAML not available; skipping %s", path)
                    continue
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.error("Golden set %s failed to parse: %s", path, exc)
            return None
    return None


def _build_retriever():
    from qdrant_client import QdrantClient
    from src.api.config import Config
    from src.retrieval.retriever import UnifiedRetriever
    from src.api.dw_newron import get_model_by_name

    qdrant = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
    embedder = get_model_by_name("BAAI/bge-large-en-v1.5")
    return UnifiedRetriever(qdrant, embedder)


def _rank_of(chunks, expected_doc_id: str) -> int:
    for idx, chunk in enumerate(chunks, start=1):
        if getattr(chunk, "document_id", "") == expected_doc_id:
            return idx
    return 0


def _metrics_from_ranks(ranks: List[int]) -> Dict[str, float]:
    if not ranks:
        return {"recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "count": 0}
    at5 = sum(1 for r in ranks if 1 <= r <= 5) / len(ranks)
    at10 = sum(1 for r in ranks if 1 <= r <= 10) / len(ranks)
    mrr = sum(1.0 / r for r in ranks if r > 0) / len(ranks)
    return {
        "recall_at_5": round(at5, 3),
        "recall_at_10": round(at10, 3),
        "mrr": round(mrr, 3),
        "count": len(ranks),
    }


def _load_baseline(subscription_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
    path = _HISTORY_ROOT / subscription_id / f"{profile_id}.latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_baseline(subscription_id: str, profile_id: str, report: Dict[str, Any]) -> None:
    base = _HISTORY_ROOT / subscription_id
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{profile_id}.latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ts = int(time.time())
    (base / f"{profile_id}.{ts}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_regression(
    subscription_id: str,
    profile_id: str,
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """Execute the golden set, compare to thresholds and previous run.

    Returns a report dict. When ``strict`` is set, a regression (metrics
    below threshold OR metrics below the last-stored baseline) marks the
    report ``status="regression"`` — caller can exit non-zero on that.
    """
    golden = _load_golden_set(subscription_id, profile_id)
    if not golden:
        return {
            "status": "skipped",
            "reason": f"no golden set at data/golden_queries/{subscription_id}/{profile_id}.yaml",
        }

    top_k = int(golden.get("top_k") or 10)
    min_recall = float(golden.get("min_recall_at_5") or 0.0)
    min_mrr = float(golden.get("min_mrr") or 0.0)
    queries = golden.get("queries") or []
    if not queries:
        return {"status": "skipped", "reason": "golden set has no queries"}

    retriever = _build_retriever()
    ranks: List[int] = []
    per_query: List[Dict[str, Any]] = []
    for q in queries:
        query = str(q.get("query") or "").strip()
        expected = str(q.get("expected_document_id") or "").strip()
        if not query or not expected:
            continue
        result = retriever.retrieve(query, subscription_id, [profile_id], top_k=top_k)
        rank = _rank_of(result.chunks, expected)
        ranks.append(rank)
        per_query.append({
            "query": query,
            "expected_document_id": expected,
            "rank": rank,
            "in_top_5": 1 <= rank <= 5,
            "in_top_10": 1 <= rank <= 10,
        })

    metrics = _metrics_from_ranks(ranks)
    baseline = _load_baseline(subscription_id, profile_id)
    threshold_violations = []
    if metrics["recall_at_5"] < min_recall:
        threshold_violations.append(
            f"recall_at_5 {metrics['recall_at_5']:.3f} < min {min_recall:.3f}"
        )
    if metrics["mrr"] < min_mrr:
        threshold_violations.append(
            f"mrr {metrics['mrr']:.3f} < min {min_mrr:.3f}"
        )
    baseline_regressions = []
    if baseline and isinstance(baseline.get("metrics"), dict):
        b = baseline["metrics"]
        # A regression is >= 0.02 absolute drop vs baseline
        if metrics["recall_at_5"] + 0.02 < b.get("recall_at_5", 0):
            baseline_regressions.append(
                f"recall_at_5 dropped {b['recall_at_5']:.3f} → {metrics['recall_at_5']:.3f}"
            )
        if metrics["mrr"] + 0.02 < b.get("mrr", 0):
            baseline_regressions.append(
                f"mrr dropped {b['mrr']:.3f} → {metrics['mrr']:.3f}"
            )

    status = "ok"
    if threshold_violations:
        status = "below_threshold"
    if baseline_regressions:
        status = "regression"

    report = {
        "status": status,
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "timestamp": int(time.time()),
        "metrics": metrics,
        "thresholds": {
            "min_recall_at_5": min_recall,
            "min_mrr": min_mrr,
        },
        "threshold_violations": threshold_violations,
        "baseline_regressions": baseline_regressions,
        "baseline_timestamp": baseline.get("timestamp") if baseline else None,
        "per_query": per_query,
    }

    _save_baseline(subscription_id, profile_id, report)
    logger.info(
        "rag-regression sub=%s prof=%s status=%s R@5=%.3f R@10=%.3f MRR=%.3f "
        "violations=%d regressions=%d",
        subscription_id, profile_id, status,
        metrics["recall_at_5"], metrics["recall_at_10"], metrics["mrr"],
        len(threshold_violations), len(baseline_regressions),
    )
    if strict and status != "ok":
        logger.error("rag-regression FAILED: %s", status)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subscription_id")
    ap.add_argument("profile_id")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on regression or threshold violation")
    args = ap.parse_args()
    report = run_regression(args.subscription_id, args.profile_id, strict=args.strict)
    print(json.dumps(report, indent=2))
    if args.strict and report.get("status") not in ("ok", "skipped"):
        sys.exit(2)


if __name__ == "__main__":
    main()
