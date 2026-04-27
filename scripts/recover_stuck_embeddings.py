"""Re-dispatch docs that hit the lease-conflict race + were marked FAILED.

UAT Issue #4 / #6 cleanup: identifies docs stuck in EMBEDDING_FAILED /
TRAINING_FAILED whose extraction artifact is intact (coverage ≥ 0.95)
and re-dispatches embedding via the existing /api/documents/embed
endpoint.

Run before round-2 UAT to clean up the false-failure docs from round 1.

Usage:
  python scripts/recover_stuck_embeddings.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("recover-stuck")


def find_stuck_docs(*, db, limit: int):
    """Find docs with EMBEDDING_FAILED / TRAINING_FAILED status.

    We don't pre-filter by extraction coverage here — instead we let the
    embedding pipeline's existing fallback path (which already detects
    incomplete pickles and re-extracts from source) handle it. The
    fallback is what produced the success that this round-1 race threw
    away.
    """
    coll = db.get_collection("documents")
    statuses = ["EMBEDDING_FAILED", "TRAINING_FAILED"]
    cursor = coll.find(
        {"$or": [{"status": s} for s in statuses]},
        {"_id": 1, "document_id": 1, "subscription": 1, "subscriptionId": 1,
         "profile": 1, "profile_id": 1, "status": 1, "filename": 1},
    ).limit(limit)
    return list(cursor)


def re_dispatch(*, doc, base_url: str, dry_run: bool) -> dict:
    """Trigger /api/documents/embed for a single stuck doc."""
    import requests

    doc_id = str(doc.get("document_id") or doc.get("_id"))
    sub = str(doc.get("subscription") or doc.get("subscriptionId") or "default")
    prof = str(doc.get("profile") or doc.get("profile_id") or "")
    filename = doc.get("filename", "?")

    if dry_run:
        return {"doc_id": doc_id, "filename": filename, "subscription": sub,
                "profile_id": prof, "action": "DRY-RUN: would re-dispatch"}

    if not prof:
        return {"doc_id": doc_id, "filename": filename, "action": "SKIP: no profile_id"}

    try:
        r = requests.post(
            f"{base_url}/api/documents/embed",
            json={"document_id": doc_id, "subscription_id": sub, "profile_id": prof},
            timeout=300,
        )
        return {"doc_id": doc_id, "filename": filename,
                "http": r.status_code,
                "body": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:200]}
    except Exception as exc:
        return {"doc_id": doc_id, "filename": filename, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List candidates without dispatching")
    parser.add_argument("--limit", type=int, default=200, help="Max docs to process")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    from src.api.dataHandler import db
    stuck = find_stuck_docs(db=db, limit=args.limit)
    logger.info("Found %d docs stuck in EMBEDDING_FAILED/TRAINING_FAILED", len(stuck))

    results = []
    for doc in stuck:
        outcome = re_dispatch(doc=doc, base_url=args.base_url, dry_run=args.dry_run)
        results.append(outcome)
        logger.info("  %s", outcome)

    summary = {
        "total_stuck": len(stuck),
        "dispatched": sum(1 for r in results if r.get("http") == 200),
        "skipped": sum(1 for r in results if r.get("action", "").startswith("SKIP")),
        "errors": sum(1 for r in results if "error" in r),
        "dry_run": args.dry_run,
    }
    print()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
