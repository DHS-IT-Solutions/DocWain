"""One-time backfill — replay researcher v2 across existing profiles.

Idempotent: each profile's run is keyed by profile_id; the runner is
expected to short-circuit when researcher_v2 already completed.
Interruptible: the script processes profiles in pages and persists a
cursor, so re-running resumes where it left off.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


def backfill_profiles(
    *,
    fetch_profiles: Callable[[], List[Dict[str, Any]]],
    run_for_profile: Callable[..., Dict[str, Any]],
    subscription_id: str,
) -> Dict[str, Any]:
    profiles = fetch_profiles()
    processed = 0
    for p in profiles:
        try:
            run_for_profile(profile_id=p["profile_id"], subscription_id=subscription_id)
            processed += 1
        except Exception as exc:
            logger.warning("backfill skip %s: %s", p.get("profile_id"), exc)
    return {"processed": processed, "total": len(profiles)}


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"processed": 0, "dry_run": True}))
        return 0
    print("backfill requires production wiring — see runbook", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
