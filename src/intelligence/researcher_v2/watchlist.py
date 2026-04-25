"""Watchlist evaluator — fires `next_action` insights when adapter
predicates evaluate true.

v1 supports a tiny expression DSL of the shape:
  expr:doc.<field> - now < <N>d
This is intentionally narrow — adapter authors can always declare a
plain insight via the researcher path for richer logic.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from src.intelligence.adapters.schema import Adapter

logger = logging.getLogger(__name__)


@dataclass
class WatchlistFiring:
    watchlist_id: str
    document_id: str
    fires_insight_type: str
    description: str


_DATE_DELTA_RX = re.compile(
    r"expr:\s*doc\.([a-zA-Z_][a-zA-Z_0-9]*)\s*-\s*now\s*<\s*(\d+)d\s*$"
)


def _parse_iso(value: Any):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def evaluate_watchlists(*, adapter: Adapter, documents: List[Dict[str, Any]]) -> List[WatchlistFiring]:
    fired: List[WatchlistFiring] = []
    now = datetime.now(tz=timezone.utc)
    for w in adapter.watchlists:
        m = _DATE_DELTA_RX.match(w.eval.strip())
        if not m:
            logger.debug("unsupported watchlist expr: %s", w.eval)
            continue
        field, days = m.group(1), int(m.group(2))
        threshold = timedelta(days=days)
        for doc in documents:
            fields = doc.get("fields") or {}
            dt = _parse_iso(fields.get(field))
            if dt is None:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if (dt - now) < threshold and (dt - now) >= timedelta(0):
                fired.append(WatchlistFiring(
                    watchlist_id=w.id,
                    document_id=str(doc.get("document_id", "")),
                    fires_insight_type=w.fires_insight_type,
                    description=w.description,
                ))
    return fired
