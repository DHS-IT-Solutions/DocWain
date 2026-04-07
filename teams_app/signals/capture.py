"""Learning signal capture for finetuning from Teams interactions."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


def _sanitize_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Strip document content from sources — only keep titles/references."""
    return [{"title": s.get("title", "")} for s in sources if s.get("title")]


class SignalCapture:
    """Appends query/response pairs to JSONL files for finetuning."""

    def __init__(self, signals_dir: str):
        self.signals_dir = signals_dir
        os.makedirs(signals_dir, exist_ok=True)

    def record(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        grounded: bool,
        context_found: bool,
        signal: str,
        tenant_id: str,
        pipeline: str = "",
        latency_ms: int = 0,
    ) -> None:
        """Record a learning signal to the appropriate JSONL file."""
        entry = {
            "query": query,
            "response": response,
            "sources": _sanitize_sources(sources),
            "grounded": grounded,
            "context_found": context_found,
            "source": "teams",
            "signal": signal,
            "tenant_id": tenant_id,
            "pipeline": pipeline,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if signal == "positive":
            filename = "high_quality.jsonl"
        else:
            filename = "finetune_buffer.jsonl"

        path = os.path.join(self.signals_dir, filename)
        line = json.dumps(entry, ensure_ascii=False) + "\n"

        with _write_lock:
            with open(path, "a") as f:
                f.write(line)

        logger.debug("Recorded %s signal to %s", signal, filename)
