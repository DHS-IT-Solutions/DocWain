"""Blob-backed JSONL trace writers (spec §11).

Task 7 ships :class:`SynthesisTraceWriter` — paths
``sme_traces/synthesis/{sub}/{prof}/{synthesis_id}.jsonl``. Task 8 adds
:class:`QueryTraceWriter` for per-query traces.

Canonical method (ERRATA §5): ``.append(event)``. ``.record`` is retained as a
back-compat alias to ease migration; new code should use ``.append``.
"""
from __future__ import annotations

import json
from typing import Any, Protocol


class TraceBlobAppender(Protocol):
    """Blob append-path used by every trace writer. Implementations call the
    Azure Append Blob API or an equivalent durable append-log backend. No
    per-append timeout is set at the DocWain layer (spec §3 inv. 8); the
    underlying Blob SDK enforces its own safety limits.
    """

    def append(self, path: str, line: str) -> None: ...


class _Base:
    """Shared open/append/close state for every trace writer.

    Subclasses set ``self._path`` inside their own ``open(...)`` method; the
    path template is the only thing that differs between synthesis and query
    traces. Appending before ``open()`` raises to keep trace misuse loud.
    """

    def __init__(self, appender: TraceBlobAppender) -> None:
        self._a = appender
        self._path: str | None = None

    def append(self, event: dict[str, Any]) -> None:
        """Append one JSON line to the current trace path. Must be called
        between ``open(...)`` and ``close()`` (ERRATA §5 canonical method)."""
        if self._path is None:
            raise RuntimeError("Trace writer not open; call open() first")
        line = json.dumps(event, default=str, ensure_ascii=False) + "\n"
        self._a.append(self._path, line)

    # Back-compat alias — ERRATA §5 keeps ``.record`` as a deprecated alias.
    record = append

    def close(self) -> None:
        """Detach from the current path; subsequent ``append`` calls error."""
        self._path = None


class SynthesisTraceWriter(_Base):
    """Trace writer for ingestion-time artifact synthesis runs.

    Path: ``sme_traces/synthesis/{subscription_id}/{profile_id}/{synthesis_id}.jsonl``.
    One writer instance per synthesis run; emitted by the synthesizer
    orchestrator (Phase 2) to record builder start/end/verdict events.
    """

    def open(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        synthesis_id: str,
    ) -> None:
        self._path = (
            f"sme_traces/synthesis/"
            f"{subscription_id}/{profile_id}/{synthesis_id}.jsonl"
        )
