"""Blob-backed JSONL trace writers (spec §11).

* :class:`SynthesisTraceWriter` — ingestion-time synthesis runs;
  path ``sme_traces/synthesis/{sub}/{prof}/{synthesis_id}.jsonl``.
* :class:`QueryTraceWriter` — per-query retrieval/generation traces;
  path ``sme_traces/queries/{sub}/{prof}/{YYYY-MM-DD}/{query_id}.jsonl``.

Canonical method (ERRATA §5): ``.append(event)``. ``.record`` is retained as a
back-compat alias to ease migration; new code should use ``.append``.

Both writers capture their path template at ``open(...)`` time. For the query
writer the date partition is set at open — long-running queries that cross
midnight still write to the opening day's directory, so Phase 3 retrieval
tooling can locate every event for a given query by listing one directory.

Per project constraints we never call ``datetime.utcnow()``; the default
``now`` callable returns ``datetime.now(timezone.utc)``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Protocol


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


def _utc_now() -> datetime:
    """Default ``now`` callable for :class:`QueryTraceWriter`. Returns a
    UTC-aware ``datetime``; never uses the deprecated ``datetime.utcnow()``
    (project constraint)."""
    return datetime.now(timezone.utc)


class QueryTraceWriter(_Base):
    """Trace writer for per-query retrieval + generation events.

    Path: ``sme_traces/queries/{sub}/{prof}/{YYYY-MM-DD}/{query_id}.jsonl``.

    The date partition is captured at ``open(...)`` time (not per append) so a
    query that crosses midnight keeps all its events in a single directory.
    """

    def __init__(
        self,
        appender: TraceBlobAppender,
        *,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        super().__init__(appender)
        self._now = now

    def open(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        query_id: str,
    ) -> None:
        day = self._now().strftime("%Y-%m-%d")
        self._path = (
            f"sme_traces/queries/"
            f"{subscription_id}/{profile_id}/{day}/{query_id}.jsonl"
        )
