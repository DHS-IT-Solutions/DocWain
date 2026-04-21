"""Trace loader for SME synthesis + query JSONL blobs.

Reads the Azure Blob paths defined in spec Section 11 (and emitted by the
Phase 1/2 trace writers in ``src/intelligence/sme/trace.py``)::

    sme_traces/synthesis/{sub}/{prof}/{synthesis_id}.jsonl
    sme_traces/queries/{sub}/{prof}/{YYYY-MM-DD}/{query_id}.jsonl

The loader depends on two callables injected at construction::

    list_blobs(prefix: str) -> Iterable[str]
    read_blob(name: str) -> str

This indirection keeps the loader testable without Azure SDK mocks. The
production CLI wires these to :mod:`src.storage.azure_blob_client` at startup.

Implementation detail: we narrow bare ``Exception`` catches to
``ValidationError | ValueError | TypeError | KeyError`` where the error is
data-quality; logging still emits the traceback so operators can find the bad
blob, but a single malformed line never aborts the monthly run.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from scripts.sme_patterns.schema import (
    BuilderTrace,
    QueryFeedback,
    QueryRun,
    SynthesisRun,
    VerifierDrop,
)

logger = logging.getLogger(__name__)

_DATE_DIR = re.compile(r"/(\d{4}-\d{2}-\d{2})/")
_SYNTH_PREFIX = "sme_traces/synthesis/"
_QUERY_PREFIX = "sme_traces/queries/"

# Sentinel timestamp used when a crashed synthesis trace lacks any
# ``started_at`` event — loader skips these rather than propagate a default.
_EPOCH_SENTINEL = datetime(1, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class TraceWindow:
    start: datetime
    end: datetime

    def contains(self, ts: datetime) -> bool:
        # Normalize tz awareness so a mix of naive (fixtures) and aware
        # (production) timestamps never trips a TypeError.
        if ts.tzinfo is not None and self.start.tzinfo is None:
            ts = ts.replace(tzinfo=None)
        elif ts.tzinfo is None and self.start.tzinfo is not None:
            ts = ts.replace(tzinfo=self.start.tzinfo)
        return self.start <= ts <= self.end


def _parse_iso(value: str) -> datetime:
    # Accept both ``2026-04-05T10:00:00`` and ``...+00:00`` / ``...Z``.
    cleaned = value.replace("Z", "+00:00")
    if "+" in cleaned:
        cleaned = cleaned.split("+")[0]
    return datetime.fromisoformat(cleaned)


def parse_synth_jsonl(text: str) -> SynthesisRun:
    """Assemble a SynthesisRun from a synthesis trace blob's JSONL content."""
    started: datetime | None = None
    completed: datetime | None = None
    subscription_id = profile_id = profile_domain = ""
    synthesis_id = ""
    adapter_version = ""
    adapter_content_hash = ""
    per_builder: dict[str, BuilderTrace] = {}
    drops: list[VerifierDrop] = []

    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("skipping malformed synth trace line")
            continue
        if not isinstance(ev, dict):
            continue
        kind = ev.get("event")
        if kind == "synthesis_started":
            synthesis_id = ev.get("synthesis_id", synthesis_id)
            subscription_id = ev.get("subscription_id", subscription_id)
            profile_id = ev.get("profile_id", profile_id)
            profile_domain = ev.get("profile_domain", profile_domain)
            adapter_version = ev.get("adapter_version", adapter_version)
            adapter_content_hash = ev.get("adapter_content_hash", adapter_content_hash)
            ts = ev.get("started_at")
            if ts:
                try:
                    started = _parse_iso(ts)
                except ValueError:
                    logger.warning("synth trace has unparseable started_at %r", ts)
        elif kind == "synthesis_completed":
            ts = ev.get("completed_at")
            if ts:
                try:
                    completed = _parse_iso(ts)
                except ValueError:
                    logger.warning("synth trace has unparseable completed_at %r", ts)
        elif kind == "builder_complete":
            bn = ev.get("builder", "unknown")
            try:
                per_builder[bn] = BuilderTrace(
                    builder_name=bn,
                    items_produced=int(ev.get("items_produced", 0)),
                    items_persisted=int(ev.get("items_persisted", 0)),
                    duration_ms=(
                        float(ev["duration_ms"])
                        if ev.get("duration_ms") is not None
                        else None
                    ),
                    errors=list(ev.get("errors", []) or []),
                )
            except (ValidationError, ValueError, TypeError):
                logger.warning("skipping malformed builder_complete event for %s", bn)
        elif kind == "verifier_drop":
            try:
                drops.append(
                    VerifierDrop(
                        item_id=str(ev.get("item_id", "")),
                        builder=str(ev.get("builder", "")),
                        reason_code=str(ev.get("reason_code", "unknown")),
                        detail=str(ev.get("detail", "")),
                    )
                )
            except (ValidationError, ValueError, TypeError):
                logger.warning("skipping malformed verifier_drop event")
        else:
            # Unknown event kinds tolerated — trace writers may add new ones.
            continue

    if started is None:
        started = _EPOCH_SENTINEL

    return SynthesisRun(
        subscription_id=subscription_id,
        profile_id=profile_id,
        synthesis_id=synthesis_id,
        started_at=started,
        completed_at=completed,
        adapter_version=adapter_version,
        adapter_content_hash=adapter_content_hash,
        profile_domain=profile_domain,
        per_builder=per_builder,
        verifier_drops=drops,
    )


def parse_query_jsonl(text: str) -> Iterator[QueryRun]:
    """Yield QueryRun records for every well-formed ``query_complete`` event."""
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("skipping malformed query trace line")
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("event") != "query_complete":
            continue
        try:
            fb_raw = ev.get("feedback")
            feedback = QueryFeedback(**fb_raw) if isinstance(fb_raw, dict) else None
            yield QueryRun(
                subscription_id=str(ev.get("subscription_id", "")),
                profile_id=str(ev.get("profile_id", "")),
                profile_domain=str(ev.get("profile_domain", "")),
                query_id=str(ev.get("query_id", "")),
                query_text=str(ev.get("query_text", "")),
                query_fingerprint=str(ev.get("query_fingerprint", "")),
                intent=str(ev.get("intent", "unknown")),
                format_hint=ev.get("format_hint"),
                adapter_version=str(ev.get("adapter_version", "")),
                adapter_persona_role=str(ev.get("adapter_persona_role", "")),
                retrieval_layers={
                    str(k): int(v)
                    for k, v in (ev.get("retrieval_layers", {}) or {}).items()
                },
                pack_tokens=int(ev.get("pack_tokens", 0)),
                reasoner_prompt_hash=str(ev.get("reasoner_prompt_hash", "")),
                response_len_tokens=int(ev.get("response_len_tokens", 0)),
                citation_verifier_drops=int(ev.get("citation_verifier_drops", 0)),
                honest_compact_fallback=bool(ev.get("honest_compact_fallback", False)),
                url_present=bool(ev.get("url_present", False)),
                url_fetch_ok=ev.get("url_fetch_ok"),
                timing_ms={
                    str(k): float(v)
                    for k, v in (ev.get("timing_ms", {}) or {}).items()
                },
                feedback=feedback,
                captured_at=_parse_iso(ev.get("captured_at", "1970-01-01T00:00:00")),
            )
        except (ValidationError, ValueError, TypeError, KeyError):
            logger.warning("skipping query event that failed to validate")
            continue


class TraceLoader:
    """Pulls SynthesisRun and QueryRun records from Azure Blob.

    The two callables abstract the storage — production wires them to
    :mod:`src.storage.azure_blob_client`; tests pass in-memory dicts.
    """

    def __init__(
        self,
        *,
        list_blobs: Callable[[str], Iterable[str]],
        read_blob: Callable[[str], str],
    ) -> None:
        self._list = list_blobs
        self._read = read_blob

    def iter_synthesis_runs(self, window: TraceWindow) -> Iterator[SynthesisRun]:
        for name in self._list(_SYNTH_PREFIX):
            try:
                text = self._read(name)
            except (OSError, ValueError, KeyError):
                logger.exception("failed to read synth blob %s", name)
                continue
            try:
                run = parse_synth_jsonl(text)
            except (ValidationError, ValueError, TypeError):
                logger.exception("failed to parse synth blob %s", name)
                continue
            if run.started_at == _EPOCH_SENTINEL:
                continue
            if not window.contains(run.started_at):
                continue
            yield run

    def iter_query_runs(self, window: TraceWindow) -> Iterator[QueryRun]:
        for name in self._list(_QUERY_PREFIX):
            # Fast-reject by date prefix in the path to avoid downloading
            # blobs obviously outside the window.
            m = _DATE_DIR.search(name)
            if m:
                try:
                    day = datetime.strptime(m.group(1), "%Y-%m-%d")
                    if day.date() < window.start.date() or day.date() > window.end.date():
                        continue
                except ValueError:
                    pass
            try:
                text = self._read(name)
            except (OSError, ValueError, KeyError):
                logger.exception("failed to read query blob %s", name)
                continue
            for q in parse_query_jsonl(text):
                if window.contains(q.captured_at):
                    yield q
