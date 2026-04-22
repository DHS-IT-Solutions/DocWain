"""Incremental synthesis — deterministic input-hash short-circuit.

Phase 2 Task 15 (plan) / user Task 13: re-running synthesis on unchanged
inputs is wasteful. This module computes a stable hash of the synthesis
inputs — sorted ``(doc_id, chunk_id, text_hash)`` triples plus adapter
version + content hash — and lets :func:`finalize_training_for_doc`
short-circuit when the current hash equals the profile's
``sme_last_input_hash``.

The hash is persisted on the profile record (control-plane only, per the
storage separation rule) via the ``sme_last_input_hash`` allowlisted key
in ``src/api/document_status.py``. Adapter version + content hash live
directly on the :class:`Adapter` model returned by ``AdapterLoader.load``
per ERRATA §1, so we never re-open the YAML from disk here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class ChunkRef:
    """Minimal chunk descriptor the hash needs.

    ``text`` is kept as a plain string so callers can pass the raw chunk
    payload; we hash it internally. ``doc_id`` + ``chunk_id`` are the
    sort keys that make the hash reorder-stable.
    """

    doc_id: str
    chunk_id: str
    text: str


@dataclass(frozen=True)
class InputHashInputs:
    """Bundle of everything the hash consumes.

    ``adapter_version`` and ``adapter_content_hash`` come straight off
    the loaded :class:`Adapter`. ``chunks`` is the full set of chunks in
    the profile (one :class:`ChunkRef` per chunk). Ordering within
    ``chunks`` is irrelevant — the hash normalises by sorting on
    ``(doc_id, chunk_id)`` before hashing, so a Mongo-order swap does
    not invalidate the cache.
    """

    subscription_id: str
    profile_id: str
    chunks: tuple[ChunkRef, ...]
    adapter_version: str
    adapter_content_hash: str
    extras: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def compute_input_hash(inputs: InputHashInputs) -> str:
    """Return a deterministic SHA-256 hex hash of the synthesis inputs.

    Stable under chunk-order permutation. Changes when any of: chunk
    text, chunk identity (doc_id/chunk_id), adapter version, adapter
    content hash, or an extra ``(key, value)`` pair mutates.
    """
    h = hashlib.sha256()
    for chunk in sorted(inputs.chunks, key=lambda c: (c.doc_id, c.chunk_id)):
        line = f"{chunk.doc_id}|{chunk.chunk_id}|{_hash_text(chunk.text)}\n"
        h.update(line.encode("utf-8"))
    h.update(
        (
            f"adapter:{inputs.adapter_version}|{inputs.adapter_content_hash}"
        ).encode("utf-8")
    )
    for key, value in inputs.extras:
        h.update(f"extra:{key}={value}\n".encode("utf-8"))
    return h.hexdigest()


def build_inputs_from_chunks(
    *,
    subscription_id: str,
    profile_id: str,
    chunks: Iterable[dict[str, Any] | ChunkRef],
    adapter_version: str,
    adapter_content_hash: str,
    extras: tuple[tuple[str, str], ...] = (),
) -> InputHashInputs:
    """Helper: build :class:`InputHashInputs` from an iterable of dicts.

    Accepts either :class:`ChunkRef` instances directly or plain dicts with
    ``doc_id`` / ``chunk_id`` / ``text`` keys. Missing keys raise
    ``KeyError`` so a caller that forgot to pass a field fails loudly.
    """
    refs: list[ChunkRef] = []
    for entry in chunks:
        if isinstance(entry, ChunkRef):
            refs.append(entry)
            continue
        refs.append(
            ChunkRef(
                doc_id=str(entry["doc_id"]),
                chunk_id=str(entry["chunk_id"]),
                text=str(entry.get("text", "")),
            )
        )
    return InputHashInputs(
        subscription_id=subscription_id,
        profile_id=profile_id,
        chunks=tuple(refs),
        adapter_version=adapter_version,
        adapter_content_hash=adapter_content_hash,
        extras=tuple(extras),
    )


def input_hash_unchanged(
    *,
    subscription_id: str,
    profile_id: str,
    current_hash: str,
) -> bool:
    """Return True iff the profile's stored ``sme_last_input_hash``
    equals ``current_hash``.

    When the profile record is missing (first run for this profile) we
    return False so synthesis always fires on the first call. Callers
    must persist ``current_hash`` (via ``update_profile_record``) after
    a successful synthesis so the next run can short-circuit.
    """
    from src.api.document_status import get_profile_record

    rec = get_profile_record(subscription_id, profile_id) or {}
    stored = rec.get("sme_last_input_hash")
    return bool(stored) and stored == current_hash


def compute_input_hash_for_profile(
    *,
    subscription_id: str,
    profile_id: str,
    chunks: Iterable[dict[str, Any] | ChunkRef],
    adapter_version: str,
    adapter_content_hash: str,
    extras: tuple[tuple[str, str], ...] = (),
) -> str:
    """One-call convenience wrapper used by :func:`finalize_training_for_doc`.

    Wraps :func:`build_inputs_from_chunks` + :func:`compute_input_hash`.
    """
    return compute_input_hash(
        build_inputs_from_chunks(
            subscription_id=subscription_id,
            profile_id=profile_id,
            chunks=chunks,
            adapter_version=adapter_version,
            adapter_content_hash=adapter_content_hash,
            extras=extras,
        )
    )
