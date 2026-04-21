"""Tests for the SME input-hash short-circuit module (user Task 13).

Contract pinned here:

* :func:`compute_input_hash` is stable under chunk reorder.
* Adding, removing, or mutating any chunk changes the hash.
* Mutating the adapter version or content hash changes the hash.
* :func:`input_hash_unchanged` reads the profile's
  ``sme_last_input_hash`` via :func:`get_profile_record`.
* First-run (no stored hash) always returns ``False`` so synthesis fires.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.intelligence.sme.input_hash import (
    ChunkRef,
    build_inputs_from_chunks,
    compute_input_hash,
    compute_input_hash_for_profile,
    input_hash_unchanged,
)


def _inputs(triples, adapter_hash, adapter_version="1.0.0"):
    return build_inputs_from_chunks(
        subscription_id="s",
        profile_id="p",
        chunks=[{"doc_id": d, "chunk_id": c, "text": t} for d, c, t in triples],
        adapter_version=adapter_version,
        adapter_content_hash=adapter_hash,
    )


def test_hash_stable_under_chunk_reorder() -> None:
    a = compute_input_hash(
        _inputs([("d1", "c1", "t1"), ("d2", "c3", "t3")], "h1")
    )
    b = compute_input_hash(
        _inputs([("d2", "c3", "t3"), ("d1", "c1", "t1")], "h1")
    )
    assert a == b


def test_hash_changes_on_new_chunk() -> None:
    a = compute_input_hash(_inputs([("d1", "c1", "t1")], "h1"))
    b = compute_input_hash(
        _inputs([("d1", "c1", "t1"), ("d2", "c2", "t2")], "h1")
    )
    assert a != b


def test_hash_changes_on_chunk_text_mutation() -> None:
    a = compute_input_hash(_inputs([("d1", "c1", "t1")], "h1"))
    b = compute_input_hash(_inputs([("d1", "c1", "t1_MUT")], "h1"))
    assert a != b


def test_hash_changes_on_chunk_id_mutation() -> None:
    a = compute_input_hash(_inputs([("d1", "c1", "t1")], "h1"))
    b = compute_input_hash(_inputs([("d1", "c9", "t1")], "h1"))
    assert a != b


def test_hash_changes_on_adapter_hash_mutation() -> None:
    a = compute_input_hash(_inputs([("d1", "c1", "t1")], "h1"))
    b = compute_input_hash(_inputs([("d1", "c1", "t1")], "h2"))
    assert a != b


def test_hash_changes_on_adapter_version_mutation() -> None:
    a = compute_input_hash(
        _inputs([("d1", "c1", "t1")], "h1", adapter_version="1.0.0")
    )
    b = compute_input_hash(
        _inputs([("d1", "c1", "t1")], "h1", adapter_version="1.0.1")
    )
    assert a != b


def test_hash_stable_empty_profile() -> None:
    a = compute_input_hash(_inputs([], "h1"))
    b = compute_input_hash(_inputs([], "h1"))
    assert a == b


def test_convenience_wrapper_matches_direct_call() -> None:
    direct = compute_input_hash(_inputs([("d1", "c1", "t1")], "h1"))
    viawrap = compute_input_hash_for_profile(
        subscription_id="s",
        profile_id="p",
        chunks=[{"doc_id": "d1", "chunk_id": "c1", "text": "t1"}],
        adapter_version="1.0.0",
        adapter_content_hash="h1",
    )
    assert direct == viawrap


def test_accepts_chunk_ref_instances_directly() -> None:
    via_dicts = compute_input_hash_for_profile(
        subscription_id="s",
        profile_id="p",
        chunks=[{"doc_id": "d1", "chunk_id": "c1", "text": "t"}],
        adapter_version="v",
        adapter_content_hash="h",
    )
    via_refs = compute_input_hash_for_profile(
        subscription_id="s",
        profile_id="p",
        chunks=[ChunkRef(doc_id="d1", chunk_id="c1", text="t")],
        adapter_version="v",
        adapter_content_hash="h",
    )
    assert via_dicts == via_refs


def test_missing_doc_id_raises() -> None:
    with pytest.raises(KeyError):
        compute_input_hash_for_profile(
            subscription_id="s",
            profile_id="p",
            chunks=[{"chunk_id": "c1", "text": "t"}],
            adapter_version="v",
            adapter_content_hash="h",
        )


def test_input_hash_unchanged_returns_false_on_first_run() -> None:
    with patch(
        "src.api.document_status.get_profile_record",
        return_value=None,
    ):
        assert (
            input_hash_unchanged(
                subscription_id="s",
                profile_id="p",
                current_hash="anything",
            )
            is False
        )


def test_input_hash_unchanged_matches_stored_hash() -> None:
    with patch(
        "src.api.document_status.get_profile_record",
        return_value={"sme_last_input_hash": "deadbeef"},
    ):
        assert (
            input_hash_unchanged(
                subscription_id="s",
                profile_id="p",
                current_hash="deadbeef",
            )
            is True
        )


def test_input_hash_unchanged_detects_mismatch() -> None:
    with patch(
        "src.api.document_status.get_profile_record",
        return_value={"sme_last_input_hash": "deadbeef"},
    ):
        assert (
            input_hash_unchanged(
                subscription_id="s",
                profile_id="p",
                current_hash="NEW_HASH",
            )
            is False
        )


def test_input_hash_unchanged_false_when_stored_is_empty_string() -> None:
    with patch(
        "src.api.document_status.get_profile_record",
        return_value={"sme_last_input_hash": ""},
    ):
        assert (
            input_hash_unchanged(
                subscription_id="s",
                profile_id="p",
                current_hash="",
            )
            is False
        )
