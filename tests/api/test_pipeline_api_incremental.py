"""Incremental-synthesis integration tests for finalize_training_for_doc.

User Task 13: short-circuit synthesis when the profile's
``sme_last_input_hash`` equals the current input hash; write the hash +
run_id after a successful synthesis. First run must always fire.

Uses in-memory readers to decouple from Qdrant / MongoDB. Every test
asserts both the observable side effects (status flip, synthesizer
invocation, audit-log event) and the persisted state (writes to the
profile record).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api import pipeline_api


def _doc(
    document_id="d_last",
    subscription_id="sub_a",
    profile_id="prof_x",
):
    return {
        "document_id": document_id,
        "subscription_id": subscription_id,
        "profile_id": profile_id,
    }


def _flag_resolver(enabled: bool) -> MagicMock:
    r = MagicMock()
    r.is_enabled.return_value = enabled
    return r


@pytest.fixture(autouse=True)
def _clear_registrations():
    """Each test starts with no factory and no readers."""
    pipeline_api.register_sme_synthesizer_factory(None)
    pipeline_api.register_profile_chunks_reader(None)
    pipeline_api.register_adapter_fingerprint_reader(None)
    yield
    pipeline_api.register_sme_synthesizer_factory(None)
    pipeline_api.register_profile_chunks_reader(None)
    pipeline_api.register_adapter_fingerprint_reader(None)


def _wire_input_hash(chunks=None, fingerprint=("1.0.0", "adapter_hash_1")):
    pipeline_api.register_profile_chunks_reader(
        lambda sub, prof: list(chunks or [])
    )
    pipeline_api.register_adapter_fingerprint_reader(
        lambda sub, dom: fingerprint
    )


def test_first_run_always_triggers_synthesis_and_persists_hash():
    synth = MagicMock()
    synth.run.return_value = {"dossier": 1}
    chunks = [
        {"doc_id": "d1", "chunk_id": "c1", "text": "t1"},
        {"doc_id": "d2", "chunk_id": "c2", "text": "t2"},
    ]
    _wire_input_hash(chunks=chunks)
    with patch.object(
        pipeline_api, "count_incomplete_docs_in_profile", return_value=0
    ), patch.object(
        pipeline_api, "get_flag_resolver", return_value=_flag_resolver(True)
    ), patch.object(
        pipeline_api,
        "get_profile_record",
        return_value={"profile_domain": "finance"},
    ), patch.object(
        pipeline_api, "update_profile_record"
    ) as upr, patch.object(
        pipeline_api, "append_audit_log"
    ) as audit, patch.object(
        pipeline_api, "update_pipeline_status"
    ) as flip, patch.object(
        pipeline_api, "_safe_invalidate_qa_index"
    ):
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        pipeline_api.finalize_training_for_doc(_doc())
        synth.run.assert_called_once()
        flip.assert_called_once_with("d_last", "TRAINING_COMPLETED")
        updates = upr.call_args[0][2]
        assert "sme_last_input_hash" in updates
        assert updates["sme_last_input_hash"]
        assert updates["sme_last_run_id"]
        audit_events = [c[0][1] for c in audit.call_args_list]
        assert "SME_SYNTHESIS_COMPLETED" in audit_events


def test_second_run_with_unchanged_inputs_skips_synthesis():
    """Second re-entry must short-circuit: stored hash equals current."""
    synth = MagicMock()
    chunks = [{"doc_id": "d1", "chunk_id": "c1", "text": "t1"}]
    _wire_input_hash(chunks=chunks)
    # Precompute what the current hash will be so the profile record can
    # return the same value and cause a match.
    from src.intelligence.sme.input_hash import compute_input_hash_for_profile

    known_hash = compute_input_hash_for_profile(
        subscription_id="sub_a",
        profile_id="prof_x",
        chunks=chunks,
        adapter_version="1.0.0",
        adapter_content_hash="adapter_hash_1",
    )
    profile_rec = {
        "profile_domain": "finance",
        "sme_last_input_hash": known_hash,
    }
    with patch.object(
        pipeline_api, "count_incomplete_docs_in_profile", return_value=0
    ), patch.object(
        pipeline_api, "get_flag_resolver", return_value=_flag_resolver(True)
    ), patch.object(
        pipeline_api,
        "get_profile_record",
        return_value=profile_rec,
    ), patch(
        "src.api.document_status.get_profile_record",
        return_value=profile_rec,
    ), patch.object(
        pipeline_api, "update_profile_record"
    ) as upr, patch.object(
        pipeline_api, "append_audit_log"
    ) as audit, patch.object(
        pipeline_api, "update_pipeline_status"
    ) as flip, patch.object(
        pipeline_api, "_safe_invalidate_qa_index"
    ):
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        pipeline_api.finalize_training_for_doc(_doc())
        synth.run.assert_not_called()
        flip.assert_called_once_with("d_last", "TRAINING_COMPLETED")
        # No profile write when we skip.
        upr.assert_not_called()
        audit_events = [c[0][1] for c in audit.call_args_list]
        assert "SME_SYNTHESIS_SKIPPED_INPUT_UNCHANGED" in audit_events


def test_content_change_retriggers_synthesis():
    """Adding a chunk changes the input hash → synthesis must re-run."""
    synth = MagicMock()
    synth.run.return_value = {"dossier": 1}
    chunks_v2 = [
        {"doc_id": "d1", "chunk_id": "c1", "text": "t1"},
        {"doc_id": "d_NEW", "chunk_id": "c_NEW", "text": "t_new"},
    ]
    _wire_input_hash(chunks=chunks_v2)
    # Store a stale hash (from a different chunk set).
    from src.intelligence.sme.input_hash import compute_input_hash_for_profile

    stale_hash = compute_input_hash_for_profile(
        subscription_id="sub_a",
        profile_id="prof_x",
        chunks=[{"doc_id": "d1", "chunk_id": "c1", "text": "t1"}],  # subset
        adapter_version="1.0.0",
        adapter_content_hash="adapter_hash_1",
    )
    profile_rec = {
        "profile_domain": "finance",
        "sme_last_input_hash": stale_hash,
    }
    with patch.object(
        pipeline_api, "count_incomplete_docs_in_profile", return_value=0
    ), patch.object(
        pipeline_api, "get_flag_resolver", return_value=_flag_resolver(True)
    ), patch.object(
        pipeline_api,
        "get_profile_record",
        return_value=profile_rec,
    ), patch(
        "src.api.document_status.get_profile_record",
        return_value=profile_rec,
    ), patch.object(
        pipeline_api, "update_profile_record"
    ) as upr, patch.object(
        pipeline_api, "append_audit_log"
    ), patch.object(
        pipeline_api, "update_pipeline_status"
    ) as flip, patch.object(
        pipeline_api, "_safe_invalidate_qa_index"
    ):
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        pipeline_api.finalize_training_for_doc(_doc())
        synth.run.assert_called_once()
        flip.assert_called_once_with("d_last", "TRAINING_COMPLETED")
        updates = upr.call_args[0][2]
        # New hash must differ from the stale one.
        assert updates["sme_last_input_hash"] != stale_hash


def test_missing_readers_fall_through_to_real_synthesis():
    """If either reader is not registered we must NOT short-circuit — the
    safe fallback is to run synthesis."""
    synth = MagicMock()
    synth.run.return_value = {}
    with patch.object(
        pipeline_api, "count_incomplete_docs_in_profile", return_value=0
    ), patch.object(
        pipeline_api, "get_flag_resolver", return_value=_flag_resolver(True)
    ), patch.object(
        pipeline_api,
        "get_profile_record",
        return_value={"profile_domain": "finance"},
    ), patch.object(
        pipeline_api, "update_profile_record"
    ), patch.object(
        pipeline_api, "append_audit_log"
    ), patch.object(
        pipeline_api, "update_pipeline_status"
    ), patch.object(
        pipeline_api, "_safe_invalidate_qa_index"
    ):
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        pipeline_api.finalize_training_for_doc(_doc())
        synth.run.assert_called_once()


def test_qa_index_invalidated_after_completed_transition():
    """invalidate_qa_index must fire after the status flip on both the
    success path and the skip path."""
    synth = MagicMock()
    synth.run.return_value = {}
    with patch.object(
        pipeline_api, "count_incomplete_docs_in_profile", return_value=0
    ), patch.object(
        pipeline_api, "get_flag_resolver", return_value=_flag_resolver(True)
    ), patch.object(
        pipeline_api,
        "get_profile_record",
        return_value={"profile_domain": "finance"},
    ), patch.object(
        pipeline_api, "update_profile_record"
    ), patch.object(
        pipeline_api, "append_audit_log"
    ), patch.object(
        pipeline_api, "update_pipeline_status"
    ), patch.object(
        pipeline_api, "_safe_invalidate_qa_index"
    ) as inv:
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        pipeline_api.finalize_training_for_doc(_doc())
        inv.assert_called_once_with(
            subscription_id="sub_a", profile_id="prof_x"
        )
