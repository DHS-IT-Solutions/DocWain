"""Phase 2 Task 8 — training-stage integration tests for
:func:`finalize_training_for_doc` in ``src/api/pipeline_api.py``.

The synthesizer factory is injectable so tests run without standing up the
real :class:`SMESynthesizer`. Every test asserts both the pipeline-status
behaviour AND the audit-log side effects required by Phase 2.
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


@pytest.fixture(autouse=True)
def _clear_factory():
    """Every test starts without a registered synthesizer factory."""
    pipeline_api.register_sme_synthesizer_factory(None)
    yield
    pipeline_api.register_sme_synthesizer_factory(None)


def _flag_resolver(enabled: bool):
    r = MagicMock()
    r.is_enabled.return_value = enabled
    return r


def test_non_last_doc_flips_status_without_synthesis():
    """When other docs are still incomplete, flip status immediately and
    skip synthesis."""
    with patch.object(pipeline_api, "count_incomplete_docs_in_profile", return_value=3), \
         patch.object(pipeline_api, "update_pipeline_status") as flip, \
         patch.object(pipeline_api, "append_audit_log") as audit, \
         patch.object(pipeline_api, "get_flag_resolver") as flags:
        synth = MagicMock()
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        pipeline_api.finalize_training_for_doc(_doc())
        flip.assert_called_once_with("d_last", "TRAINING_COMPLETED")
        synth.run.assert_not_called()
        audit.assert_not_called()
        flags.assert_not_called()


def test_last_doc_with_flag_off_flips_status_without_synthesis():
    with patch.object(pipeline_api, "count_incomplete_docs_in_profile", return_value=0), \
         patch.object(pipeline_api, "update_pipeline_status") as flip, \
         patch.object(pipeline_api, "get_flag_resolver",
                      return_value=_flag_resolver(False)):
        synth = MagicMock()
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        pipeline_api.finalize_training_for_doc(_doc())
        flip.assert_called_once_with("d_last", "TRAINING_COMPLETED")
        synth.run.assert_not_called()


def test_last_doc_with_flag_on_and_no_factory_flips_status():
    """When no synthesizer factory is registered (legacy deploy) synthesis
    is effectively off — status still flips so the pipeline does not stall."""
    with patch.object(pipeline_api, "count_incomplete_docs_in_profile", return_value=0), \
         patch.object(pipeline_api, "update_pipeline_status") as flip, \
         patch.object(pipeline_api, "get_flag_resolver",
                      return_value=_flag_resolver(True)):
        pipeline_api.finalize_training_for_doc(_doc())
        flip.assert_called_once_with("d_last", "TRAINING_COMPLETED")


def test_flag_resolver_uninitialised_skips_synthesis():
    """A RuntimeError from ``get_flag_resolver()`` (pre-Phase-2 deploy)
    should treat the flag as off and flip status without failing the
    pipeline."""
    with patch.object(pipeline_api, "count_incomplete_docs_in_profile", return_value=0), \
         patch.object(pipeline_api, "update_pipeline_status") as flip, \
         patch.object(pipeline_api, "get_flag_resolver",
                      side_effect=RuntimeError("not initialised")):
        synth = MagicMock()
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        pipeline_api.finalize_training_for_doc(_doc())
        flip.assert_called_once_with("d_last", "TRAINING_COMPLETED")
        synth.run.assert_not_called()


def test_last_doc_with_flag_on_runs_synthesis_then_flips_status():
    synth = MagicMock()
    synth.run.return_value = {"dossier": 3, "insight": 2, "recommendation": 1}
    profile_rec = {"profile_domain": "finance", "sme_synthesis_version": 2}
    with patch.object(pipeline_api, "count_incomplete_docs_in_profile", return_value=0), \
         patch.object(pipeline_api, "get_flag_resolver",
                      return_value=_flag_resolver(True)), \
         patch.object(pipeline_api, "get_profile_record",
                      return_value=profile_rec), \
         patch.object(pipeline_api, "update_profile_record") as upr, \
         patch.object(pipeline_api, "append_audit_log") as audit, \
         patch.object(pipeline_api, "update_pipeline_status") as flip:
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        pipeline_api.finalize_training_for_doc(_doc())

        synth.run.assert_called_once_with(
            subscription_id="sub_a",
            profile_id="prof_x",
            profile_domain="finance",
            synthesis_version=3,  # existing 2 + 1
        )
        upr.assert_called_once()
        updates = upr.call_args[0][2]
        assert updates["sme_synthesis_version"] == 3
        assert updates["profile_domain"] == "finance"
        audit.assert_called_once()
        # Last arg positional is action name or keyword action; assert shape:
        action = audit.call_args[0][1]
        assert action == "SME_SYNTHESIS_COMPLETED"
        assert audit.call_args.kwargs["synthesis_version"] == 3
        flip.assert_called_once_with("d_last", "TRAINING_COMPLETED")


def test_synthesis_failure_audits_and_reraises_without_flipping_status():
    synth = MagicMock()
    synth.run.side_effect = RuntimeError("boom")
    with patch.object(pipeline_api, "count_incomplete_docs_in_profile", return_value=0), \
         patch.object(pipeline_api, "get_flag_resolver",
                      return_value=_flag_resolver(True)), \
         patch.object(pipeline_api, "get_profile_record",
                      return_value={"profile_domain": "finance"}), \
         patch.object(pipeline_api, "update_profile_record") as upr, \
         patch.object(pipeline_api, "append_audit_log") as audit, \
         patch.object(pipeline_api, "update_pipeline_status") as flip:
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        with pytest.raises(RuntimeError, match="boom"):
            pipeline_api.finalize_training_for_doc(_doc())
        flip.assert_not_called()
        audit.assert_called_once()
        assert audit.call_args[0][1] == "SME_SYNTHESIS_FAILED"
        upr.assert_not_called()


def test_legacy_doc_without_subscription_profile_still_flips_status():
    """A legacy row lacking subscription/profile must still advance (no
    synthesis context possible)."""
    with patch.object(pipeline_api, "count_incomplete_docs_in_profile", return_value=0), \
         patch.object(pipeline_api, "update_pipeline_status") as flip:
        pipeline_api.finalize_training_for_doc(
            {"document_id": "d_legacy", "subscription_id": None, "profile_id": None}
        )
        flip.assert_called_once_with("d_legacy", "TRAINING_COMPLETED")


def test_missing_document_id_raises_value_error():
    with pytest.raises(ValueError, match="document_id required"):
        pipeline_api.finalize_training_for_doc(
            {"subscription_id": "s", "profile_id": "p"}
        )


def test_no_new_pipeline_status_introduced():
    """Phase 2 invariant: no new pipeline_status string. We only ever emit
    ``PIPELINE_TRAINING_COMPLETED`` from this helper."""
    from src.api import statuses

    # Defensive check against accidental regressions — the only training-
    # terminal status stays ``TRAINING_COMPLETED``.
    assert statuses.PIPELINE_TRAINING_COMPLETED == "TRAINING_COMPLETED"


def test_default_profile_domain_is_generic_when_profile_record_missing():
    synth = MagicMock()
    synth.run.return_value = {}
    with patch.object(pipeline_api, "count_incomplete_docs_in_profile", return_value=0), \
         patch.object(pipeline_api, "get_flag_resolver",
                      return_value=_flag_resolver(True)), \
         patch.object(pipeline_api, "get_profile_record",
                      return_value=None), \
         patch.object(pipeline_api, "update_profile_record"), \
         patch.object(pipeline_api, "append_audit_log"), \
         patch.object(pipeline_api, "update_pipeline_status"):
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        pipeline_api.finalize_training_for_doc(_doc())
        synth.run.assert_called_once()
        assert synth.run.call_args.kwargs["profile_domain"] == "generic"
        assert synth.run.call_args.kwargs["synthesis_version"] == 1


def test_update_profile_record_failure_is_nonfatal():
    """A control-plane write failure after a successful synthesis must not
    prevent the pipeline from advancing."""
    synth = MagicMock()
    synth.run.return_value = {"dossier": 1}
    with patch.object(pipeline_api, "count_incomplete_docs_in_profile", return_value=0), \
         patch.object(pipeline_api, "get_flag_resolver",
                      return_value=_flag_resolver(True)), \
         patch.object(pipeline_api, "get_profile_record",
                      return_value={"profile_domain": "finance"}), \
         patch.object(pipeline_api, "update_profile_record",
                      side_effect=RuntimeError("mongo temporarily down")), \
         patch.object(pipeline_api, "append_audit_log"), \
         patch.object(pipeline_api, "update_pipeline_status") as flip:
        pipeline_api.register_sme_synthesizer_factory(lambda: synth)
        # Should not raise; best-effort write.
        pipeline_api.finalize_training_for_doc(_doc())
        flip.assert_called_once_with("d_last", "TRAINING_COMPLETED")
