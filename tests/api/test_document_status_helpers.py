"""Phase 2 Task 9 — unit tests for the four new SME control-plane helpers.

Every helper either reads or writes only lightweight control-plane fields on
MongoDB; the allowlist on :func:`update_profile_record` enforces the project
memory rule "MongoDB = control plane only".
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api import document_status as ds


def test_count_incomplete_docs_excludes_current_doc_and_terminal_status():
    collection = MagicMock()
    collection.count_documents.return_value = 2
    with patch.object(ds, "_documents_collection", return_value=collection):
        n = ds.count_incomplete_docs_in_profile(
            subscription_id="s",
            profile_id="p",
            exclude_document_id="d_last",
        )
    assert n == 2
    args, _ = collection.count_documents.call_args
    filt = args[0]
    assert filt["subscription_id"] == "s"
    assert filt["profile_id"] == "p"
    assert filt["document_id"] == {"$ne": "d_last"}
    assert filt["pipeline_status"] == {"$ne": "TRAINING_COMPLETED"}


def test_count_incomplete_docs_returns_zero_when_collection_unavailable():
    with patch.object(ds, "_documents_collection", return_value=None):
        assert (
            ds.count_incomplete_docs_in_profile(
                subscription_id="s",
                profile_id="p",
                exclude_document_id="d",
            )
            == 0
        )


def test_get_subscription_record_returns_dict_when_found():
    subs = MagicMock()
    subs.find_one.return_value = {"subscription_id": "s", "tier": "gold"}
    with patch.object(ds, "_subscriptions_collection", return_value=subs):
        assert ds.get_subscription_record("s") == {
            "subscription_id": "s",
            "tier": "gold",
        }
    subs.find_one.assert_called_once_with({"subscription_id": "s"})


def test_get_subscription_record_returns_none_when_missing():
    subs = MagicMock()
    subs.find_one.return_value = None
    with patch.object(ds, "_subscriptions_collection", return_value=subs):
        assert ds.get_subscription_record("missing") is None


def test_get_profile_record_returns_dict_when_found():
    profs = MagicMock()
    profs.find_one.return_value = {
        "subscription_id": "s",
        "profile_id": "p",
        "profile_domain": "finance",
    }
    with patch.object(ds, "_profiles_collection", return_value=profs):
        assert ds.get_profile_record("s", "p") == {
            "subscription_id": "s",
            "profile_id": "p",
            "profile_domain": "finance",
        }
    profs.find_one.assert_called_once_with(
        {"subscription_id": "s", "profile_id": "p"}
    )


def test_get_profile_record_returns_none_when_missing():
    profs = MagicMock()
    profs.find_one.return_value = None
    with patch.object(ds, "_profiles_collection", return_value=profs):
        assert ds.get_profile_record("s", "missing") is None


def test_update_profile_record_allows_allowlisted_keys():
    profs = MagicMock()
    with patch.object(ds, "_profiles_collection", return_value=profs):
        ds.update_profile_record(
            "s",
            "p",
            {
                "sme_synthesis_version": 3,
                "sme_last_input_hash": "abc",
                "sme_last_run_id": "run_001",
            },
        )
    args, kwargs = profs.update_one.call_args
    assert args[0] == {"subscription_id": "s", "profile_id": "p"}
    assert args[1]["$set"]["sme_synthesis_version"] == 3
    assert args[1]["$set"]["sme_last_input_hash"] == "abc"
    assert kwargs.get("upsert") is True


def test_update_profile_record_rejects_content_keys():
    profs = MagicMock()
    with patch.object(ds, "_profiles_collection", return_value=profs):
        with pytest.raises(ValueError, match="only control-plane keys allowed"):
            ds.update_profile_record("s", "p", {"narrative": "oops"})
    profs.update_one.assert_not_called()


def test_update_profile_record_rejects_mixed_good_and_bad_keys():
    profs = MagicMock()
    with patch.object(ds, "_profiles_collection", return_value=profs):
        with pytest.raises(ValueError):
            ds.update_profile_record(
                "s",
                "p",
                {"sme_synthesis_version": 3, "payload": {"x": "y"}},
            )
    profs.update_one.assert_not_called()


def test_update_profile_record_noop_on_empty_updates():
    profs = MagicMock()
    with patch.object(ds, "_profiles_collection", return_value=profs):
        ds.update_profile_record("s", "p", {})
    profs.update_one.assert_not_called()


def test_append_audit_log_still_present_and_untouched():
    """Regression guard: the Phase 2 helpers must not break the existing
    ``append_audit_log`` surface that Phase 2 Task 8 reuses."""
    assert callable(ds.append_audit_log)
    assert ds.append_audit_log.__module__ == "src.api.document_status"
