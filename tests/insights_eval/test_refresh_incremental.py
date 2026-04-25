from unittest.mock import MagicMock, patch


def test_incremental_refresh_marks_stale_then_re_runs(monkeypatch):
    monkeypatch.setenv("REFRESH_ON_UPLOAD_ENABLED", "true")
    monkeypatch.setenv("REFRESH_INCREMENTAL_ENABLED", "true")
    monkeypatch.setenv("INSIGHTS_TYPE_ANOMALY_ENABLED", "true")

    from src.tasks.researcher_v2_refresh import refresh_for_new_doc

    mark_calls = []
    runner_calls = []

    def fake_mark(**kwargs):
        mark_calls.append(kwargs)
        return 2

    def fake_runner(**kwargs):
        runner_calls.append(kwargs)
        return {"status": "ok"}

    with patch(
        "src.tasks.researcher_v2_refresh.mark_stale_for_documents", side_effect=fake_mark
    ), patch(
        "src.tasks.researcher_v2_refresh.run_researcher_v2_for_doc", side_effect=fake_runner
    ), patch(
        "src.tasks.researcher_v2_refresh.resolve_default_index_collection", return_value=MagicMock()
    ):
        result = refresh_for_new_doc(
            document_id="D-NEW", profile_id="P", subscription_id="S",
            document_text="text", domain_hint="generic",
        )
    assert result["status"] == "ok"
    assert mark_calls and mark_calls[0]["document_ids"] == ["D-NEW"]
    assert runner_calls and runner_calls[0]["document_id"] == "D-NEW"


def test_incremental_short_circuits_when_flag_off(monkeypatch):
    from src.api.feature_flags import FLAG_NAMES
    for name in FLAG_NAMES:
        monkeypatch.delenv(name, raising=False)
    from src.tasks.researcher_v2_refresh import refresh_for_new_doc
    result = refresh_for_new_doc(
        document_id="D", profile_id="P", subscription_id="S",
        document_text="x", domain_hint="generic",
    )
    assert result["status"] == "skipped_flag_off"
