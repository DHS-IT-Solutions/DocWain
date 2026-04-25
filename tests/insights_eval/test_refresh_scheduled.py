from unittest.mock import patch


def test_scheduled_runs_only_when_flag_on(monkeypatch):
    from src.api.feature_flags import FLAG_NAMES
    for name in FLAG_NAMES:
        monkeypatch.delenv(name, raising=False)
    from src.tasks.researcher_v2_refresh import refresh_scheduled_pass
    result = refresh_scheduled_pass(profile_id="P", subscription_id="S")
    assert result["status"] == "skipped_flag_off"


def test_scheduled_dispatches_profile_pass(monkeypatch):
    monkeypatch.setenv("REFRESH_SCHEDULED_ENABLED", "true")
    monkeypatch.setenv("INSIGHTS_TYPE_COMPARISON_ENABLED", "true")

    from src.tasks.researcher_v2_refresh import refresh_scheduled_pass

    docs = [{"document_id": "D1", "text": "x"}, {"document_id": "D2", "text": "y"}]
    runner_calls = []

    def fake_runner(**kwargs):
        runner_calls.append(kwargs)
        return {"status": "ok"}

    with patch(
        "src.tasks.researcher_v2_refresh.fetch_active_profile_documents", return_value=docs
    ), patch(
        "src.tasks.researcher_v2_refresh.run_researcher_v2_for_profile", side_effect=fake_runner
    ):
        result = refresh_scheduled_pass(profile_id="P", subscription_id="S")
    assert result["status"] == "ok"
    assert runner_calls and len(runner_calls[0]["documents"]) == 2
