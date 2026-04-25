from src.intelligence.actions.handlers import (
    artifact_handler, form_fill_handler, plan_handler, reminder_handler,
)
from src.intelligence.adapters.schema import ActionTemplate


def _action(action_type="artifact"):
    return ActionTemplate(
        action_id="a1", title="X", action_type=action_type,
        artifact_template="t.md", requires_confirmation=False,
    )


def test_artifact_handler_renders_template_and_uploads(tmp_path, monkeypatch):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "t.md").write_text("Hello {{ profile_id }} — input is {{ inputs['foo'] }}")
    monkeypatch.setenv("ACTION_TEMPLATE_ROOT", str(template_dir))

    out = artifact_handler(
        action=_action(),
        profile_id="P-1",
        inputs={"foo": "bar"},
    )
    assert "artifact_blob_url" in out
    assert "Hello P-1" in out["artifact_content"]
    assert "input is bar" in out["artifact_content"]


def test_form_fill_handler_returns_filled_form():
    out = form_fill_handler(
        action=_action(action_type="form_fill"),
        profile_id="P-1",
        inputs={"name": "Test", "policy": "ABC-001"},
    )
    assert out["status"] == "filled"
    assert out["form_data"]["name"] == "Test"


def test_plan_handler_returns_checklist():
    out = plan_handler(
        action=_action(action_type="plan"),
        profile_id="P-1",
        inputs={"steps": ["A", "B", "C"]},
    )
    assert "checklist" in out
    assert len(out["checklist"]) == 3


def test_reminder_handler_in_system_only():
    out = reminder_handler(
        action=_action(action_type="reminder"),
        profile_id="P-1",
        inputs={"fire_at": "2026-12-31T00:00:00+00:00", "message": "Renew"},
    )
    assert out["status"] == "scheduled_in_system"
    assert out["fire_at"] == "2026-12-31T00:00:00+00:00"
