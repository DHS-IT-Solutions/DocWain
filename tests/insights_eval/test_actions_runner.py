from src.intelligence.actions.runner import (
    ActionRunner,
    ActionExecutionResult,
)
from src.intelligence.adapters.schema import ActionTemplate


def _action(*, requires_confirmation=False, action_type="artifact"):
    return ActionTemplate(
        action_id="a1", title="Test", action_type=action_type,
        artifact_template="t.md", requires_confirmation=requires_confirmation,
    )


class FakeHandler:
    def __init__(self):
        self.executed = False
    def __call__(self, *, action, profile_id, inputs):
        self.executed = True
        return {"artifact_blob_url": "blob://x"}


def test_unconfirmed_action_returns_preview():
    handler = FakeHandler()
    runner = ActionRunner(
        handlers={"artifact": handler}, audit_writer=lambda **kw: None,
    )
    result = runner.execute(
        action=_action(requires_confirmation=True),
        profile_id="P", inputs={}, confirmed=False,
    )
    assert isinstance(result, ActionExecutionResult)
    assert result.status == "needs_confirmation"
    assert handler.executed is False


def test_confirmed_action_executes():
    handler = FakeHandler()
    audit_calls = []
    runner = ActionRunner(
        handlers={"artifact": handler}, audit_writer=lambda **kw: audit_calls.append(kw),
    )
    result = runner.execute(
        action=_action(requires_confirmation=True),
        profile_id="P", inputs={"k": "v"}, confirmed=True,
    )
    assert result.status == "executed"
    assert handler.executed is True
    assert audit_calls and audit_calls[0]["action_id"] == "a1"


def test_safe_action_no_confirmation_required():
    handler = FakeHandler()
    runner = ActionRunner(
        handlers={"artifact": handler}, audit_writer=lambda **kw: None,
    )
    result = runner.execute(
        action=_action(requires_confirmation=False),
        profile_id="P", inputs={}, confirmed=False,
    )
    assert result.status == "executed"


def test_external_side_effect_gated_by_separate_flag(monkeypatch):
    from src.api.feature_flags import FLAG_NAMES
    for name in FLAG_NAMES:
        monkeypatch.delenv(name, raising=False)
    handler = FakeHandler()
    runner = ActionRunner(
        handlers={"reminder": handler}, audit_writer=lambda **kw: None,
    )
    action = ActionTemplate(
        action_id="a-ext", title="Send email", action_type="reminder",
        requires_confirmation=True,
    )
    setattr(action, "_side_effect", "external")
    result = runner.execute(
        action=action, profile_id="P", inputs={}, confirmed=True,
    )
    assert result.status == "external_side_effects_disabled"
    assert handler.executed is False


def test_audit_writer_records_call():
    from src.intelligence.actions.audit import make_audit_writer

    written = []
    class FakeColl:
        def insert_one(self, doc):
            written.append(doc)
    writer = make_audit_writer(collection=FakeColl())
    writer(
        action_id="a1",
        profile_id="P",
        inputs={"k": "v"},
        output={"status": "executed"},
    )
    assert len(written) == 1
    rec = written[0]
    assert rec["action_id"] == "a1"
    assert rec["profile_id"] == "P"
    assert rec["output"]["status"] == "executed"
    assert "at" in rec
