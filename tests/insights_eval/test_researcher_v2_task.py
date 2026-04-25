from unittest.mock import MagicMock, patch

from src.intelligence.adapters.schema import (
    Adapter, AppliesWhen, ResearcherSection, InsightTypeConfig, KnowledgeConfig,
)
from src.intelligence.insights.schema import Insight, EvidenceSpan


def _adapter() -> Adapter:
    return Adapter(
        name="generic", version="1.0", description="t",
        applies_when=AppliesWhen(),
        researcher=ResearcherSection(insight_types={
            "anomaly": InsightTypeConfig(prompt_template="", enabled=True),
        }),
        knowledge=KnowledgeConfig(),
    )


def test_run_researcher_v2_calls_runner_per_enabled_type(monkeypatch):
    monkeypatch.setenv("INSIGHTS_TYPE_ANOMALY_ENABLED", "true")
    from src.tasks.researcher_v2 import run_researcher_v2_for_doc

    fake_store = MagicMock()
    fake_runner = MagicMock()
    fake_runner.return_value.insights = [
        Insight(
            insight_id="i", profile_id="p", subscription_id="s",
            document_ids=["D"], domain="generic", insight_type="anomaly",
            headline="H", body="H grounded quote",
            evidence_doc_spans=[EvidenceSpan(
                document_id="D", page=1, char_start=0, char_end=2, quote="H"
            )],
            confidence=0.5, severity="notice", adapter_version="generic@1.0",
        )
    ]
    fake_runner.return_value.skipped_reason = None
    with patch(
        "src.tasks.researcher_v2.resolve_default_store", return_value=fake_store
    ), patch(
        "src.tasks.researcher_v2.resolve_default_adapter", return_value=_adapter()
    ), patch(
        "src.tasks.researcher_v2.resolve_default_llm", return_value=lambda **kw: ""
    ), patch(
        "src.tasks.researcher_v2.run_per_doc_insight_pass", side_effect=fake_runner
    ):
        run_researcher_v2_for_doc(
            document_id="D", profile_id="p", subscription_id="s",
            document_text="x",
        )
    fake_store.write.assert_called()


def test_disabled_flag_short_circuits(monkeypatch):
    # Ensure no flags are set
    from src.api.feature_flags import FLAG_NAMES
    for name in FLAG_NAMES:
        monkeypatch.delenv(name, raising=False)

    from src.tasks.researcher_v2 import run_researcher_v2_for_doc
    result = run_researcher_v2_for_doc(
        document_id="D", profile_id="p", subscription_id="s",
        document_text="x",
    )
    assert result["status"] == "skipped_flag_off"
