from src.intelligence.insights.schema import Insight, EvidenceSpan
from src.intelligence.visualizations.generator import (
    generate_visualizations_for_insight,
    generate_visualizations_for_profile,
)


def _ins(itype="anomaly", refreshed_at="2026-04-25T10:00:00+00:00", insight_id="i1"):
    return Insight(
        insight_id=insight_id, profile_id="P", subscription_id="S",
        document_ids=["D1"], domain="generic", insight_type=itype,
        headline="H", body="b grounded in quote",
        evidence_doc_spans=[EvidenceSpan(
            document_id="D1", page=1, char_start=0, char_end=2, quote="b"
        )],
        confidence=0.5, severity="notice", adapter_version="generic@1.0",
        refreshed_at=refreshed_at,
    )


def test_trend_insight_produces_trend_chart_data():
    out = generate_visualizations_for_insight(_ins(itype="trend"))
    assert any(v["viz_id"] == "trend_chart" for v in out)


def test_comparison_insight_produces_table():
    out = generate_visualizations_for_insight(_ins(itype="comparison"))
    assert any(v["viz_id"] == "comparison_table" for v in out)


def test_profile_aggregator_collects_timeline_from_dated_insights():
    insights = [
        _ins(refreshed_at="2026-04-01T10:00:00+00:00", insight_id="a"),
        _ins(refreshed_at="2026-04-15T10:00:00+00:00", insight_id="b"),
        _ins(refreshed_at="2026-04-25T10:00:00+00:00", insight_id="c"),
    ]
    out = generate_visualizations_for_profile(insights)
    timelines = [v for v in out if v["viz_id"] == "timeline"]
    assert len(timelines) == 1
    assert len(timelines[0]["data"]["events"]) == 3


def test_other_types_produce_no_viz():
    out = generate_visualizations_for_insight(_ins(itype="recommendation"))
    assert out == []
