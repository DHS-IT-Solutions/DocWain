import time

from src.generation.insight_injection import (
    select_insights_for_query,
    format_related_findings,
    INJECTION_BUDGET_MS,
)


def test_filters_by_severity_threshold():
    rows = [
        {"insight_id": "i1", "headline": "info-only", "severity": "info"},
        {"insight_id": "i2", "headline": "notice-level", "severity": "notice"},
        {"insight_id": "i3", "headline": "warn-level", "severity": "warn"},
    ]
    selected = select_insights_for_query(query="any", profile_insights=rows, query_entities=set())
    ids = {r["insight_id"] for r in selected}
    assert "i1" not in ids
    assert "i2" in ids and "i3" in ids


def test_relevance_filter_uses_query_entities():
    rows = [
        {"insight_id": "i1", "headline": "Premium $1800", "severity": "notice", "tags": ["premium"]},
        {"insight_id": "i2", "headline": "No flood coverage", "severity": "warn", "tags": ["flood"]},
    ]
    selected = select_insights_for_query(
        query="What is the premium?", profile_insights=rows, query_entities={"premium"}
    )
    ids = {r["insight_id"] for r in selected}
    assert "i1" in ids


def test_budget_truncates_when_exceeded():
    rows = [
        {"insight_id": f"i{i}", "headline": f"H{i}", "severity": "notice", "tags": [f"t{i}"]}
        for i in range(200)
    ]
    start = time.perf_counter_ns()
    selected = select_insights_for_query(query="x", profile_insights=rows, query_entities=set())
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    assert elapsed_ms < INJECTION_BUDGET_MS
    assert len(selected) <= 5


def test_format_related_findings_renders():
    rows = [
        {"headline": "No flood coverage", "severity": "warn"},
        {"headline": "Premium $1800", "severity": "notice"},
    ]
    out = format_related_findings(rows)
    assert "Related findings" in out
    assert "No flood coverage" in out


def test_compose_response_appends_findings_when_flag_on(monkeypatch):
    monkeypatch.setenv("INSIGHTS_PROACTIVE_INJECTION", "true")
    from src.generation.prompts import compose_response_with_insights

    base = "The premium is $1800."
    insights = [{"headline": "No flood coverage", "severity": "warn"}]
    out = compose_response_with_insights(
        base_answer=base,
        profile_insights=insights,
        query="premium",
        query_entities=set(),
    )
    assert "premium is $1800" in out
    assert "Related findings" in out
    assert "No flood coverage" in out


def test_compose_response_unchanged_when_flag_off(monkeypatch):
    from src.api.feature_flags import FLAG_NAMES
    for name in FLAG_NAMES:
        monkeypatch.delenv(name, raising=False)
    from src.generation.prompts import compose_response_with_insights
    base = "The premium is $1800."
    insights = [{"headline": "No flood coverage", "severity": "warn"}]
    out = compose_response_with_insights(
        base_answer=base, profile_insights=insights,
        query="premium", query_entities=set(),
    )
    assert out == base
