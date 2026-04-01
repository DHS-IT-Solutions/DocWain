"""Unit tests for src/visualization/insights.py."""

import pytest

from src.visualization.insights import (
    categorize_insight,
    classify_severity,
    insight_to_visualization,
)


# ---------------------------------------------------------------------------
# categorize_insight — pattern_recognition
# ---------------------------------------------------------------------------


class TestCategorizePattern:
    def test_consistently_triggers_pattern_recognition(self):
        result = categorize_insight("The system consistently returns the same value.")
        assert result["category"] == "pattern_recognition"

    def test_pattern_keyword_scores(self):
        result = categorize_insight("A pattern was observed repeatedly across 5 runs.")
        assert result["category"] == "pattern_recognition"
        assert result["score"] >= 2

    def test_always_keyword(self):
        result = categorize_insight("This value is always the highest in the set.")
        assert result["category"] == "pattern_recognition"

    def test_regularly_keyword(self):
        result = categorize_insight("Errors occur regularly every time the flag is set.")
        assert result["category"] == "pattern_recognition"
        assert result["score"] >= 2

    def test_text_preserved(self):
        text = "The system consistently behaves this way."
        result = categorize_insight(text)
        assert result["text"] == text


# ---------------------------------------------------------------------------
# categorize_insight — anomaly_detection
# ---------------------------------------------------------------------------


class TestCategorizeAnomaly:
    def test_anomal_keyword(self):
        result = categorize_insight("An anomaly was detected in the pipeline output.")
        assert result["category"] == "anomaly_detection"

    def test_outlier_keyword(self):
        result = categorize_insight("This data point is a clear outlier.")
        assert result["category"] == "anomaly_detection"

    def test_significantly_higher(self):
        result = categorize_insight("The value is significantly higher than expected.")
        assert result["category"] == "anomaly_detection"

    def test_multiplier_pattern(self):
        result = categorize_insight("Latency is 3x higher than the baseline.")
        assert result["category"] == "anomaly_detection"

    def test_unexpected_keyword(self):
        result = categorize_insight("An unexpected result was produced by the model.")
        assert result["category"] == "anomaly_detection"

    def test_deviates_keyword(self):
        result = categorize_insight("The metric deviates from the expected distribution.")
        assert result["category"] == "anomaly_detection"

    def test_score_returned(self):
        result = categorize_insight("An unusual outlier that deviates significantly higher.")
        assert result["score"] >= 3


# ---------------------------------------------------------------------------
# categorize_insight — trend_analysis
# ---------------------------------------------------------------------------


class TestCategorizeTrend:
    def test_trend_keyword(self):
        result = categorize_insight("A downward trend was observed over the last quarter.")
        assert result["category"] == "trend_analysis"

    def test_rising_falling(self):
        result = categorize_insight("Revenue is rising while costs are falling.")
        assert result["category"] == "trend_analysis"
        assert result["score"] >= 2

    def test_year_over_year(self):
        result = categorize_insight("Year-over-year growth has increased by 12%.")
        assert result["category"] == "trend_analysis"

    def test_quarter_over_quarter(self):
        result = categorize_insight("Quarter-over-quarter the trend is decreasing.")
        assert result["category"] == "trend_analysis"


# ---------------------------------------------------------------------------
# categorize_insight — comparative_analysis
# ---------------------------------------------------------------------------


class TestCategorizeComparative:
    def test_compare_keyword(self):
        result = categorize_insight("Comparing version 2 with the previous release.")
        assert result["category"] == "comparative_analysis"

    def test_versus_keyword(self):
        result = categorize_insight("Model A versus model B shows a clear difference.")
        assert result["category"] == "comparative_analysis"

    def test_in_contrast(self):
        result = categorize_insight("In contrast, the new approach differs significantly.")
        assert result["category"] == "comparative_analysis"


# ---------------------------------------------------------------------------
# categorize_insight — gap_analysis
# ---------------------------------------------------------------------------


class TestCategorizeGap:
    def test_missing_keyword(self):
        result = categorize_insight("The report is missing key financial data.")
        assert result["category"] == "gap_analysis"

    def test_covers_of_pattern(self):
        result = categorize_insight("The document covers 3 of 7 required sections.")
        assert result["category"] == "gap_analysis"

    def test_incomplete_keyword(self):
        result = categorize_insight("The dataset is incomplete and lacks validation labels.")
        assert result["category"] == "gap_analysis"
        assert result["score"] >= 2

    def test_absent_keyword(self):
        result = categorize_insight("The required configuration is absent from the file.")
        assert result["category"] == "gap_analysis"


# ---------------------------------------------------------------------------
# categorize_insight — no-match fallback
# ---------------------------------------------------------------------------


class TestCategorizeNoMatch:
    def test_no_match_returns_default_category(self):
        result = categorize_insight("Hello world, this is a generic sentence.")
        assert result["category"] == "pattern_recognition"
        assert result["score"] == 0

    def test_no_match_text_preserved(self):
        text = "Nothing relevant here."
        result = categorize_insight(text)
        assert result["text"] == text


# ---------------------------------------------------------------------------
# insight_to_visualization
# ---------------------------------------------------------------------------


class TestInsightToVisualization:
    @pytest.mark.parametrize("category,expected_chart", [
        ("pattern_recognition", "bar"),
        ("anomaly_detection", "bar"),
        ("trend_analysis", "line"),
        ("comparative_analysis", "bar"),
        ("gap_analysis", "pie"),
    ])
    def test_chart_type_mapping(self, category, expected_chart):
        result = insight_to_visualization(category, data={})
        assert result["chart_type"] == expected_chart

    def test_data_passed_through(self):
        data = [1, 2, 3]
        result = insight_to_visualization("trend_analysis", data=data)
        assert result["data"] == data

    def test_title_default_empty(self):
        result = insight_to_visualization("gap_analysis", data={})
        assert result["title"] == ""

    def test_title_set(self):
        result = insight_to_visualization("gap_analysis", data={}, title="Coverage Report")
        assert result["title"] == "Coverage Report"

    def test_unknown_category_defaults_to_bar(self):
        result = insight_to_visualization("unknown_category", data={})
        assert result["chart_type"] == "bar"

    def test_result_keys(self):
        result = insight_to_visualization("pattern_recognition", data={})
        assert set(result.keys()) == {"chart_type", "data", "title"}


# ---------------------------------------------------------------------------
# classify_severity
# ---------------------------------------------------------------------------


class TestClassifySeverity:
    def test_anomaly_high_confidence_is_critical(self):
        assert classify_severity("anomaly_detection", 0.9) == "critical"

    def test_anomaly_exact_threshold_is_critical(self):
        assert classify_severity("anomaly_detection", 0.8) == "critical"

    def test_anomaly_low_confidence_is_warning(self):
        assert classify_severity("anomaly_detection", 0.5) == "warning"

    def test_anomaly_zero_confidence_is_warning(self):
        assert classify_severity("anomaly_detection", 0.0) == "warning"

    def test_other_category_high_confidence_is_warning(self):
        assert classify_severity("trend_analysis", 0.9) == "warning"

    def test_other_category_exact_threshold_is_warning(self):
        assert classify_severity("pattern_recognition", 0.8) == "warning"

    def test_other_category_low_confidence_is_info(self):
        assert classify_severity("gap_analysis", 0.5) == "info"

    def test_other_category_zero_confidence_is_info(self):
        assert classify_severity("comparative_analysis", 0.0) == "info"

    def test_below_threshold_boundary(self):
        assert classify_severity("trend_analysis", 0.79) == "info"
