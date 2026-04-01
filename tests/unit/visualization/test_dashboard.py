"""Unit tests for src/visualization/dashboard.py."""

import pytest

from src.visualization.dashboard import compose_dashboard


# ---------------------------------------------------------------------------
# Empty data
# ---------------------------------------------------------------------------


class TestEmptyData:
    def test_empty_dict_returns_minimal_structure(self):
        result = compose_dashboard({}, query="What are the totals?")
        assert result == {"sections": [], "query": "What are the totals?"}

    def test_empty_data_has_no_document_count_key(self):
        result = compose_dashboard({}, query="test")
        assert "document_count" not in result

    def test_empty_data_preserves_query(self):
        q = "Compare revenue across all docs"
        result = compose_dashboard({}, query=q)
        assert result["query"] == q


# ---------------------------------------------------------------------------
# Multiple documents — full data
# ---------------------------------------------------------------------------


class TestMultipleDocuments:
    @pytest.fixture()
    def full_data(self):
        return {
            "documents": ["Report A", "Report B", "Report C"],
            "values": [100, 250, 175],
            "dates": ["2026-01-01", "2026-02-01", "2026-03-01"],
        }

    def test_returns_sections_list(self, full_data):
        result = compose_dashboard(full_data, query="Compare reports")
        assert isinstance(result["sections"], list)

    def test_returns_query(self, full_data):
        result = compose_dashboard(full_data, query="Compare reports")
        assert result["query"] == "Compare reports"

    def test_returns_document_count(self, full_data):
        result = compose_dashboard(full_data, query="Compare reports")
        assert result["document_count"] == 3

    def test_has_summary_table(self, full_data):
        result = compose_dashboard(full_data, query="q")
        types = [s["type"] for s in result["sections"]]
        assert "table" in types

    def test_table_section_has_rows(self, full_data):
        result = compose_dashboard(full_data, query="q")
        table = next(s for s in result["sections"] if s["type"] == "table")
        assert len(table["data"]) == 3

    def test_table_row_has_document_value_date(self, full_data):
        result = compose_dashboard(full_data, query="q")
        table = next(s for s in result["sections"] if s["type"] == "table")
        row = table["data"][0]
        assert row["document"] == "Report A"
        assert row["value"] == 100
        assert row["date"] == "2026-01-01"

    def test_has_bar_chart(self, full_data):
        result = compose_dashboard(full_data, query="q")
        charts = [s for s in result["sections"] if s.get("chart_type") == "bar"]
        assert len(charts) == 1

    def test_bar_chart_has_x_y(self, full_data):
        result = compose_dashboard(full_data, query="q")
        bar = next(s for s in result["sections"] if s.get("chart_type") == "bar")
        assert bar["x"] == ["Report A", "Report B", "Report C"]
        assert bar["y"] == [100, 250, 175]

    def test_has_line_chart(self, full_data):
        result = compose_dashboard(full_data, query="q")
        charts = [s for s in result["sections"] if s.get("chart_type") == "line"]
        assert len(charts) == 1

    def test_line_chart_has_x_y(self, full_data):
        result = compose_dashboard(full_data, query="q")
        line = next(s for s in result["sections"] if s.get("chart_type") == "line")
        assert line["x"] == ["2026-01-01", "2026-02-01", "2026-03-01"]
        assert line["y"] == [100, 250, 175]

    def test_section_count_default(self, full_data):
        result = compose_dashboard(full_data, query="q")
        # With full data we expect 3 sections (table + bar + line)
        assert len(result["sections"]) == 3


# ---------------------------------------------------------------------------
# Values only — no documents, no dates
# ---------------------------------------------------------------------------


class TestValuesOnly:
    def test_bar_chart_generated_without_documents(self):
        result = compose_dashboard({"values": [10, 20, 30]}, query="q")
        bar_sections = [s for s in result["sections"] if s.get("chart_type") == "bar"]
        assert len(bar_sections) == 1

    def test_no_table_without_documents(self):
        result = compose_dashboard({"values": [10, 20]}, query="q")
        table_sections = [s for s in result["sections"] if s["type"] == "table"]
        assert len(table_sections) == 0

    def test_no_line_chart_without_dates(self):
        result = compose_dashboard({"values": [10, 20, 30]}, query="q")
        line_sections = [s for s in result["sections"] if s.get("chart_type") == "line"]
        assert len(line_sections) == 0

    def test_single_value_no_bar_chart(self):
        result = compose_dashboard({"values": [42]}, query="q")
        bar_sections = [s for s in result["sections"] if s.get("chart_type") == "bar"]
        assert len(bar_sections) == 0


# ---------------------------------------------------------------------------
# Documents and values only (no dates)
# ---------------------------------------------------------------------------


class TestDocumentsAndValuesOnly:
    @pytest.fixture()
    def data(self):
        return {
            "documents": ["Doc X", "Doc Y"],
            "values": [500, 300],
        }

    def test_table_generated(self, data):
        result = compose_dashboard(data, query="q")
        table_sections = [s for s in result["sections"] if s["type"] == "table"]
        assert len(table_sections) == 1

    def test_table_row_has_no_date_key_when_no_dates(self, data):
        result = compose_dashboard(data, query="q")
        table = next(s for s in result["sections"] if s["type"] == "table")
        for row in table["data"]:
            assert "date" not in row

    def test_bar_chart_generated(self, data):
        result = compose_dashboard(data, query="q")
        bar_sections = [s for s in result["sections"] if s.get("chart_type") == "bar"]
        assert len(bar_sections) == 1

    def test_no_line_chart(self, data):
        result = compose_dashboard(data, query="q")
        line_sections = [s for s in result["sections"] if s.get("chart_type") == "line"]
        assert len(line_sections) == 0

    def test_document_count_correct(self, data):
        result = compose_dashboard(data, query="q")
        assert result["document_count"] == 2


# ---------------------------------------------------------------------------
# max_sections cap
# ---------------------------------------------------------------------------


class TestMaxSections:
    @pytest.fixture()
    def full_data(self):
        return {
            "documents": ["A", "B", "C"],
            "values": [1, 2, 3],
            "dates": ["2026-01-01", "2026-02-01", "2026-03-01"],
        }

    def test_max_sections_one(self, full_data):
        result = compose_dashboard(full_data, query="q", max_sections=1)
        assert len(result["sections"]) == 1

    def test_max_sections_two(self, full_data):
        result = compose_dashboard(full_data, query="q", max_sections=2)
        assert len(result["sections"]) == 2

    def test_max_sections_default_five_not_exceeded(self, full_data):
        result = compose_dashboard(full_data, query="q")
        assert len(result["sections"]) <= 5

    def test_max_sections_larger_than_available(self, full_data):
        result = compose_dashboard(full_data, query="q", max_sections=10)
        assert len(result["sections"]) == 3


# ---------------------------------------------------------------------------
# Section structure validation
# ---------------------------------------------------------------------------


class TestSectionStructure:
    def test_table_section_has_type_title_data(self):
        data = {"documents": ["D1", "D2"], "values": [1, 2]}
        result = compose_dashboard(data, query="q")
        table = next(s for s in result["sections"] if s["type"] == "table")
        assert "type" in table
        assert "title" in table
        assert "data" in table

    def test_bar_chart_has_type_chart_type_title_x_y(self):
        data = {"documents": ["D1", "D2"], "values": [1, 2]}
        result = compose_dashboard(data, query="q")
        bar = next(s for s in result["sections"] if s.get("chart_type") == "bar")
        assert bar["type"] == "chart"
        assert bar["chart_type"] == "bar"
        assert "title" in bar
        assert "x" in bar
        assert "y" in bar

    def test_line_chart_has_type_chart_type_title_x_y(self):
        data = {
            "documents": ["D1", "D2"],
            "values": [10, 20],
            "dates": ["2026-01-01", "2026-02-01"],
        }
        result = compose_dashboard(data, query="q")
        line = next(s for s in result["sections"] if s.get("chart_type") == "line")
        assert line["type"] == "chart"
        assert line["chart_type"] == "line"
        assert "title" in line
        assert "x" in line
        assert "y" in line
