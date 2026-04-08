"""Tests for src/api/standalone_output.py — output format conversion."""
import csv
import io
import pytest

from src.api.standalone_output import convert_output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TABLE_DATA = {
    "tables": [
        {
            "headers": ["Name", "Age", "City"],
            "rows": [["Alice", "30", "London"], ["Bob", "25", "Paris"]],
            "page": 1,
            "caption": "Sample table",
        }
    ]
}

ENTITY_DATA = {
    "entities": [
        {"text": "Alice Smith", "type": "PERSON", "page": 1, "confidence": 0.95},
        {"text": "London", "type": "LOCATION", "page": 2, "confidence": 0.88},
    ]
}

SUMMARY_DATA = {
    "sections": [
        {
            "title": "Introduction",
            "summary": "This document covers key findings.",
            "key_points": ["Point one", "Point two", "Point three"],
        }
    ]
}


# ---------------------------------------------------------------------------
# Table converters
# ---------------------------------------------------------------------------


def test_convert_table_to_csv():
    result = convert_output(TABLE_DATA, "table", "csv")
    assert isinstance(result, str)
    reader = csv.reader(io.StringIO(result))
    rows = list(reader)
    # First row is headers
    assert rows[0] == ["Name", "Age", "City"]
    # Data rows follow
    assert ["Alice", "30", "London"] in rows
    assert ["Bob", "25", "Paris"] in rows


def test_convert_table_to_markdown():
    result = convert_output(TABLE_DATA, "table", "markdown")
    assert isinstance(result, str)
    lines = [l for l in result.splitlines() if l.strip()]
    # Header row must be pipe-delimited
    assert "|" in lines[0]
    assert "Name" in lines[0]
    assert "Age" in lines[0]
    # Separator row
    assert "---" in lines[1]
    # Data rows
    assert "Alice" in result
    assert "Bob" in result


def test_convert_table_to_html():
    result = convert_output(TABLE_DATA, "table", "html")
    assert isinstance(result, str)
    assert "<table" in result
    assert "<thead" in result
    assert "<th" in result
    assert "Name" in result
    assert "Alice" in result
    assert "Bob" in result


# ---------------------------------------------------------------------------
# Entity converters
# ---------------------------------------------------------------------------


def test_convert_entities_to_csv():
    result = convert_output(ENTITY_DATA, "entities", "csv")
    assert isinstance(result, str)
    reader = csv.reader(io.StringIO(result))
    rows = list(reader)
    # Header row
    assert rows[0] == ["text", "type", "page", "confidence"]
    texts = [r[0] for r in rows[1:]]
    types = [r[1] for r in rows[1:]]
    assert "Alice Smith" in texts
    assert "PERSON" in types
    assert "London" in texts


def test_convert_entities_to_markdown():
    result = convert_output(ENTITY_DATA, "entities", "markdown")
    assert isinstance(result, str)
    assert "## Entities" in result
    assert "Alice Smith" in result
    assert "PERSON" in result
    assert "London" in result
    # Bullet list items
    assert "- **" in result


def test_convert_entities_to_html():
    result = convert_output(ENTITY_DATA, "entities", "html")
    assert isinstance(result, str)
    assert "<dl" in result
    assert "<dt" in result
    assert "<dd" in result
    assert "Alice Smith" in result


# ---------------------------------------------------------------------------
# Summary converters
# ---------------------------------------------------------------------------


def test_convert_summary_to_csv():
    result = convert_output(SUMMARY_DATA, "summary", "csv")
    assert isinstance(result, str)
    reader = csv.reader(io.StringIO(result))
    rows = list(reader)
    assert rows[0] == ["title", "summary", "key_points"]
    assert rows[1][0] == "Introduction"
    # key_points joined with "; "
    assert "Point one" in rows[1][2]
    assert "Point two" in rows[1][2]


def test_convert_summary_to_markdown():
    result = convert_output(SUMMARY_DATA, "summary", "markdown")
    assert isinstance(result, str)
    assert "## Introduction" in result
    assert "This document covers key findings." in result
    assert "**Key Points:**" in result
    assert "Point one" in result
    assert "Point two" in result


def test_convert_summary_to_html():
    result = convert_output(SUMMARY_DATA, "summary", "html")
    assert isinstance(result, str)
    assert "<section" in result
    assert "<h2" in result
    assert "Introduction" in result
    assert "<p>" in result
    assert "<ul" in result
    assert "<li" in result
    assert "Point one" in result


# ---------------------------------------------------------------------------
# JSON passthrough
# ---------------------------------------------------------------------------


def test_json_passthrough_table():
    result = convert_output(TABLE_DATA, "table", "json")
    assert result is TABLE_DATA


def test_json_passthrough_entities():
    result = convert_output(ENTITY_DATA, "entities", "json")
    assert result is ENTITY_DATA


def test_json_passthrough_summary():
    result = convert_output(SUMMARY_DATA, "summary", "json")
    assert result is SUMMARY_DATA


# ---------------------------------------------------------------------------
# HTML escaping
# ---------------------------------------------------------------------------


def test_html_escaping_in_table():
    data = {
        "tables": [
            {
                "headers": ["Item & Price"],
                "rows": [["<b>Widget</b>", "5 > 3"]],
                "page": 1,
                "caption": "",
            }
        ]
    }
    result = convert_output(data, "table", "html")
    assert "&amp;" in result or "&lt;" in result or "&gt;" in result
    # Raw < > & should not appear unescaped inside tag content
    # (allow them in tag names like <table>, <th>, etc.)
    import re
    # Strip all HTML tags and check the text content is escaped
    text_content = re.sub(r"<[^>]+>", "", result)
    assert "<b>" not in text_content
    assert ">" not in text_content or "&gt;" in result
