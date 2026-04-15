import json
import pytest


def test_format_json_passthrough():
    from standalone.output_formatter import format_output

    raw = '{"document_type": "invoice", "total": 500}'
    result = format_output(raw, "json")
    assert isinstance(result, dict)
    assert result["document_type"] == "invoice"


def test_format_json_from_non_json_text():
    from standalone.output_formatter import format_output

    raw = "The document is an invoice with total $500."
    result = format_output(raw, "json")
    assert isinstance(result, dict)
    assert "content" in result


def test_format_csv():
    from standalone.output_formatter import format_output

    raw = '{"tables": [{"headers": ["Name", "Amount"], "rows": [["Alice", "100"], ["Bob", "200"]]}]}'
    result = format_output(raw, "csv")
    assert isinstance(result, str)
    assert "Name" in result
    assert "Alice" in result


def test_format_sections():
    from standalone.output_formatter import format_output

    raw = '{"sections": [{"title": "Introduction", "content": "This is the intro."}]}'
    result = format_output(raw, "sections")
    assert isinstance(result, dict)
    assert "sections" in result


def test_format_flatfile():
    from standalone.output_formatter import format_output

    raw = '{"document_type": "invoice", "vendor": "Acme", "total": "500"}'
    result = format_output(raw, "flatfile")
    assert isinstance(result, str)
    assert "document_type" in result
    assert "invoice" in result


def test_format_tables():
    from standalone.output_formatter import format_output

    raw = '{"tables": [{"headers": ["A", "B"], "rows": [["1", "2"]]}]}'
    result = format_output(raw, "tables")
    assert isinstance(result, (dict, list))


def test_format_json_extracts_from_markdown_fence():
    from standalone.output_formatter import format_output

    raw = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
    result = format_output(raw, "json")
    assert isinstance(result, dict)
    assert result["key"] == "value"
