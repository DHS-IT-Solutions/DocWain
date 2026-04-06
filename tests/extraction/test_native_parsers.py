"""Tests for the native CSV/Excel parser path."""

import io

import pytest


def test_csv_native_parse_returns_structured_result():
    from src.extraction.native_parsers import parse_csv

    csv_content = b"name,age,salary\nAlice,30,80000\nBob,25,75000\nCharlie,35,90000\n"
    result = parse_csv(csv_content, filename="test.csv")
    assert result["parser"] == "native_csv"
    assert result["row_count"] == 3
    assert result["columns"] == ["name", "age", "salary"]
    assert "statistical_profile" in result
    assert result["statistical_profile"]["age"]["min"] == 25
    assert result["statistical_profile"]["age"]["max"] == 35
    assert "sample_rows" in result
    assert len(result["sample_rows"]) == 3


def test_csv_large_file_uses_sampling():
    from src.extraction.native_parsers import parse_csv

    header = b"id,value\n"
    rows = b"".join(f"{i},{i*1.5}\n".encode() for i in range(15000))
    csv_content = header + rows
    result = parse_csv(csv_content, filename="big.csv")
    assert result["row_count"] == 15000
    assert len(result["sample_rows"]) < 1000
    assert "statistical_profile" in result


def test_excel_native_parse():
    import openpyxl

    from src.extraction.native_parsers import parse_excel

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Product", "Price", "Quantity"])
    ws.append(["Widget", 9.99, 100])
    ws.append(["Gadget", 19.99, 50])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    result = parse_excel(buf.read(), filename="test.xlsx")
    assert result["parser"] == "native_excel"
    assert result["sheet_count"] == 1
    assert result["sheets"][0]["name"] == "Sheet1"
    assert result["sheets"][0]["row_count"] == 2
    assert result["sheets"][0]["columns"] == ["Product", "Price", "Quantity"]


def test_csv_text_representation_not_raw():
    from src.extraction.native_parsers import parse_csv

    csv_content = b"name,age\nAlice,30\nBob,25\n"
    result = parse_csv(csv_content, filename="test.csv")
    assert result["text"] != "name,age\nAlice,30\nBob,25\n"
    assert "columns" in result["text"].lower() or "name" in result["text"]


def test_is_native_parseable():
    from src.extraction.native_parsers import is_native_parseable

    assert is_native_parseable("data.csv") is True
    assert is_native_parseable("data.tsv") is True
    assert is_native_parseable("report.xlsx") is True
    assert is_native_parseable("report.xls") is True
    assert is_native_parseable("document.pdf") is False
    assert is_native_parseable("image.png") is False


def test_parse_native_routes_correctly():
    from src.extraction.native_parsers import parse_native

    csv_content = b"a,b\n1,2\n"
    result = parse_native(csv_content, filename="test.csv")
    assert result is not None
    assert result["parser"] == "native_csv"

    # Non-parseable returns None
    result = parse_native(b"hello world", filename="test.txt")
    assert result is None


def test_csv_empty_file():
    from src.extraction.native_parsers import parse_csv

    csv_content = b"col_a,col_b\n"
    result = parse_csv(csv_content, filename="empty.csv")
    assert result["row_count"] == 0
    assert result["columns"] == ["col_a", "col_b"]
    assert result["sample_rows"] == []


def test_csv_with_nulls():
    from src.extraction.native_parsers import parse_csv

    csv_content = b"name,age\nAlice,30\nBob,\n,25\n"
    result = parse_csv(csv_content, filename="nulls.csv")
    assert result["row_count"] == 3
    assert result["statistical_profile"]["age"]["null_count"] == 1
    assert result["statistical_profile"]["name"]["null_count"] == 1
