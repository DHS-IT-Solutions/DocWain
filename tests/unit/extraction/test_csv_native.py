from src.extraction.adapters.csv_native import extract_csv_native


def test_csv_native_extracts_comma_delimited():
    data = b"h1,h2,h3\n1,2,3\n4,5,6\n"
    result = extract_csv_native(data, doc_id="d1", filename="t.csv")
    assert result.format == "csv"
    assert result.path_taken == "native"
    assert len(result.pages) == 1
    assert len(result.pages[0].tables) == 1
    assert result.pages[0].tables[0].rows == [
        ["h1", "h2", "h3"],
        ["1", "2", "3"],
        ["4", "5", "6"],
    ]


def test_csv_native_handles_semicolon_dialect():
    data = b"a;b\n1;2\n"
    result = extract_csv_native(data, doc_id="d2", filename="t.csv")
    assert result.pages[0].tables[0].rows == [["a", "b"], ["1", "2"]]
