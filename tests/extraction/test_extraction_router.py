"""Tests for smart extraction routing."""

from src.extraction.extraction_router import route_extraction


def test_csv_routes_to_native():
    route = route_extraction("data.csv", 1024)
    assert route.method == "native_structured"


def test_tsv_routes_to_native():
    route = route_extraction("data.tsv", 2048)
    assert route.method == "native_structured"


def test_excel_routes_to_native():
    route = route_extraction("report.xlsx", 50000)
    assert route.method == "native_structured"


def test_xls_routes_to_native():
    route = route_extraction("legacy.xls", 30000)
    assert route.method == "native_structured"


def test_small_pdf_routes_to_standard():
    route = route_extraction("invoice.pdf", 100_000)  # ~2 pages
    assert route.method == "standard"


def test_medium_pdf_routes_to_parallel():
    route = route_extraction("report.pdf", 2_000_000)  # ~40 pages
    assert route.method == "parallel_pages"


def test_large_pdf_routes_to_bulk():
    route = route_extraction("manual.pdf", 10_000_000)  # ~200 pages
    assert route.method == "background_bulk"


def test_image_routes_to_vision():
    route = route_extraction("scan.png", 500_000)
    assert route.method == "vision"


def test_jpg_routes_to_vision():
    route = route_extraction("photo.jpg", 300_000)
    assert route.method == "vision"


def test_jpeg_routes_to_vision():
    route = route_extraction("photo.jpeg", 300_000)
    assert route.method == "vision"


def test_tiff_routes_to_vision():
    route = route_extraction("scan.tiff", 1_000_000)
    assert route.method == "vision"


def test_docx_routes_to_standard():
    route = route_extraction("contract.docx", 200_000)
    assert route.method == "standard"


def test_txt_routes_to_standard():
    route = route_extraction("notes.txt", 5000)
    assert route.method == "standard"


def test_md_routes_to_standard():
    route = route_extraction("readme.md", 3000)
    assert route.method == "standard"


def test_unknown_defaults_to_standard():
    route = route_extraction("data.xyz", 100_000)
    assert route.method == "standard"


def test_route_has_reason():
    route = route_extraction("data.csv", 1024)
    assert route.reason  # non-empty string


def test_route_has_positive_estimate():
    route = route_extraction("report.pdf", 2_000_000)
    assert route.estimated_seconds > 0


def test_pdf_boundary_at_5_pages():
    # Exactly 5 pages (250KB) should go to parallel_pages
    route = route_extraction("doc.pdf", 5 * 50_000)
    assert route.method == "parallel_pages"


def test_pdf_just_under_5_pages():
    # Just under 5 pages should stay standard
    route = route_extraction("doc.pdf", 4 * 50_000)
    assert route.method == "standard"
