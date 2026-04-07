import pytest
from teams_app.pipeline.fast_path import classify_file, Pipeline


def test_csv_is_express():
    assert classify_file("report.csv") == Pipeline.EXPRESS

def test_txt_is_express():
    assert classify_file("notes.txt") == Pipeline.EXPRESS

def test_xlsx_is_express():
    assert classify_file("data.xlsx") == Pipeline.EXPRESS

def test_json_is_express():
    assert classify_file("config.json") == Pipeline.EXPRESS

def test_md_is_express():
    assert classify_file("readme.md") == Pipeline.EXPRESS

def test_xml_is_express():
    assert classify_file("feed.xml") == Pipeline.EXPRESS

def test_html_is_express():
    assert classify_file("page.html") == Pipeline.EXPRESS

def test_pdf_is_full():
    assert classify_file("document.pdf") == Pipeline.FULL

def test_docx_is_full():
    assert classify_file("letter.docx") == Pipeline.FULL

def test_pptx_is_full():
    assert classify_file("slides.pptx") == Pipeline.FULL

def test_image_is_full():
    assert classify_file("scan.png") == Pipeline.FULL
    assert classify_file("photo.jpg") == Pipeline.FULL
    assert classify_file("page.tiff") == Pipeline.FULL

def test_unknown_extension_is_full():
    assert classify_file("data.parquet") == Pipeline.FULL

def test_case_insensitive():
    assert classify_file("REPORT.CSV") == Pipeline.EXPRESS
    assert classify_file("Document.PDF") == Pipeline.FULL

def test_no_extension_is_full():
    assert classify_file("Makefile") == Pipeline.FULL

def test_should_escalate_short_text():
    from teams_app.pipeline.fast_path import should_escalate
    assert should_escalate("hi", min_chars=50) is True

def test_should_not_escalate_long_text():
    from teams_app.pipeline.fast_path import should_escalate
    assert should_escalate("x" * 100, min_chars=50) is False

def test_should_escalate_empty():
    from teams_app.pipeline.fast_path import should_escalate
    assert should_escalate("", min_chars=50) is True
