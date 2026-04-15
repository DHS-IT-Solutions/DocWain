import io
import pytest


def test_read_plain_text():
    from standalone.file_reader import read_file

    content = read_file("test.txt", b"Hello, this is a test document.")
    assert content == "Hello, this is a test document."


def test_read_json_file():
    from standalone.file_reader import read_file

    data = b'{"name": "DocWain", "version": 2}'
    content = read_file("data.json", data)
    assert "DocWain" in content


def test_read_csv_file():
    from standalone.file_reader import read_file

    csv_data = b"Name,Age,City\nAlice,30,London\nBob,25,Paris\n"
    content = read_file("people.csv", csv_data)
    assert "Alice" in content
    assert "Bob" in content
    assert "London" in content


def test_read_tsv_file():
    from standalone.file_reader import read_file

    tsv_data = b"Name\tAge\nAlice\t30\n"
    content = read_file("people.tsv", tsv_data)
    assert "Alice" in content
    assert "30" in content


def test_unsupported_format_raises():
    from standalone.file_reader import read_file, UnsupportedFileType

    with pytest.raises(UnsupportedFileType):
        read_file("archive.7z", b"\x37\x7a some 7z data")


def test_detect_type_by_extension():
    from standalone.file_reader import detect_file_type

    assert detect_file_type("report.pdf", b"") == "pdf"
    assert detect_file_type("doc.docx", b"") == "docx"
    assert detect_file_type("sheet.xlsx", b"") == "xlsx"
    assert detect_file_type("data.csv", b"") == "csv"
    assert detect_file_type("notes.txt", b"") == "text"
    assert detect_file_type("photo.png", b"") == "image"
    assert detect_file_type("photo.jpg", b"") == "image"


def test_detect_type_pdf_by_magic_bytes():
    from standalone.file_reader import detect_file_type

    assert detect_file_type("unknown.bin", b"%PDF-1.4 content") == "pdf"


def test_read_file_returns_string():
    from standalone.file_reader import read_file

    result = read_file("test.txt", b"some content")
    assert isinstance(result, str)


def test_read_file_metadata():
    from standalone.file_reader import read_file_with_metadata

    content, meta = read_file_with_metadata("test.csv", b"a,b\n1,2\n3,4\n")
    assert isinstance(content, str)
    assert meta["file_type"] == "csv"
    assert meta["size_bytes"] == len(b"a,b\n1,2\n3,4\n")


def test_read_image_returns_base64_data_uri():
    from standalone.file_reader import read_file

    # A tiny 1x1 PNG
    png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    content = read_file("photo.png", png_data)
    assert content.startswith("data:image/png;base64,")


def test_read_image_jpeg_mime():
    from standalone.file_reader import read_file

    content = read_file("photo.jpg", b"\xff\xd8\xff dummy jpeg")
    assert content.startswith("data:image/jpeg;base64,")


def test_image_metadata_has_pages():
    from standalone.file_reader import read_file_with_metadata

    _, meta = read_file_with_metadata("photo.png", b"\x89PNG dummy")
    assert meta["file_type"] == "image"
    assert meta["pages"] == 1
