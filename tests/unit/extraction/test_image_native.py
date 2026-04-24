import pytest

from src.extraction.adapters.errors import NotNativePathError
from src.extraction.adapters.image_native import extract_image_native


def test_image_adapter_always_raises_not_native():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    with pytest.raises(NotNativePathError):
        extract_image_native(png, doc_id="d1", filename="pic.png")


def test_image_adapter_raises_for_jpg_and_tiff():
    with pytest.raises(NotNativePathError):
        extract_image_native(b"\xff\xd8\xff\xe0", doc_id="d2", filename="photo.jpg")
    with pytest.raises(NotNativePathError):
        extract_image_native(b"MM\x00*", doc_id="d3", filename="scan.tiff")
