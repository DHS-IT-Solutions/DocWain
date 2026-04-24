import base64

import httpx
import pytest

from src.extraction.vision.client import VisionClient, VisionClientError


def _png_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )


def test_vision_client_builds_openai_vision_payload():
    client = VisionClient(base_url="http://localhost:8100/v1", model="docwain-fast")
    payload = client.build_payload(
        system="You are a document extractor.",
        user_text="Extract all text.",
        image_bytes=_png_bytes(),
        max_tokens=512,
        temperature=0.0,
    )
    assert payload["model"] == "docwain-fast"
    assert payload["max_tokens"] == 512
    assert payload["temperature"] == 0.0
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    user = messages[1]
    assert user["role"] == "user"
    parts = user["content"]
    assert any(p.get("type") == "text" and "Extract all text" in p.get("text", "") for p in parts)
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    url = image_parts[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_vision_client_handles_http_error(monkeypatch):
    client = VisionClient(base_url="http://localhost:8100/v1", model="docwain-fast")

    class FakeResponse:
        status_code = 500

        def raise_for_status(self):
            raise httpx.HTTPStatusError("boom", request=None, response=None)

        def json(self):  # pragma: no cover
            raise AssertionError("json() should not be reached on error")

    def fake_post(url, json, timeout):
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(VisionClientError):
        client.call(system="s", user_text="u", image_bytes=_png_bytes())
