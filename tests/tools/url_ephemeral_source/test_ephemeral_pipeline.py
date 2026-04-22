"""Lean tests for the ephemeral URL pipeline.

Covers three invariants: (1) happy-path returns chunks with ephemeral
metadata, (2) fetch failure is flagged as a warning without failing the
batch, (3) the module never touches any persistence client.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.tools.url_ephemeral_source import (
    EphemeralSource,
    UrlEphemeralSource,
)
from src.tools.url_fetcher import FetchResult, SsrfError


class _StubEmbedder:
    dim = 4

    def embed(self, texts):
        return [[0.1 * (i + 1)] * self.dim for i, _ in enumerate(texts)]

    def get_sentence_embedding_dimension(self):
        return self.dim


def _mk_result(url, body: bytes = b"<html><body>hello world</body></html>"):
    return FetchResult(
        url=url,
        final_url=url,
        status=200,
        headers={"content-type": "text/html"},
        body=body,
        content_type="text/html",
        resolved_ip="1.1.1.1",
        source_url=url,
        fetched_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Test 1: happy path returns chunks with ephemeral metadata + embeddings
# ---------------------------------------------------------------------------
def test_fetch_all_happy_path_returns_ephemeral_chunks():
    src = EphemeralSource(embedder=_StubEmbedder())
    with patch(
        "src.tools.url_ephemeral_source.fetch",
        side_effect=lambda u, **kw: _mk_result(u),
    ):
        res = src.fetch_all(
            ["https://example.com/"],
            subscription_id="sub1",
            profile_id="prof1",
            session_id="s1",
        )

    assert res.warnings == []
    assert len(res.chunks) >= 1
    for ch in res.chunks:
        assert ch.metadata["ephemeral"] is True
        assert ch.metadata["source_url"] == "https://example.com/"
        assert ch.metadata["subscription_id"] == "sub1"
        assert ch.metadata["profile_id"] == "prof1"
        assert ch.metadata["session_id"] == "s1"
        assert "fetched_at" in ch.metadata
        assert ch.embedding is not None
        assert len(ch.embedding) == _StubEmbedder.dim


# ---------------------------------------------------------------------------
# Test 2: fetch failure becomes a warning; batch continues
# ---------------------------------------------------------------------------
def test_fetch_failure_flagged_without_crashing_batch():
    def _handler(url, **_kw):
        if "bad" in url:
            raise SsrfError(f"blocked: {url}")
        return _mk_result(url)

    src = EphemeralSource(embedder=_StubEmbedder())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=_handler):
        res = src.fetch_all(
            [
                "https://good.example/",
                "https://bad.example/",
                "https://also-good.example/",
            ],
            session_id="s",
        )

    successful_urls = {c.metadata["source_url"] for c in res.chunks}
    assert successful_urls == {
        "https://good.example/",
        "https://also-good.example/",
    }
    assert len(res.warnings) == 1
    w = res.warnings[0]
    assert "bad.example" in w["url"]
    assert w["error_class"] == "SsrfError"


# ---------------------------------------------------------------------------
# Test 3: no-persistence invariant — no banned client module referenced
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("banned", [
    "qdrant_client",
    "neo4j",
    "pymongo",
    "azure.storage.blob",
    "redis",
])
def test_module_does_not_reference_persistence_clients(banned):
    import importlib.util
    spec = importlib.util.find_spec("src.tools.url_ephemeral_source")
    assert spec is not None and spec.origin
    with open(spec.origin) as f:
        source_text = f.read()
    assert banned not in source_text, (
        f"ephemeral source must not reference {banned}; found literal"
    )


# ---------------------------------------------------------------------------
# Test 4: alias check — UrlEphemeralSource and EphemeralSource point to same class
# ---------------------------------------------------------------------------
def test_alias_exposed_for_plan_compatibility():
    assert UrlEphemeralSource is EphemeralSource
