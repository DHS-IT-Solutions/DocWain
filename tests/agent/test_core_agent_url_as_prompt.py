"""Phase 5 — CoreAgent URL-as-prompt wiring tests.

These tests exercise the Phase-5-specific helpers directly:

* :meth:`CoreAgent._is_url_as_prompt_enabled` — flag gate (master + feature).
* :meth:`CoreAgent._build_fetcher_config` — subscription policy shape.
* :meth:`CoreAgent._kick_off_url_fetch` — parallel submission + merge path.

A full ``handle()``-level smoke is deferred to the Phase 6 end-to-end
trace harness; here we focus on the contract surface the agent exposes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.agent.core_agent import CoreAgent
from src.agent.url_case_selector import CaseSelection
from src.config.feature_flags import (
    ENABLE_URL_AS_PROMPT,
    SME_REDESIGN_ENABLED,
    init_flag_resolver,
    reset_flag_set_version,
)
from src.retrieval.ephemeral_merge import merge_ephemeral
from src.tools.url_ephemeral_source import EphemeralChunk, EphemeralResult
from src.tools.url_fetcher import FetcherConfig


class _MutableStore:
    def __init__(self, overrides: Dict[str, Dict[str, bool]] | None = None) -> None:
        self._by = overrides or {}

    def get_subscription_overrides(self, sub: str) -> Dict[str, bool]:
        return dict(self._by.get(sub, {}))

    def set_subscription_override(self, sub: str, flag: str, value: bool) -> None:
        self._by.setdefault(sub, {})[flag] = bool(value)


@pytest.fixture(autouse=True)
def _clean_slate():
    reset_flag_set_version()
    yield
    reset_flag_set_version()


def _build_agent() -> CoreAgent:
    llm = MagicMock()
    llm.backend = "test"
    return CoreAgent(
        llm_gateway=llm,
        qdrant_client=MagicMock(),
        embedder=MagicMock(),
        mongodb=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Test 1: flag off -> URL fetch is a no-op
# ---------------------------------------------------------------------------
def test_url_fetch_disabled_when_flag_off():
    # Master on, URL flag off.
    init_flag_resolver(store=_MutableStore(
        overrides={"sub1": {SME_REDESIGN_ENABLED: True}}
    ))
    agent = _build_agent()
    assert agent._is_url_as_prompt_enabled("sub1") is False


def test_url_fetch_disabled_when_master_off():
    # URL flag override but master is off — dependent flags return False.
    init_flag_resolver(store=_MutableStore(
        overrides={"sub1": {ENABLE_URL_AS_PROMPT: True}}
    ))
    agent = _build_agent()
    assert agent._is_url_as_prompt_enabled("sub1") is False


def test_url_fetch_enabled_when_both_flags_on():
    init_flag_resolver(store=_MutableStore(
        overrides={"sub1": {
            SME_REDESIGN_ENABLED: True, ENABLE_URL_AS_PROMPT: True,
        }}
    ))
    agent = _build_agent()
    assert agent._is_url_as_prompt_enabled("sub1") is True


# ---------------------------------------------------------------------------
# Test 2: fetcher config carries safe defaults + subscription policy hook
# ---------------------------------------------------------------------------
def test_build_fetcher_config_has_safe_defaults():
    init_flag_resolver(store=_MutableStore())
    agent = _build_agent()
    cfg = agent._build_fetcher_config("sub1")
    assert isinstance(cfg, FetcherConfig)
    assert cfg.max_size == 10_000_000
    assert cfg.fetch_timeout_s == 15.0
    assert cfg.extract_timeout_s == 30.0
    assert cfg.user_agent == "DocWain-URL-Fetcher/1.0"


# ---------------------------------------------------------------------------
# Test 3: end-to-end merge smoke for the supplementary case
# ---------------------------------------------------------------------------
def _make_chunk(url: str, text: str) -> EphemeralChunk:
    return EphemeralChunk(
        text=text,
        metadata={
            "ephemeral": True,
            "source_url": url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "chunk_index": 0,
        },
        embedding=[0.0] * 4,
    )


def test_kick_off_url_fetch_returns_mergeable_result():
    """Submit a fetch, merge results, verify supplementary placement."""
    init_flag_resolver(store=_MutableStore(
        overrides={"sub1": {
            SME_REDESIGN_ENABLED: True, ENABLE_URL_AS_PROMPT: True,
        }}
    ))
    agent = _build_agent()

    captured: Dict[str, Any] = {}

    def _fake_fetch_all(self_, urls, *, subscription_id, profile_id, session_id):
        captured["urls"] = list(urls)
        captured["subscription_id"] = subscription_id
        captured["profile_id"] = profile_id
        captured["session_id"] = session_id
        return EphemeralResult(
            chunks=[
                _make_chunk("https://ok.example/", "url-derived snippet"),
            ],
        )

    with patch(
        "src.tools.url_ephemeral_source.EphemeralSource.fetch_all",
        _fake_fetch_all,
    ):
        future = agent._kick_off_url_fetch(
            urls=["https://ok.example/"],
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess1",
        )
        result = future.result()

    assert captured["urls"] == ["https://ok.example/"]
    assert captured["subscription_id"] == "sub1"
    assert captured["profile_id"] == "prof1"
    assert len(result.chunks) == 1

    # Supplementary merge: profile first, then URL chunks.
    from types import SimpleNamespace
    profile_chunks = [SimpleNamespace(
        text="profile-1", score=0.9, metadata={}, document_id="d1", chunk_id="c1",
    )]
    merged = merge_ephemeral(
        profile_chunks, list(result.chunks), case=CaseSelection.SUPPLEMENTARY,
    )
    assert merged[0].text == "profile-1"
    assert merged[1].metadata["ephemeral"] is True
    assert merged[1].metadata["source_url"] == "https://ok.example/"


# ---------------------------------------------------------------------------
# Test 4: URL fetch errors fold into warnings without killing the pipeline
# ---------------------------------------------------------------------------
def test_url_fetch_future_catches_exceptions():
    init_flag_resolver(store=_MutableStore())
    agent = _build_agent()

    def _raise(self_, urls, **kw):
        raise RuntimeError("boom")

    with patch(
        "src.tools.url_ephemeral_source.EphemeralSource.fetch_all",
        _raise,
    ):
        future = agent._kick_off_url_fetch(
            urls=["https://ok.example/"],
            subscription_id="sub1",
            profile_id="prof1",
            session_id="sess1",
        )
        result = future.result()

    assert result.chunks == []
    assert len(result.warnings) == 1
    assert "boom" in result.warnings[0]["error"]
    assert result.warnings[0]["error_class"] == "RuntimeError"
