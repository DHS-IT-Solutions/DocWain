"""Domain adapter loader — Blob fetch, TTL cache, generic fallback."""
import time
from unittest.mock import MagicMock

import pytest

from src.docwain.adapters.loader import AdapterLoader
from src.docwain.adapters.schema import DomainAdapter


def test_generic_seed_loaded_when_blob_unavailable(monkeypatch):
    """If Blob fetch raises, loader returns the baked-in generic adapter."""
    loader = AdapterLoader(subscription_id="sub-x")
    # Monkeypatch _fetch_from_blob to raise (simulate Blob down or not configured).
    monkeypatch.setattr(loader, "_fetch_from_blob", lambda path: (_ for _ in ()).throw(RuntimeError("blob down")))
    adapter = loader.load("generic")
    assert isinstance(adapter, DomainAdapter)
    assert adapter.domain == "generic"
    assert adapter.prompt_fragment  # non-empty


def test_unknown_domain_falls_back_to_generic(monkeypatch):
    """Asking for 'alien_domain' when only generic exists → returns generic adapter."""
    loader = AdapterLoader(subscription_id="sub-x")
    monkeypatch.setattr(loader, "_fetch_from_blob", lambda path: (_ for _ in ()).throw(FileNotFoundError(path)))
    adapter = loader.load("alien_domain")
    assert adapter.domain == "generic"


def test_subscription_override_tried_first(monkeypatch):
    """Loader tries {BLOB_PREFIX}/{sub_id}/{domain}.yaml before /global/{domain}.yaml."""
    loader = AdapterLoader(subscription_id="sub-x")

    paths_tried = []

    def fake_fetch(path):
        paths_tried.append(path)
        raise FileNotFoundError(path)  # all paths fail → generic fallback

    monkeypatch.setattr(loader, "_fetch_from_blob", fake_fetch)
    loader.load("finance")
    assert any("sub-x" in p and "finance" in p for p in paths_tried), paths_tried
    assert any("global" in p and "finance" in p for p in paths_tried), paths_tried


def test_cache_honors_ttl(monkeypatch):
    """Second load within TTL returns the cached adapter without re-fetching."""
    loader = AdapterLoader(subscription_id="sub-x", cache_ttl_seconds=10)
    call_count = {"n": 0}

    def fake_fetch(path):
        call_count["n"] += 1
        if "generic" in path:
            # Return the YAML text for a minimal generic adapter
            return "domain: generic\nversion: v1\nprompt_fragment: hi\nkey_entities: []\nanalysis_hints: {}\nquestions_to_ask: []\n"
        raise FileNotFoundError(path)

    monkeypatch.setattr(loader, "_fetch_from_blob", fake_fetch)
    a1 = loader.load("generic")
    a2 = loader.load("generic")
    assert a1 is a2 or (a1.domain == a2.domain)
    # Only called once — second load hit the cache
    # (call_count may be >1 for sub-specific then global path attempts; we just assert no third call on second load)
    count_after_first = call_count["n"]
    loader.load("generic")
    assert call_count["n"] == count_after_first  # no additional fetches on cached load


def test_parses_yaml_into_dataclass(monkeypatch):
    loader = AdapterLoader(subscription_id="sub-x")
    yaml_text = (
        "domain: finance\nversion: v2\n"
        "prompt_fragment: Focus on financial details.\n"
        "key_entities: [person, money]\n"
        "analysis_hints: {summary_style: concise}\n"
        "questions_to_ask: [What is the total?]\n"
    )

    def fake_fetch(path):
        if "finance" in path:
            return yaml_text
        raise FileNotFoundError(path)

    monkeypatch.setattr(loader, "_fetch_from_blob", fake_fetch)
    a = loader.load("finance")
    assert a.domain == "finance"
    assert a.version == "v2"
    assert "Focus on financial" in a.prompt_fragment
    assert "person" in a.key_entities
    assert a.analysis_hints.get("summary_style") == "concise"
    assert any("total" in q.lower() for q in a.questions_to_ask)
