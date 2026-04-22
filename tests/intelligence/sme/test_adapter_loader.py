"""Tests for AdapterLoader: Blob fetch, TTL cache, last-resort fallback."""
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme import adapter_loader as _al_mod
from src.intelligence.sme.adapter_loader import (
    AdapterLoader,
    AdapterLoadError,
    BlobReader,
    get_adapter_loader,
    init_adapter_loader,
)
from src.intelligence.sme.adapter_schema import Adapter

_GENERIC = """
domain: generic
version: 1.0.0
persona: {role: smex, voice: neutral, grounding_rules: [cite sources]}
dossier: {section_weights: {overview: 0.5, findings: 0.5}, prompt_template: p/g.md}
insight_detectors: []
comparison_axes: []
kg_inference_rules: []
recommendation_frames: []
response_persona_prompts: {diagnose: p/d.md, analyze: p/a.md, recommend: p/r.md}
retrieval_caps: {max_pack_tokens: {analyze: 6000, diagnose: 5000, recommend: 4500, investigate: 8000}}
output_caps: {analyze: 1200, diagnose: 1500, recommend: 1000, investigate: 2000}
"""


@pytest.fixture
def blob():
    b = MagicMock(spec=BlobReader)
    b.read_text.return_value = _GENERIC
    return b


@pytest.fixture
def lr(tmp_path: Path) -> Path:
    p = tmp_path / "generic.yaml"
    p.write_text(_GENERIC)
    return p


def _L(blob, lr, ttl=60):
    return AdapterLoader(blob=blob, last_resort_path=lr, ttl_seconds=ttl)


def test_resolves_subscription_override_first(blob, lr):
    _L(blob, lr).load("sub_a", "finance")
    assert (
        blob.read_text.call_args_list[0][0][0]
        == "sme_adapters/subscription/sub_a/finance.yaml"
    )


def test_falls_through_to_global(blob, lr):
    def side(p):
        if p.startswith("sme_adapters/subscription/"):
            raise FileNotFoundError(p)
        return _GENERIC

    blob.read_text.side_effect = side
    assert isinstance(_L(blob, lr).load("sub_a", "finance"), Adapter)


def test_falls_through_to_generic(blob, lr):
    def side(p):
        if "mystery" in p:
            raise FileNotFoundError(p)
        return _GENERIC

    blob.read_text.side_effect = side
    assert _L(blob, lr).load("sub_a", "mystery").domain == "generic"


def test_ttl_cache_hits(blob, lr):
    loader = _L(blob, lr, ttl=60)
    loader.load("sub_a", "finance")
    loader.load("sub_a", "finance")
    assert blob.read_text.call_count == 1


def test_ttl_cache_expires(blob, lr):
    loader = _L(blob, lr, ttl=0.01)
    loader.load("sub_a", "finance")
    time.sleep(0.05)
    loader.load("sub_a", "finance")
    assert blob.read_text.call_count == 2


def test_invalidate_forces_refetch(blob, lr):
    loader = _L(blob, lr, ttl=3600)
    loader.load("sub_a", "finance")
    loader.invalidate("sub_a", "finance")
    loader.load("sub_a", "finance")
    assert blob.read_text.call_count == 2


def test_blob_unreachable_uses_last_resort(lr):
    bad = MagicMock(spec=BlobReader)
    bad.read_text.side_effect = ConnectionError("blob down")
    loader = AdapterLoader(blob=bad, last_resort_path=lr, ttl_seconds=60)
    assert loader.load("sub_a", "finance").domain == "generic"
    assert loader.health_status() == "degraded"


def test_records_version_and_hash(blob, lr):
    loader = _L(blob, lr)
    adapter = loader.load("sub_a", "finance")
    meta = loader.last_load_metadata("sub_a", "finance")
    assert meta["version"] == "1.0.0" and meta["content_hash"]
    # ERRATA §1: the adapter instance itself carries content_hash + source_path.
    assert adapter.content_hash == meta["content_hash"]
    assert adapter.source_path == meta["source_path"]
    assert adapter.version == "1.0.0"


def test_missing_last_resort_raises(blob, tmp_path):
    with pytest.raises(AdapterLoadError):
        AdapterLoader(
            blob=blob, last_resort_path=tmp_path / "nope.yaml", ttl_seconds=60
        )


def test_get_alias_matches_load(blob, lr):
    loader = _L(blob, lr)
    # ERRATA §1: `.get` is a kept alias for `.load`. Bound methods are fresh
    # descriptor objects each access, so compare the underlying function.
    assert loader.get.__func__ is loader.load.__func__
    # And verify behavioural equivalence — calling either returns an Adapter.
    assert isinstance(loader.get("sub_a", "finance"), Adapter)


def test_get_adapter_loader_requires_init(blob, lr):
    # Reset singleton to ensure clean state.
    _al_mod._adapter_loader_singleton = None
    with pytest.raises(RuntimeError, match="not initialized"):
        get_adapter_loader()


def test_init_and_get_adapter_loader(blob, lr):
    _al_mod._adapter_loader_singleton = None
    loader = init_adapter_loader(
        blob=blob, last_resort_path=lr, ttl_seconds=60
    )
    assert get_adapter_loader() is loader
    # Clean up for other tests.
    _al_mod._adapter_loader_singleton = None
