import pytest

from src.intelligence.adapters.store import (
    AdapterStore,
    AdapterNotFound,
    FilesystemAdapterBackend,
    AzureBlobAdapterBackend,
    resolve_default_backend,
)


class FakeBlobBackend:
    """Minimal in-memory backend so tests don't touch Azure."""

    def __init__(self):
        self.store = {}

    def get_text(self, key: str) -> str:
        if key not in self.store:
            raise AdapterNotFound(key)
        return self.store[key]


def _generic_yaml() -> str:
    from pathlib import Path
    return Path("src/intelligence/adapters/generic.yaml").read_text()


def _stub_yaml(name: str, version: str = "1.0") -> str:
    return (
        f"name: {name}\n"
        f"version: '{version}'\n"
        "description: x\n"
        "applies_when: {}\n"
        "researcher:\n  insight_types: {}\n"
        "knowledge:\n  sanctioned_kbs: []\n  citation_rule: doc_grounded_first\n"
        "watchlists: []\n"
        "actions: []\n"
        "visualizations: []\n"
    )


def test_global_only_resolution():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/insurance.yaml"] = _stub_yaml("insurance")
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=60)
    a = store.get(domain="insurance", subscription_id="sub-x")
    assert a.name == "insurance"


def test_subscription_overrides_global():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/insurance.yaml"] = _stub_yaml("insurance", "1.0")
    backend.store["sme_adapters/subscription/sub-x/insurance.yaml"] = _stub_yaml(
        "insurance", "2.0-tenant"
    )
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=60)
    a = store.get(domain="insurance", subscription_id="sub-x")
    assert a.version == "2.0-tenant"


def test_falls_back_to_generic_when_unknown_domain():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=60)
    a = store.get(domain="moonbase_logistics", subscription_id="sub-x")
    assert a.name == "generic"


def test_ttl_cache_reuses_within_ttl():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=60)
    a1 = store.get(domain="generic", subscription_id="sub-x")
    a2 = store.get(domain="generic", subscription_id="sub-x")
    assert a1 is a2


def test_invalidate_forces_reload():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=60)
    a1 = store.get(domain="generic", subscription_id="sub-x")
    store.invalidate(domain="generic")
    a2 = store.get(domain="generic", subscription_id="sub-x")
    assert a1 is not a2


def test_blob_failure_falls_back_to_cached():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=0)
    a1 = store.get(domain="generic", subscription_id="sub-x")

    def boom(key):
        raise RuntimeError("blob down")
    backend.get_text = boom
    a2 = store.get(domain="generic", subscription_id="sub-x")
    assert a2 is not None
    assert a2.name == "generic"


def test_filesystem_backend_reads_local_file(tmp_path):
    root = tmp_path
    (root / "sme_adapters" / "global").mkdir(parents=True)
    (root / "sme_adapters" / "global" / "generic.yaml").write_text(_generic_yaml())
    backend = FilesystemAdapterBackend(root=str(root))
    text = backend.get_text("sme_adapters/global/generic.yaml")
    assert "name: generic" in text


def test_filesystem_backend_missing_raises(tmp_path):
    backend = FilesystemAdapterBackend(root=str(tmp_path))
    with pytest.raises(AdapterNotFound):
        backend.get_text("sme_adapters/global/nope.yaml")


def test_blob_backend_is_disabled_when_flag_off(monkeypatch):
    monkeypatch.delenv("ADAPTER_BLOB_LOADING_ENABLED", raising=False)
    backend = resolve_default_backend(blob_root="/tmp/x")
    assert isinstance(backend, FilesystemAdapterBackend)


def test_blob_backend_active_when_flag_on(monkeypatch, tmp_path):
    monkeypatch.setenv("ADAPTER_BLOB_LOADING_ENABLED", "true")
    monkeypatch.setenv("ADAPTER_BLOB_CONTAINER", "fake")
    monkeypatch.setenv("ADAPTER_BLOB_CONNECTION", "fake")
    backend = resolve_default_backend(blob_root=str(tmp_path))
    assert isinstance(backend, AzureBlobAdapterBackend)
