"""Tests for SME adapter admin endpoints."""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.sme_admin_api import AdapterAdminDeps, build_router

_YAML = """domain: finance
version: 1.2.0
persona: {role: x, voice: y, grounding_rules: []}
dossier: {section_weights: {a: 1.0}, prompt_template: p/x.md}
insight_detectors: []
comparison_axes: []
kg_inference_rules: []
recommendation_frames: []
response_persona_prompts: {diagnose: p/d.md, analyze: p/a.md, recommend: p/r.md}
retrieval_caps: {max_pack_tokens: {analyze: 6000, diagnose: 5000, recommend: 4500, investigate: 8000}}
output_caps: {analyze: 1200, diagnose: 1500, recommend: 1000, investigate: 2000}
"""
_H = {"Content-Type": "application/x-yaml"}


@pytest.fixture
def deps():
    return AdapterAdminDeps(loader=MagicMock(), blob_writer=MagicMock())


@pytest.fixture
def client(deps):
    app = FastAPI()
    app.include_router(build_router(deps))
    return TestClient(app)


def test_put_global_adapter(client, deps):
    r = client.put(
        "/admin/sme-adapters/global/finance", content=_YAML, headers=_H
    )
    assert r.status_code == 200
    assert (
        deps.blob_writer.write_text.call_args[0][0]
        == "sme_adapters/global/finance.yaml"
    )
    deps.loader.invalidate_all.assert_called()


def test_put_subscription_adapter(client, deps):
    r = client.put(
        "/admin/sme-adapters/sub/sub_a/finance", content=_YAML, headers=_H
    )
    assert r.status_code == 200
    assert (
        deps.blob_writer.write_text.call_args[0][0]
        == "sme_adapters/subscription/sub_a/finance.yaml"
    )
    deps.loader.invalidate.assert_called_with("sub_a", "finance")


def test_put_rejects_invalid_yaml(client):
    r = client.put(
        "/admin/sme-adapters/global/finance",
        content="not: valid: yaml",
        headers=_H,
    )
    assert r.status_code in (400, 422)


def test_put_rejects_mismatched_domain(client):
    # URL says legal but the YAML declares finance — reject.
    r = client.put(
        "/admin/sme-adapters/global/legal", content=_YAML, headers=_H
    )
    assert r.status_code == 400


def test_delete_adapter_invalidates_cache(client, deps):
    r = client.delete("/admin/sme-adapters/global/finance")
    assert r.status_code == 200
    deps.blob_writer.delete.assert_called_once_with(
        "sme_adapters/global/finance.yaml"
    )
    deps.loader.invalidate_all.assert_called()


def test_get_adapter_returns_parsed_json(client, deps):
    deps.blob_writer.read_text.return_value = _YAML
    r = client.get("/admin/sme-adapters/global/finance")
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == "sme_adapters/global/finance.yaml"
    assert body["adapter"]["domain"] == "finance"
    assert body["adapter"]["version"] == "1.2.0"


def test_invalidate_scoped_and_all(client, deps):
    r = client.post(
        "/admin/sme-adapters/invalidate",
        json={"subscription_id": "sub_a", "domain": "finance"},
    )
    assert r.status_code == 200
    deps.loader.invalidate.assert_called_with("sub_a", "finance")

    r2 = client.post("/admin/sme-adapters/invalidate", json={})
    assert r2.status_code == 200
    deps.loader.invalidate_all.assert_called_once()


def test_put_rejects_bad_scope(client):
    r = client.put(
        "/admin/sme-adapters/someweirdscope/finance",
        content=_YAML,
        headers=_H,
    )
    assert r.status_code == 400
