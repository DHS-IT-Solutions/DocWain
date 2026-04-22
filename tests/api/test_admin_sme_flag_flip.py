"""Admin endpoint tests for Phase 3 per-subscription flag flips.

Covers the Phase 3 Task 2 surface:

* ``PATCH /admin/sme-flags/{subscription_id}`` — flip a single flag
* ``GET   /admin/sme-flags/{subscription_id}`` — inspect current overrides

Auth is attached at mount-level by ``src.main`` in production; these tests
exercise the router factory directly so they do not require a live auth
stack (same convention as :mod:`tests.api.test_sme_admin_api`).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.sme_admin_api import FlagAdminDeps, build_flag_router
from src.config import feature_flags as ff
from src.config.feature_flags import (
    ENABLE_CROSS_ENCODER_RERANK,
    ENABLE_HYBRID_RETRIEVAL,
    ENABLE_KG_SYNTHESIZED_EDGES,
    ENABLE_SME_RETRIEVAL,
    SME_REDESIGN_ENABLED,
    init_flag_resolver,
    reset_flag_set_version,
)


# ---------------------------------------------------------------------------
# In-memory MutableFlagStore stub — per-subscription override dict.
# ---------------------------------------------------------------------------


class _InMemoryMutableStore:
    """Satisfies the :class:`MutableFlagStore` Protocol for tests."""

    def __init__(self) -> None:
        self._by_sub: dict[str, dict[str, bool]] = {}

    def get_subscription_overrides(self, subscription_id: str) -> dict[str, bool]:
        return dict(self._by_sub.get(subscription_id, {}))

    def set_subscription_override(
        self, subscription_id: str, flag: str, value: bool
    ) -> None:
        self._by_sub.setdefault(subscription_id, {})[flag] = bool(value)


class _ReadOnlyStore:
    """Store without the setter — exercises the 500-response path."""

    def get_subscription_overrides(self, subscription_id: str) -> dict[str, bool]:
        return {}


@pytest.fixture(autouse=True)
def _reset_flag_set_version() -> None:
    reset_flag_set_version()
    yield
    reset_flag_set_version()


@pytest.fixture
def store() -> _InMemoryMutableStore:
    return _InMemoryMutableStore()


@pytest.fixture
def audit_log() -> list[dict]:
    return []


@pytest.fixture
def client(store: _InMemoryMutableStore, audit_log: list[dict]) -> TestClient:
    # Initialize the process-wide resolver against the same store so the
    # endpoint can compute the ``effective`` value on the response.
    init_flag_resolver(store=store)

    def _writer(entry: dict) -> None:
        audit_log.append(entry)

    deps = FlagAdminDeps(store=store, audit_writer=_writer)
    app = FastAPI()
    app.include_router(build_flag_router(deps))
    return TestClient(app)


def test_flip_enable_sme_retrieval_on(client, store, audit_log) -> None:
    # Master must be on for the dependent flag to resolve True end-to-end.
    r_master = client.patch(
        "/admin/sme-flags/sub_fin_1",
        json={"flag": SME_REDESIGN_ENABLED, "enabled": True, "reason": "rollout"},
    )
    assert r_master.status_code == 200, r_master.text
    body_master = r_master.json()
    assert body_master["flag"] == SME_REDESIGN_ENABLED
    assert body_master["new_value"] is True
    assert body_master["effective"] is True

    r = client.patch(
        "/admin/sme-flags/sub_fin_1",
        json={"flag": ENABLE_SME_RETRIEVAL, "enabled": True, "reason": "opt-in fin"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["subscription_id"] == "sub_fin_1"
    assert body["flag"] == ENABLE_SME_RETRIEVAL
    assert body["prior_value"] is False
    assert body["new_value"] is True
    # Master ON + dependent ON → effective True.
    assert body["effective"] is True
    # Monotonic counter advanced with each flip.
    assert body["flag_set_version"] > body_master["flag_set_version"]

    # Audit log captured both mutations.
    assert len(audit_log) == 2
    assert audit_log[-1]["flag"] == ENABLE_SME_RETRIEVAL
    assert audit_log[-1]["new_value"] is True

    # Persisted to the store (read back via GET).
    r_get = client.get("/admin/sme-flags/sub_fin_1")
    assert r_get.status_code == 200
    overrides = r_get.json()["overrides"]
    assert overrides[ENABLE_SME_RETRIEVAL] is True
    assert overrides[SME_REDESIGN_ENABLED] is True


def test_flip_rejects_unknown_flag(client) -> None:
    r = client.patch(
        "/admin/sme-flags/sub_fin_1",
        json={"flag": "made_up_flag", "enabled": True},
    )
    assert r.status_code == 400
    assert "unknown flag" in r.json()["detail"].lower()


def test_flip_rejects_blank_subscription(client) -> None:
    # Path segments are URL-decoded; a literal space becomes a blank id.
    r = client.patch(
        "/admin/sme-flags/%20",
        json={"flag": ENABLE_SME_RETRIEVAL, "enabled": True},
    )
    assert r.status_code == 400


def test_flip_writes_audit_log_with_prior_value(client, store, audit_log) -> None:
    # Seed an override directly, then flip via the endpoint.
    store.set_subscription_override("sub_a", ENABLE_HYBRID_RETRIEVAL, True)
    r = client.patch(
        "/admin/sme-flags/sub_a",
        json={"flag": ENABLE_HYBRID_RETRIEVAL, "enabled": False, "reason": "rollback"},
    )
    assert r.status_code == 200
    assert r.json()["prior_value"] is True
    assert r.json()["new_value"] is False
    assert audit_log[-1]["prior_value"] is True
    assert audit_log[-1]["new_value"] is False
    assert audit_log[-1]["reason"] == "rollback"


def test_get_returns_empty_when_no_overrides(client) -> None:
    r = client.get("/admin/sme-flags/sub_empty")
    assert r.status_code == 200
    assert r.json() == {"subscription_id": "sub_empty", "overrides": {}}


def test_get_returns_current_overrides(client, store) -> None:
    store.set_subscription_override("sub_x", SME_REDESIGN_ENABLED, True)
    store.set_subscription_override("sub_x", ENABLE_CROSS_ENCODER_RERANK, True)
    r = client.get("/admin/sme-flags/sub_x")
    assert r.status_code == 200
    overrides = r.json()["overrides"]
    assert overrides == {
        SME_REDESIGN_ENABLED: True,
        ENABLE_CROSS_ENCODER_RERANK: True,
    }


def test_flip_off_disables_layer_c(client, store) -> None:
    # Seed master ON + flag ON via the admin endpoint, then flip flag OFF.
    for flag in (SME_REDESIGN_ENABLED, ENABLE_SME_RETRIEVAL):
        client.patch(
            "/admin/sme-flags/sub_fin_1",
            json={"flag": flag, "enabled": True, "reason": "seed"},
        )

    r = client.patch(
        "/admin/sme-flags/sub_fin_1",
        json={"flag": ENABLE_SME_RETRIEVAL, "enabled": False, "reason": "rollback"},
    )
    assert r.status_code == 200
    assert r.json()["prior_value"] is True
    assert r.json()["new_value"] is False
    assert r.json()["effective"] is False


def test_all_phase3_flippable_flags_work(client) -> None:
    # Phase 3 flips must work for the full retrieval-side set.
    for flag in (
        SME_REDESIGN_ENABLED,
        ENABLE_SME_RETRIEVAL,
        ENABLE_KG_SYNTHESIZED_EDGES,
        ENABLE_HYBRID_RETRIEVAL,
        ENABLE_CROSS_ENCODER_RERANK,
    ):
        r = client.patch(
            "/admin/sme-flags/sub_multi",
            json={"flag": flag, "enabled": True},
        )
        assert r.status_code == 200, (flag, r.text)


def test_store_without_setter_returns_500() -> None:
    # Swap the resolver's store to one missing set_subscription_override.
    init_flag_resolver(store=_ReadOnlyStore())
    deps = FlagAdminDeps(store=_ReadOnlyStore(), audit_writer=None)
    app = FastAPI()
    app.include_router(build_flag_router(deps))
    tc = TestClient(app)
    r = tc.patch(
        "/admin/sme-flags/sub_x",
        json={"flag": SME_REDESIGN_ENABLED, "enabled": True},
    )
    assert r.status_code == 500


def test_flag_set_version_bumped_on_mutation(client) -> None:
    # The monotonic counter advances with every PATCH so downstream caches
    # keyed on ``flag_set_version`` (Phase 3 Task 8) invalidate naturally.
    r1 = client.patch(
        "/admin/sme-flags/sub_v",
        json={"flag": SME_REDESIGN_ENABLED, "enabled": True},
    )
    r2 = client.patch(
        "/admin/sme-flags/sub_v",
        json={"flag": ENABLE_HYBRID_RETRIEVAL, "enabled": True},
    )
    r3 = client.patch(
        "/admin/sme-flags/sub_v",
        json={"flag": ENABLE_HYBRID_RETRIEVAL, "enabled": False},
    )
    v1 = r1.json()["flag_set_version"]
    v2 = r2.json()["flag_set_version"]
    v3 = r3.json()["flag_set_version"]
    assert v1 < v2 < v3
    assert ff.get_flag_set_version() == v3
