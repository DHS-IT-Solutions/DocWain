"""Admin endpoints for SME adapter YAMLs (canonical filename per ERRATA §19).

Endpoints — admin auth attaches at mount-level via existing middleware; this
module is intentionally auth-agnostic so tests don't need to fake the full
auth stack:

* ``PUT    /admin/sme-adapters/global/{domain}``        — upload global YAML
* ``PUT    /admin/sme-adapters/sub/{sub_id}/{domain}``  — upload subscription override
* ``DELETE /admin/sme-adapters/global/{domain}``        — delete global YAML
* ``DELETE /admin/sme-adapters/sub/{sub_id}/{domain}``  — delete subscription override
* ``GET    /admin/sme-adapters/global/{domain}``        — fetch parsed JSON + version + hash
* ``GET    /admin/sme-adapters/sub/{sub_id}/{domain}``  — fetch subscription override
* ``POST   /admin/sme-adapters/invalidate``             — body ``{subscription_id?, domain?}``

An empty ``POST /invalidate`` body invalidates the entire cache; specifying
``{subscription_id, domain}`` drops just that entry.

Phase 3 adds the per-subscription feature-flag flip endpoints used by the
retrieval-layer rollout:

* ``PATCH  /admin/sme-flags/{subscription_id}``         — body ``{flag, enabled}``
* ``GET    /admin/sme-flags/{subscription_id}``         — current overrides

Per Phase 1 scope: no production path consumes the loader yet — the router is
registered in app lifespan but dormant until Phase 2 wires synthesis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import yaml
from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, Field

from src.config.feature_flags import (
    all_flag_names,
    bump_flag_set_version,
    get_flag_resolver,
)
from src.intelligence.sme.adapter_loader import AdapterLoader
from src.intelligence.sme.adapter_schema import Adapter

logger = logging.getLogger(__name__)


class BlobWriter(Protocol):
    """Minimal Blob surface used by admin endpoints. Consistent with
    :class:`src.intelligence.sme.adapter_loader.BlobReader` — any Blob client
    that exposes ``read_text`` / ``write_text`` / ``delete`` satisfies both."""

    def write_text(self, path: str, content: str) -> None: ...
    def delete(self, path: str) -> None: ...
    def read_text(self, path: str) -> str: ...


@dataclass
class AdapterAdminDeps:
    """Router dependencies: the shared :class:`AdapterLoader` singleton and a
    Blob writer for persisting YAML mutations."""

    loader: AdapterLoader
    blob_writer: BlobWriter


def _global_path(domain: str) -> str:
    return f"sme_adapters/global/{domain}.yaml"


def _subscription_path(sub_id: str, domain: str) -> str:
    return f"sme_adapters/subscription/{sub_id}/{domain}.yaml"


def _parse_adapter(raw: str) -> Adapter:
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:  # bad YAML syntax
        raise HTTPException(400, f"YAML parse error: {exc}")
    if not isinstance(data, dict):
        raise HTTPException(400, "YAML body must be a mapping")
    try:
        return Adapter(**data)
    except Exception as exc:  # pydantic ValidationError or bad key types
        raise HTTPException(422, f"Adapter validation failed: {exc}")


def build_router(deps: AdapterAdminDeps) -> APIRouter:
    """Construct the admin APIRouter bound to ``deps``.

    We keep this as a factory (rather than a module-level router with globals)
    so tests can inject mocks and the FastAPI lifespan can wire real blob /
    loader singletons without circular imports.
    """

    r = APIRouter(prefix="/admin/sme-adapters", tags=["sme-admin"])

    # --------------------- global scope ---------------------
    @r.put("/global/{domain}")
    async def put_global(domain: str, request: Request) -> dict:
        raw = (await request.body()).decode("utf-8")
        adapter = _parse_adapter(raw)
        if adapter.domain != domain:
            raise HTTPException(
                400,
                f"URL domain {domain!r} != YAML domain {adapter.domain!r}",
            )
        path = _global_path(domain)
        deps.blob_writer.write_text(path, raw)
        # Global change invalidates every subscription's cached copy.
        deps.loader.invalidate_all()
        return {"status": "ok", "path": path, "version": adapter.version}

    @r.delete("/global/{domain}")
    async def delete_global(domain: str) -> dict:
        path = _global_path(domain)
        deps.blob_writer.delete(path)
        deps.loader.invalidate_all()
        return {"status": "ok", "path": path}

    @r.get("/global/{domain}")
    async def get_global(domain: str) -> dict:
        path = _global_path(domain)
        raw = deps.blob_writer.read_text(path)
        return {
            "path": path,
            "adapter": Adapter(**yaml.safe_load(raw)).model_dump(mode="json"),
        }

    # --------------------- subscription scope ---------------------
    @r.put("/sub/{sub_id}/{domain}")
    async def put_subscription(
        sub_id: str, domain: str, request: Request
    ) -> dict:
        raw = (await request.body()).decode("utf-8")
        adapter = _parse_adapter(raw)
        if adapter.domain != domain:
            raise HTTPException(
                400,
                f"URL domain {domain!r} != YAML domain {adapter.domain!r}",
            )
        path = _subscription_path(sub_id, domain)
        deps.blob_writer.write_text(path, raw)
        # Scoped invalidation: this subscription + domain only.
        deps.loader.invalidate(sub_id, domain)
        return {"status": "ok", "path": path, "version": adapter.version}

    @r.delete("/sub/{sub_id}/{domain}")
    async def delete_subscription(sub_id: str, domain: str) -> dict:
        path = _subscription_path(sub_id, domain)
        deps.blob_writer.delete(path)
        deps.loader.invalidate(sub_id, domain)
        return {"status": "ok", "path": path}

    @r.get("/sub/{sub_id}/{domain}")
    async def get_subscription(sub_id: str, domain: str) -> dict:
        path = _subscription_path(sub_id, domain)
        raw = deps.blob_writer.read_text(path)
        return {
            "path": path,
            "adapter": Adapter(**yaml.safe_load(raw)).model_dump(mode="json"),
        }

    # --------------------- invalidate ---------------------
    @r.post("/invalidate")
    async def invalidate(body: dict = Body(default={})) -> dict:
        sub = body.get("subscription_id")
        domain = body.get("domain")
        if sub and domain:
            deps.loader.invalidate(sub, domain)
            return {"status": "ok", "scope": f"{sub}/{domain}"}
        deps.loader.invalidate_all()
        return {"status": "ok", "scope": "all"}

    # --------------------- explicit bad-scope fallback ---------------------
    # Keep a catch-all handler so a typo like ``/admin/sme-adapters/nope/x``
    # returns HTTP 400 (the plan's Task 4 contract) rather than the FastAPI
    # default 404. We register after all real routes so FastAPI's normal
    # dispatch picks the specific handlers first.
    @r.put("/{scope}/{domain}")
    async def put_unknown(scope: str, domain: str) -> dict:
        raise HTTPException(400, f"Invalid scope {scope!r}")

    @r.delete("/{scope}/{domain}")
    async def delete_unknown(scope: str, domain: str) -> dict:
        raise HTTPException(400, f"Invalid scope {scope!r}")

    @r.get("/{scope}/{domain}")
    async def get_unknown(scope: str, domain: str) -> dict:
        raise HTTPException(400, f"Invalid scope {scope!r}")

    return r


# ---------------------------------------------------------------------------
# Phase 3 — per-subscription feature-flag flip endpoints
# ---------------------------------------------------------------------------


class FlagFlipBody(BaseModel):
    """Request body for ``PATCH /admin/sme-flags/{subscription_id}``.

    ``flag`` must be one of the 8 canonical flag names (validated at the
    endpoint against :func:`src.config.feature_flags.all_flag_names`). Any
    unknown name returns HTTP 400 rather than silently persisting.
    """

    flag: str = Field(..., min_length=1)
    enabled: bool
    reason: str | None = Field(default=None, max_length=512)


@dataclass
class FlagAdminDeps:
    """Router dependencies for the flag-flip admin surface.

    ``store`` must implement :class:`src.config.feature_flags.MutableFlagStore`
    (``get_subscription_overrides`` + ``set_subscription_override``). Tests
    pass an in-memory stub; production wires MongoDB.

    ``audit_writer`` receives a dict per mutation so the control plane can
    persist an audit log (operator, flag, prior value, new value, timestamp).
    Leaving it ``None`` is allowed — the endpoint then only logs via the
    module logger and still applies the mutation.
    """

    store: Any  # MutableFlagStore — Protocol check at call time
    audit_writer: Any | None = None


def build_flag_router(deps: FlagAdminDeps) -> APIRouter:
    """Construct the Phase 3 admin router for per-subscription flag flips.

    Endpoints:

    * ``PATCH /admin/sme-flags/{subscription_id}`` — body ``{flag, enabled,
      reason?}``. Validates ``flag`` against the 8-name canonical set,
      persists the override via ``deps.store.set_subscription_override``,
      bumps the monotonic ``flag_set_version`` counter so any downstream
      cache keyed on it invalidates, and records an audit entry.
    * ``GET /admin/sme-flags/{subscription_id}`` — returns the current
      override dict for the subscription (empty ``{}`` if none set).

    Admin auth is attached at mount-level by the caller (same convention as
    :func:`build_router` above) — the factory stays auth-agnostic so unit
    tests can exercise the handler without a live auth stack.
    """

    r = APIRouter(prefix="/admin/sme-flags", tags=["sme-admin"])
    known_flags = all_flag_names()

    @r.patch("/{subscription_id}")
    async def patch_flag(subscription_id: str, body: FlagFlipBody) -> dict:
        if not subscription_id or not subscription_id.strip():
            raise HTTPException(400, "subscription_id required")
        if body.flag not in known_flags:
            raise HTTPException(400, f"unknown flag: {body.flag!r}")

        store = deps.store
        setter = getattr(store, "set_subscription_override", None)
        if not callable(setter):
            raise HTTPException(
                500,
                "backing flag store does not support override writes",
            )
        prior_overrides = store.get_subscription_overrides(subscription_id)
        prior_value = bool(prior_overrides.get(body.flag, False))

        setter(subscription_id, body.flag, bool(body.enabled))
        new_version = bump_flag_set_version()

        entry = {
            "kind": "sme_flag_flip",
            "subscription_id": subscription_id,
            "flag": body.flag,
            "prior_value": prior_value,
            "new_value": bool(body.enabled),
            "reason": body.reason,
            "flag_set_version": new_version,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if deps.audit_writer is not None:
            try:
                deps.audit_writer(entry)
            except Exception:  # noqa: BLE001 — audit failure must not mask success
                logger.warning("audit_writer failed", exc_info=True)
        logger.info(
            "sme_flag_flip sub=%s flag=%s prior=%s new=%s version=%d",
            subscription_id,
            body.flag,
            prior_value,
            bool(body.enabled),
            new_version,
        )

        # Re-resolve with master gating so the response reflects what
        # ``SMEFeatureFlags.is_enabled`` will actually return going forward.
        try:
            effective = get_flag_resolver().is_enabled(
                subscription_id, body.flag
            )
        except (RuntimeError, KeyError):
            effective = bool(body.enabled)

        return {
            "subscription_id": subscription_id,
            "flag": body.flag,
            "prior_value": prior_value,
            "new_value": bool(body.enabled),
            "effective": effective,
            "flag_set_version": new_version,
        }

    @r.get("/{subscription_id}")
    async def get_overrides(subscription_id: str) -> dict:
        if not subscription_id or not subscription_id.strip():
            raise HTTPException(400, "subscription_id required")
        overrides = deps.store.get_subscription_overrides(subscription_id)
        # Normalize to a plain ``dict[str, bool]`` so callers can assume
        # a JSON-serialisable shape regardless of the underlying backing store.
        return {
            "subscription_id": subscription_id,
            "overrides": {k: bool(v) for k, v in (overrides or {}).items()},
        }

    return r


def build_sme_admin_router(
    adapter_deps: AdapterAdminDeps, flag_deps: FlagAdminDeps
) -> APIRouter:
    """Combined admin router — mounts the adapter + flag surfaces under one.

    Apps that want both the Phase 1 adapter endpoints and the Phase 3 flag
    endpoints call this factory once in :mod:`src.main` / ``app_lifespan``
    to get a single router to ``include_router`` into the FastAPI app.
    """

    root = APIRouter()
    root.include_router(build_router(adapter_deps))
    root.include_router(build_flag_router(flag_deps))
    return root
