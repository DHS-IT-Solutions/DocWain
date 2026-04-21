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

Per Phase 1 scope: no production path consumes the loader yet — the router is
registered in app lifespan but dormant until Phase 2 wires synthesis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import yaml
from fastapi import APIRouter, Body, HTTPException, Request

from src.intelligence.sme.adapter_loader import AdapterLoader
from src.intelligence.sme.adapter_schema import Adapter


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
