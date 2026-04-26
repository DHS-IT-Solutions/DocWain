"""Production wiring for the Insights Portal.

Resolves the `NotImplementedError` hooks in:
  - src.tasks.researcher_v2.{resolve_default_store, resolve_default_adapter, resolve_default_llm}
  - src.tasks.researcher_v2_refresh.{resolve_default_index_collection, fetch_active_profile_documents}
  - src.api.insights_api.{list_insights_for_profile, get_insight_full}
  - src.api.actions_api.{list_actions_for_profile, execute_action}
  - src.api.visualizations_api.list_visualizations_for_profile
  - src.api.artifacts_api.list_artifacts_for_profile

Called once at app startup from the lifespan handler. Idempotent — safe
to call multiple times. Singletons cached on first call.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.intelligence.adapters.schema import Adapter
from src.intelligence.adapters.store import (
    AdapterStore, AdapterNotFound, AdapterBackend,
)
from src.intelligence.insights.store import (
    InsightStore, MongoIndexBackend, QdrantInsightBackend, Neo4jInsightBackend,
)
from src.intelligence.insights.schema import Insight, EvidenceSpan, KbRef
from src.intelligence.actions.runner import ActionRunner
from src.intelligence.actions.handlers import (
    artifact_handler, form_fill_handler, plan_handler, reminder_handler,
)
from src.intelligence.actions.audit import make_audit_writer
from src.intelligence.visualizations.generator import (
    generate_visualizations_for_profile,
)

logger = logging.getLogger(__name__)


# --- Adapter backend that maps Blob-style keys to repo files ----------------

class RepoAdapterBackend:
    """Reads bundled adapters directly from the repo.

    Used when ADAPTER_BLOB_LOADING_ENABLED=false (the v1 default). Maps
    canonical paths like `sme_adapters/global/generic.yaml` to the
    repo's `src/intelligence/adapters/generic.yaml`.
    """

    _MAPPINGS = {
        "sme_adapters/global/generic.yaml": "src/intelligence/adapters/generic.yaml",
        "sme_adapters/global/insurance.yaml": "src/intelligence/adapters/insurance.yaml",
        "sme_adapters/global/medical.yaml": "src/intelligence/adapters/medical.yaml",
        "sme_adapters/global/hr.yaml": "src/intelligence/adapters/hr.yaml",
        "sme_adapters/global/procurement.yaml": "src/intelligence/adapters/procurement.yaml",
        "sme_adapters/global/contract.yaml": "src/intelligence/adapters/contract.yaml",
        "sme_adapters/global/resume.yaml": "src/intelligence/adapters/resume.yaml",
    }

    def get_text(self, key: str) -> str:
        path = self._MAPPINGS.get(key)
        if path is None:
            raise AdapterNotFound(key)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError as exc:
            raise AdapterNotFound(key) from exc


# --- Singletons -------------------------------------------------------------

_ADAPTER_STORE: Optional[AdapterStore] = None
_INSIGHT_STORE: Optional[InsightStore] = None
_ACTION_RUNNER: Optional[ActionRunner] = None
_DB_NAME: Optional[str] = None
_INSIGHTS_INDEX_COLL_NAME = "insights_index"
_ARTIFACTS_COLL_NAME = "actions_artifacts"
_AUDIT_COLL_NAME = "actions_audit"


def _get_mongo_db():
    """Return the existing Mongo database used elsewhere in DocWain."""
    from src.api.dataHandler import db as _existing_db
    return _existing_db


def _get_qdrant_client():
    from src.api.dataHandler import get_qdrant_client
    return get_qdrant_client()


def _get_neo4j_driver():
    """Return a Neo4j driver from existing config; None if Neo4j unavailable."""
    try:
        from src.kg.neo4j_store import Neo4jStore
        store = Neo4jStore()
        # Neo4jStore exposes ._driver in current code; if not, leave None
        return getattr(store, "_driver", None) or getattr(store, "driver", None)
    except Exception as exc:
        logger.warning("Neo4j unavailable for InsightStore: %s", exc)
        return None


def _ensure_insights_qdrant_collection(client) -> None:
    """Create the insights collection if it doesn't exist."""
    try:
        from qdrant_client.http.models import Distance, VectorParams
        existing = {c.name for c in client.get_collections().collections}
        if "insights" not in existing:
            client.create_collection(
                collection_name="insights",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant 'insights' collection")
    except Exception as exc:
        logger.warning("ensure_insights_qdrant_collection failed: %s", exc)


def get_adapter_store() -> AdapterStore:
    global _ADAPTER_STORE
    if _ADAPTER_STORE is None:
        from src.api.config import insight_flag_enabled
        if insight_flag_enabled("ADAPTER_BLOB_LOADING_ENABLED"):
            from src.intelligence.adapters.store import resolve_default_backend
            backend = resolve_default_backend(blob_root=".")
        else:
            backend = RepoAdapterBackend()
        _ADAPTER_STORE = AdapterStore(backend=backend, cache_ttl_seconds=300)
    return _ADAPTER_STORE


def _ensure_insights_index_mongo_indexes(coll) -> None:
    """Create indexes on the Mongo insights_index for lookup paths.

    Idempotent — Mongo's create_index is no-op if index already exists.
    """
    try:
        coll.create_index("dedup_key", unique=True, background=True)
        coll.create_index("profile_id", background=True)
        coll.create_index([("profile_id", 1), ("severity", 1)], background=True)
        coll.create_index([("profile_id", 1), ("refreshed_at", -1)], background=True)
        coll.create_index("insight_id", background=True)
    except Exception as exc:
        logger.warning("ensure_insights_index_mongo_indexes failed: %s", exc)


def get_insight_store() -> InsightStore:
    global _INSIGHT_STORE
    if _INSIGHT_STORE is None:
        # Lazy collection resolver — survives transient Mongo connection drops
        # because the client is re-imported each call. See journal logs from
        # 2026-04-25: occasional CosmosDB disconnects fall back to localhost
        # if the module-level client is captured eagerly.
        def _coll_factory():
            return _get_mongo_db()[_INSIGHTS_INDEX_COLL_NAME]

        # Best-effort index creation on first init
        try:
            _ensure_insights_index_mongo_indexes(_coll_factory())
        except Exception as exc:
            logger.warning("index ensure deferred: %s", exc)

        mongo_index = MongoIndexBackend(collection=_coll_factory)

        qdrant_client = _get_qdrant_client()
        _ensure_insights_qdrant_collection(qdrant_client)
        qdrant_backend = QdrantInsightBackend(client=qdrant_client, collection_name="insights", embedder=None)

        neo4j_driver = _get_neo4j_driver()
        neo4j_backend = Neo4jInsightBackend(driver=neo4j_driver) if neo4j_driver else None

        _INSIGHT_STORE = InsightStore(
            mongo_index=mongo_index,
            qdrant=qdrant_backend,
            neo4j=neo4j_backend,
        )
    return _INSIGHT_STORE


def get_action_runner() -> ActionRunner:
    global _ACTION_RUNNER
    if _ACTION_RUNNER is None:
        db = _get_mongo_db()
        audit = make_audit_writer(collection=db[_AUDIT_COLL_NAME])
        _ACTION_RUNNER = ActionRunner(
            handlers={
                "artifact": artifact_handler,
                "form_fill": form_fill_handler,
                "plan": plan_handler,
                "reminder": reminder_handler,
            },
            audit_writer=audit,
        )
    return _ACTION_RUNNER


# --- LLM call adapter -------------------------------------------------------

def _llm_call(*, system: str = "", user: str = "", **kwargs) -> str:
    """Adapt the existing LLMGateway to the researcher's call signature."""
    from src.llm.gateway import get_llm_gateway
    gw = get_llm_gateway()
    return gw.generate(prompt=user, system=system, max_tokens=kwargs.get("max_tokens", 1024))


# --- Hook implementations ---------------------------------------------------

def _resolve_default_store_impl() -> InsightStore:
    return get_insight_store()


def _resolve_default_adapter_impl(*, domain: str, subscription_id: str) -> Adapter:
    return get_adapter_store().get(domain=domain, subscription_id=subscription_id)


def _resolve_default_llm_impl():
    return _llm_call


def _resolve_default_index_collection_impl():
    return _get_mongo_db()[_INSIGHTS_INDEX_COLL_NAME]


def _fetch_active_profile_documents_impl(*, profile_id: str) -> List[Dict[str, Any]]:
    """Fetch all documents for a profile from existing Mongo doc collection.

    Reads from the same documents collection used by extraction/embedding.
    Returns minimal {document_id, text} pairs for profile-level passes.
    """
    db = _get_mongo_db()
    docs_coll = db.get_collection("documents")
    out: List[Dict[str, Any]] = []
    try:
        cursor = docs_coll.find(
            {"profile_id": profile_id},
            {"document_id": 1, "extracted_text": 1, "_id": 0},
        ).limit(50)
        for doc in cursor:
            text = doc.get("extracted_text") or ""
            if isinstance(text, dict):
                text = str(text.get("text") or "")
            out.append({
                "document_id": str(doc.get("document_id") or ""),
                "text": str(text)[:8000],
            })
    except Exception as exc:
        logger.warning("fetch_active_profile_documents failed: %s", exc)
    return out


# --- API hook implementations -----------------------------------------------

def _list_insights_for_profile_impl(
    *,
    profile_id: str,
    insight_types=None,
    severities=None,
    domain=None,
    since=None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    store = get_insight_store()
    return store.list_for_profile(
        profile_id=profile_id,
        insight_types=insight_types,
        severities=severities,
        domain=domain,
        since=since,
        limit=limit,
        offset=offset,
    )


def _get_insight_full_impl(*, insight_id: str) -> Optional[Dict[str, Any]]:
    """Fetch full insight from Qdrant payload."""
    try:
        client = _get_qdrant_client()
        result = client.retrieve(
            collection_name="insights",
            ids=[insight_id],
            with_payload=True,
            with_vectors=False,
        )
        if not result:
            return None
        return dict(result[0].payload or {})
    except Exception as exc:
        logger.warning("get_insight_full failed: %s", exc)
        return None


def _list_actions_for_profile_impl(*, profile_id: str) -> List[Dict[str, Any]]:
    """Enumerate actions from the profile's resolved adapter."""
    db = _get_mongo_db()
    profile = db.get_collection("profiles").find_one({"profile_id": profile_id}) or {}
    domain = str(profile.get("domain") or "generic")
    subscription_id = str(profile.get("subscription_id") or "")
    try:
        adapter = get_adapter_store().get(domain=domain, subscription_id=subscription_id)
    except Exception as exc:
        logger.warning("list_actions adapter load failed: %s", exc)
        return []
    return [
        {
            "action_id": a.action_id,
            "title": a.title,
            "action_type": a.action_type,
            "requires_confirmation": a.requires_confirmation,
            "preview": "",
        }
        for a in adapter.actions
    ]


def _execute_action_impl(
    *, profile_id: str, action_id: str, inputs: Dict[str, Any], confirmed: bool
) -> Dict[str, Any]:
    db = _get_mongo_db()
    profile = db.get_collection("profiles").find_one({"profile_id": profile_id}) or {}
    domain = str(profile.get("domain") or "generic")
    subscription_id = str(profile.get("subscription_id") or "")
    adapter = get_adapter_store().get(domain=domain, subscription_id=subscription_id)
    action = next((a for a in adapter.actions if a.action_id == action_id), None)
    if action is None:
        return {"status": "unknown_action", "action_id": action_id}
    runner = get_action_runner()
    result = runner.execute(
        action=action, profile_id=profile_id, inputs=inputs, confirmed=confirmed
    )
    return {
        "status": result.status,
        "preview": result.preview,
        "output": result.output,
    }


def _list_visualizations_for_profile_impl(*, profile_id: str) -> List[Dict[str, Any]]:
    """Generate viz specs from current insights for the profile."""
    store = get_insight_store()
    rows = store.list_for_profile(profile_id=profile_id, limit=200)
    # Reconstruct minimal Insight objects (timeline only needs refreshed_at + headline)
    insights: List[Insight] = []
    for r in rows:
        try:
            insight = Insight(
                insight_id=r.get("insight_id", ""),
                profile_id=profile_id,
                subscription_id=r.get("subscription_id", ""),
                document_ids=list(r.get("document_ids") or []),
                domain=r.get("domain", "generic"),
                insight_type=r.get("insight_type", "anomaly"),
                headline=r.get("insight_id", ""),
                body="(viz reconstruction)",
                evidence_doc_spans=[EvidenceSpan(
                    document_id="x", page=0, char_start=0, char_end=1, quote="x"
                )],
                confidence=0.0,
                severity=r.get("severity", "info"),
                adapter_version=r.get("adapter_version", "generic@1.0"),
                refreshed_at=r.get("refreshed_at", ""),
            )
            insights.append(insight)
        except Exception:
            continue
    return generate_visualizations_for_profile(insights)


def _list_artifacts_for_profile_impl(*, profile_id: str) -> List[Dict[str, Any]]:
    db = _get_mongo_db()
    coll = db.get_collection(_ARTIFACTS_COLL_NAME)
    try:
        rows = list(coll.find(
            {"profile_id": profile_id},
            {"_id": 0},
        ).limit(100))
    except Exception as exc:
        logger.warning("list_artifacts_for_profile failed: %s", exc)
        rows = []
    return rows


# --- Wire everything --------------------------------------------------------

def wire_insights_portal() -> None:
    """Patch all hook functions in the v2 modules. Idempotent."""
    from src.tasks import researcher_v2 as rv2
    from src.tasks import researcher_v2_refresh as rv2r
    from src.api import insights_api, actions_api, visualizations_api, artifacts_api

    rv2.resolve_default_store = _resolve_default_store_impl
    rv2.resolve_default_adapter = _resolve_default_adapter_impl
    rv2.resolve_default_llm = _resolve_default_llm_impl

    rv2r.resolve_default_index_collection = _resolve_default_index_collection_impl
    rv2r.fetch_active_profile_documents = _fetch_active_profile_documents_impl

    insights_api.list_insights_for_profile = _list_insights_for_profile_impl
    insights_api.get_insight_full = _get_insight_full_impl

    actions_api.list_actions_for_profile = _list_actions_for_profile_impl
    actions_api.execute_action = _execute_action_impl

    visualizations_api.list_visualizations_for_profile = _list_visualizations_for_profile_impl
    artifacts_api.list_artifacts_for_profile = _list_artifacts_for_profile_impl

    logger.info("Insights Portal hooks wired (researcher_v2 + refresh + 4 surface APIs)")
