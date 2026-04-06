"""
Request execution router.
Routes /api/ask requests through the Fast Path (SIMPLE queries) or Core Agent.
"""
import logging
import os
from typing import Any, Generator, Optional

from src.execution.common import ExecutionResult
from src.mode.execution_mode import ExecutionMode

logger = logging.getLogger(__name__)

FAST_PATH_ENABLED = os.getenv("FAST_PATH_ENABLED", "true").lower() == "true"

# Lazy singleton
_core_agent = None


def _get_core_agent():
    """Lazy-initialize the Core Agent singleton."""
    global _core_agent
    if _core_agent is None:
        from src.agent.core_agent import CoreAgent
        from src.llm.gateway import get_llm_gateway
        from src.api.config import Config
        from src.embedding.model_loader import get_embedding_model
        from qdrant_client import QdrantClient
        from pymongo import MongoClient

        llm = get_llm_gateway()
        qdrant = QdrantClient(
            url=Config.Qdrant.URL,
            api_key=Config.Qdrant.API,
        )
        embedder, _ = get_embedding_model()
        mongo_client = MongoClient(Config.MongoDB.URI)
        mongodb = mongo_client[Config.MongoDB.DB][Config.MongoDB.DOCUMENTS]

        try:
            from src.intelligence.kg_query import KGQueryService
            kg_service = KGQueryService()
        except Exception:
            kg_service = None

        # Get cross-encoder from app state (loaded at startup on CPU)
        ce = None
        try:
            from src.api.rag_state import get_app_state
            app_state = get_app_state()
            if app_state:
                ce = getattr(app_state, "reranker", None)
        except Exception:
            pass

        _core_agent = CoreAgent(
            llm_gateway=llm,
            qdrant_client=qdrant,
            embedder=embedder,
            mongodb=mongodb,
            kg_query_service=kg_service,
            cross_encoder=ce,
        )
    return _core_agent


def _load_conversation_history(request: Any, ctx: Any) -> Optional[list]:
    """Load conversation history from session context."""
    if hasattr(ctx, "session_id") and ctx.session_id:
        try:
            from src.api.dw_chat import get_current_session_context
            return get_current_session_context(
                user_id=getattr(request, "user_id", ""),
                session_id=ctx.session_id,
                max_messages=5,
            )
        except Exception as e:
            logger.debug("[ROUTER] Could not load conversation history: %s", e)
    return None


def execute_request(
    request: Any,
    session_state: Any,
    ctx: Any,
    *,
    stream: bool = False,
    debug: bool = False,
) -> ExecutionResult:
    """Execute an /api/ask request through Fast Path or Core Agent."""
    query = getattr(request, "query", ctx.query)
    subscription_id = getattr(request, "subscription_id", ctx.subscription_id)
    profile_id = getattr(request, "profile_id", ctx.profile_id)

    # --- Fast path for SIMPLE queries ---
    if FAST_PATH_ENABLED:
        try:
            from src.execution.query_classifier import classify_query
            classification = classify_query(query)
            if classification.query_type == "SIMPLE":
                logger.info(
                    "[ROUTER] Fast path: type=%s confidence=%.2f signals=%s",
                    classification.query_type, classification.confidence, classification.signals,
                )
                from src.execution.fast_path import execute_fast_path
                from src.api.rag_state import get_app_state
                app_state = get_app_state()
                if app_state and app_state.llm_gateway:
                    answer = execute_fast_path(query, profile_id, subscription_id, app_state)
                    debug_info = answer.get("metadata", {}) if debug else {}
                    return ExecutionResult(
                        answer=answer,
                        mode=ExecutionMode.AGENT,
                        debug=debug_info,
                    )
        except Exception as e:
            logger.warning("[ROUTER] Fast path failed, falling back to CoreAgent: %s", e)

    # --- Full path via CoreAgent ---
    agent = _get_core_agent()
    conversation_history = _load_conversation_history(request, ctx)

    answer = agent.handle(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        user_id=getattr(request, "user_id", getattr(ctx, "user_id", "")),
        session_id=getattr(ctx, "session_id", None),
        conversation_history=conversation_history,
        agent_name=getattr(request, "agent_name", None),
        document_id=getattr(request, "document_id", None),
        debug=debug,
    )

    debug_info = answer.get("metadata", {}) if debug else {}

    return ExecutionResult(
        answer=answer,
        mode=ExecutionMode.AGENT,
        debug=debug_info,
    )


def execute_request_stream(
    request: Any,
    session_state: Any,
    ctx: Any,
) -> Generator[str, None, None]:
    """Stream response tokens from Fast Path or Core Agent.

    UNDERSTAND + RETRIEVE run synchronously, then REASON tokens stream out
    as they arrive from the LLM.
    """
    query = getattr(request, "query", ctx.query)
    subscription_id = getattr(request, "subscription_id", ctx.subscription_id)
    profile_id = getattr(request, "profile_id", ctx.profile_id)

    # --- Fast path for SIMPLE queries ---
    if FAST_PATH_ENABLED:
        try:
            from src.execution.query_classifier import classify_query
            classification = classify_query(query)
            if classification.query_type == "SIMPLE":
                logger.info(
                    "[ROUTER_STREAM] Fast path: type=%s confidence=%.2f signals=%s",
                    classification.query_type, classification.confidence, classification.signals,
                )
                from src.execution.fast_path import execute_fast_path_stream
                from src.api.rag_state import get_app_state
                app_state = get_app_state()
                if app_state and app_state.llm_gateway:
                    yield from execute_fast_path_stream(query, profile_id, subscription_id, app_state)
                    return
        except Exception as e:
            logger.warning("[ROUTER_STREAM] Fast path failed, falling back to CoreAgent: %s", e)

    # --- Full path via CoreAgent ---
    agent = _get_core_agent()
    conversation_history = _load_conversation_history(request, ctx)

    yield from agent.handle_stream(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        user_id=getattr(request, "user_id", getattr(ctx, "user_id", "")),
        session_id=getattr(ctx, "session_id", None),
        conversation_history=conversation_history,
        agent_name=getattr(request, "agent_name", None),
        document_id=getattr(request, "document_id", None),
    )
