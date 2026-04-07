"""Lightweight lifespan for the Teams service — loads only what's needed."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI

logger = logging.getLogger(__name__)


@dataclass
class TeamsAppState:
    """Minimal state for the Teams service."""
    embedding_model: Any = None
    qdrant_client: Any = None
    mongo_db: Any = None
    redis_client: Any = None
    bot: Any = None
    bot_adapter: Any = None
    tenant_manager: Any = None
    orchestrator: Any = None
    query_handler: Any = None
    signal_capture: Any = None


@asynccontextmanager
async def teams_lifespan(app: FastAPI):
    """Initialize Teams service dependencies — lightweight, ~10-15s startup."""
    from teams_app.config import TeamsAppConfig
    import asyncio

    config = TeamsAppConfig()
    state = TeamsAppState()

    logger.info("Starting DocWain Teams service on port %d", config.port)

    # 1. Embedding model (~5s)
    logger.info("Loading embedding model...")
    try:
        from src.api.dw_newron import get_model
        state.embedding_model = await asyncio.to_thread(get_model)
        logger.info("Embedding model loaded")
    except Exception as exc:
        logger.warning("Embedding model not available (document ingestion will fail): %s", exc)
        state.embedding_model = None

    # 2. Qdrant
    logger.info("Connecting to Qdrant...")
    from src.api.dw_newron import get_qdrant_client
    state.qdrant_client = get_qdrant_client()
    logger.info("Qdrant connected")

    # 3. MongoDB — dataHandler exposes a module-level db object
    logger.info("Connecting to MongoDB...")
    from src.api.dataHandler import db as mongo_db
    state.mongo_db = mongo_db
    logger.info("MongoDB connected")

    # 4. Redis
    logger.info("Connecting to Redis...")
    try:
        from src.api.dw_newron import get_redis_client
        state.redis_client = get_redis_client()
        logger.info("Redis connected")
    except Exception as exc:
        logger.warning("Redis not available, using in-memory fallback: %s", exc)
        state.redis_client = None

    # 5. Teams components
    from teams_app.storage.tenant import TenantManager
    from teams_app.signals.capture import SignalCapture
    from teams_app.proxy.query_handler import TeamsQueryHandler
    from teams_app.pipeline.orchestrator import TeamsAutoOrchestrator
    from src.teams.state import TeamsStateStore
    from src.teams.teams_storage import TeamsDocumentStorage

    state.tenant_manager = TenantManager(db=state.mongo_db, qdrant_client=state.qdrant_client)
    state.signal_capture = SignalCapture(signals_dir=config.signals_dir)
    state.query_handler = TeamsQueryHandler(qdrant_client=state.qdrant_client)

    storage = TeamsDocumentStorage()
    teams_state_store = TeamsStateStore()

    state.orchestrator = TeamsAutoOrchestrator(
        storage=storage,
        state_store=teams_state_store,
        tenant_manager=state.tenant_manager,
        signal_capture=state.signal_capture,
        config=config,
    )

    # 6. Bot Framework adapter + bot
    from src.teams.bot_app import bot_adapter
    from teams_app.bot.handler import StandaloneTeamsBot

    state.bot_adapter = bot_adapter
    state.bot = StandaloneTeamsBot(
        orchestrator=state.orchestrator,
        query_handler=state.query_handler,
        tenant_manager=state.tenant_manager,
        signal_capture=state.signal_capture,
        config=config,
    )

    app.state.teams = state
    logger.info("DocWain Teams service ready")

    yield

    logger.info("Shutting down DocWain Teams service")
