"""DocWain Teams App — Standalone FastAPI Service."""

from __future__ import annotations

import json
import logging
import os
import sys

# Ensure project root is on sys.path for src imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from teams_app.config import TeamsAppConfig
from teams_app.lifespan import teams_lifespan

logger = logging.getLogger(__name__)

config = TeamsAppConfig()

app = FastAPI(
    title="DocWain Teams Service",
    version="1.0.0",
    lifespan=teams_lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    state = app.state.teams
    checks = {}

    try:
        state.qdrant_client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as exc:
        checks["qdrant"] = f"error: {exc}"

    try:
        state.mongo_db.command("ping")
        checks["mongodb"] = "ok"
    except Exception as exc:
        checks["mongodb"] = f"error: {exc}"

    if state.redis_client:
        try:
            state.redis_client.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {exc}"
    else:
        checks["redis"] = "unavailable (in-memory fallback)"

    healthy = all(v == "ok" for k, v in checks.items() if k != "redis")
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "healthy" if healthy else "degraded", "checks": checks},
    )


@app.post("/teams/messages")
async def handle_teams_messages(request: Request):
    """Handle incoming Teams bot messages."""
    state = app.state.teams

    try:
        body = await request.body()
        if not body:
            return JSONResponse(status_code=400, content={"error": "Empty body"})
        activity_payload = json.loads(body)
    except Exception as exc:
        logger.error("Failed to parse Teams activity: %s", exc)
        return JSONResponse(status_code=400, content={"error": str(exc)})

    auth_header = request.headers.get("Authorization", "")

    try:
        from botbuilder.schema import Activity

        activity_obj = Activity().deserialize(activity_payload)

        async def _run_bot(turn_context):
            await state.bot.on_turn(turn_context)

        response = await state.bot_adapter.process_activity(
            activity_obj, auth_header, _run_bot,
        )

        if response:
            return JSONResponse(status_code=response.status, content=response.body)
        return JSONResponse(status_code=200, content={})

    except PermissionError as exc:
        logger.warning("Auth failed: %s", exc)
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    except Exception as exc:
        logger.exception("Teams message handling failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/teams/feedback")
async def handle_feedback(request: Request):
    """Handle thumbs up/down feedback from response cards."""
    state = app.state.teams
    try:
        data = await request.json()
        signal = data.get("signal", "implicit")
        query = data.get("query", "")
        response = data.get("response", "")
        tenant_id = data.get("tenant_id", "")

        state.signal_capture.record(
            query=query, response=response, sources=[], grounded=True,
            context_found=True, signal=signal, tenant_id=tenant_id,
        )
        return JSONResponse(status_code=200, content={"status": "recorded"})
    except Exception as exc:
        logger.error("Feedback recording failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    uvicorn.run(app, host=config.host, port=config.port)
