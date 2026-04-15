from contextlib import asynccontextmanager

from fastapi import FastAPI

from standalone.dependencies import cleanup, get_vllm_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await cleanup()


app = FastAPI(
    title="DocWain Standalone",
    description="Document extraction and intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
from standalone.endpoints.extract import router as extract_router
from standalone.endpoints.intelligence import router as intelligence_router
from standalone.endpoints.keys import router as keys_router

app.include_router(extract_router)
app.include_router(intelligence_router)
app.include_router(keys_router)


@app.get("/health")
async def health():
    vllm = get_vllm_client()
    vllm_ok = await vllm.health_check()
    return {"status": "ok" if vllm_ok else "degraded", "vllm": vllm_ok}
