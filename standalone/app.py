from fastapi import FastAPI

app = FastAPI(
    title="DocWain Standalone",
    description="Document extraction and intelligence API",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}
