# DocWain vLLM Production Wiring — Design Spec

**Date:** 2026-04-04
**Status:** Approved
**Author:** Muthu / Claude

## Summary

Wire vLLM as the primary serving backend for DocWain, replacing Ollama Cloud as the default inference path. Dual-instance architecture: 14B fine-tuned DocWain V2 (fast) + 27B Qwen3.5 (smart). Ollama Cloud (397b) as fallback during training windows. GPU scheduler manages coexistence of serving and periodic training on a single A100 80GB.

## Goals

1. Eliminate query latency — vLLM local serving with prefix caching and speculative decoding
2. Dual-model intelligence — fast 14B for simple queries, smart 27B for complex reasoning
3. Always-on serving — vLLM is primary, training scavenges GPU when idle
4. Zero-downtime model updates — hot-swap fine-tuned weights without user impact
5. Graceful degradation — Ollama Cloud fallback during training, Ollama local as emergency

## Non-Goals (Phase 2: Agentic Layer)

- Tool-calling agent loop and sub-agent spawning
- Dynamic tool registry
- Parallel sub-agent execution
- These will be built on top of the vLLM serving layer after it's stable

## Architecture

```
                     FastAPI App
  ┌──────────────────────────────────────────┐
  │  Query Pipeline  │  Doc Processing       │
  │  (interaction)   │  (background)         │
  │       │          │       │               │
  │       └────┬─────┘───────┘               │
  │            │                              │
  │    ┌───────▼────────┐                    │
  │    │  VLLMManager   │  (client-only)     │
  │    │  - health check│                    │
  │    │  - query route │                    │
  │    │  - fallback    │                    │
  │    └──┬──────┬──────┘                    │
  └───────┼──────┼───────────────────────────┘
          │      │
  ┌───────▼──┐ ┌─▼───────────┐
  │ vLLM Fast│ │ vLLM Smart  │  systemd services
  │ 14B:8100 │ │ 27B:8200    │
  └──────────┘ └─────────────┘
          │      │
          └──┬───┘ fallback
     ┌───────▼────────┐
     │ Ollama Cloud   │  (397b, always available)
     │ Ollama Local   │  (14b, emergency)
     └────────────────┘

  ┌──────────────────────┐
  │   GPU Scheduler      │  separate systemd service
  │   - idle detection   │
  │   - stop vLLM        │
  │   - run training     │
  │   - hot-swap model   │
  │   - restart vLLM     │
  └──────────────────────┘
```

## Component Details

### 1. vLLM Systemd Services

Two independent systemd unit files:

**docwain-vllm-fast.service** (14B DocWain V2):
- Model: symlink at `models/docwain-v2-active/` pointing to latest fine-tuned weights
- Port: 8100
- Quantization: FP8
- Features: prefix caching ON, EAGLE3 speculative decoding
- GPU memory: `--gpu-memory-utilization 0.25` (~20GB)
- Served model name: `docwain-fast`
- Auto-restart on crash

**docwain-vllm-smart.service** (27B Qwen3.5):
- Model: `Qwen/Qwen3.5-27B` (HuggingFace, later fine-tuned)
- Port: 8200
- Quantization: FP8
- Features: chunked prefill ON, 32K context
- GPU memory: `--gpu-memory-utilization 0.65` (~52GB)
- Served model name: `docwain-smart`
- Auto-restart on crash

**Startup order:** fast first (lighter), then smart. Health check before marking ready.

### 2. VLLMManager Refactor (Client-Only)

**Remove:** All subprocess/Popen management (`start_instance`, `stop_instance`, process handles, threading locks for subprocesses).

**Keep and enhance:**

```python
class VLLMManager:
    def __init__(self, fast_url, smart_url, ollama_cloud_url, ollama_local_url):
        ...

    def health_check(self, instance: str) -> bool:
        """GET /health on instance port."""

    def get_gpu_mode(self) -> str:
        """Read /tmp/docwain-gpu-mode.json — 'serving' or 'training'."""

    def get_active_backends(self) -> dict:
        """Return which backends are healthy."""

    async def query(self, prompt, *, route="auto", system=None,
                    response_format=None, temperature=0.3) -> str:
        """Main entry point. Routes based on intent or explicit route.

        Fallback chain:
          serving mode: vLLM fast/smart → Ollama Cloud → Ollama local
          training mode: Ollama Cloud → Ollama local
        """

    async def query_fast(self, prompt, **kwargs) -> str:
        """Force 14B path."""

    async def query_smart(self, prompt, **kwargs) -> str:
        """Force 27B path."""
```

**Fallback chain logic:**
```
if gpu_mode == "training":
    → Ollama Cloud (397b)
    → Ollama local (14b) if Cloud down
elif health_check("fast") and route == "fast":
    → vLLM 14B
    → Ollama Cloud if vLLM down
elif health_check("smart") and route == "smart":
    → vLLM 27B
    → Ollama Cloud if vLLM down
else:
    → Ollama Cloud
    → Ollama local
```

### 3. App Lifecycle Wiring

**src/api/app_lifespan.py** modifications:

```python
async def initialize_app_state(app):
    # Existing: embedding model, qdrant, redis, etc.
    ...

    # New: VLLMManager (client-only, no subprocess)
    vllm_manager = VLLMManager(
        fast_url=settings.VLLM_FAST_URL,       # http://localhost:8100
        smart_url=settings.VLLM_SMART_URL,      # http://localhost:8200
        ollama_cloud_url=settings.OLLAMA_CLOUD,  # existing
        ollama_local_url=settings.OLLAMA_LOCAL,  # existing
    )
    # Log backend availability (non-blocking)
    backends = vllm_manager.get_active_backends()
    logger.info("LLM backends: %s", backends)

    app.state.vllm_manager = vllm_manager
```

**Query pipeline receives vllm_manager via clients dict:**
```python
clients = {
    "vllm_manager": app.state.vllm_manager,
    "qdrant": app.state.qdrant_client,
    ...
}
result = await run_query_pipeline(query, profile_id, subscription_id, clients)
```

No changes needed to IntentRouter, FastPathHandler, QueryPlanner, ResponseGenerator — they already accept vllm_manager.

### 4. GPU Scheduler Daemon

**docwain-gpu-scheduler.service** — separate systemd service.

```
Every 5 minutes:
  1. GET /metrics from vLLM fast + smart instances
     → extract request rate (requests in last 5 min)

  2. Check training queue:
     → Read finetune_artifacts/v2_upgrade/training_queue.json
     → Contains: {pending: bool, priority: "low"|"normal"|"high"}

  3. Decision matrix:
     ┌─────────────┬──────────────┬──────────────────────────┐
     │ Request Rate │ Training Q   │ Action                   │
     ├─────────────┼──────────────┼──────────────────────────┤
     │ > 0         │ any          │ Keep serving              │
     │ 0 for 30min │ pending      │ Enter training mode       │
     │ 0 for 30min │ empty        │ Keep serving (standby)    │
     │ N/A (train) │ running      │ Continue training         │
     │ N/A (train) │ complete     │ Hot-swap + resume serving │
     └─────────────┴──────────────┴──────────────────────────┘

  Enter training mode:
    a. Write /tmp/docwain-gpu-mode.json = {"mode": "training", "since": "..."}
    b. Wait 10s for in-flight requests to drain
    c. systemctl stop docwain-vllm-smart
    d. systemctl stop docwain-vllm-fast
    e. Run: python -m src.finetune.v2.autonomous_trainer --resume
    f. On completion:
       - Update model symlink
       - systemctl start docwain-vllm-fast
       - systemctl start docwain-vllm-smart
       - Write gpu-mode.json = {"mode": "serving"}
```

### 5. Model Hot-Swap

After fine-tuning completes new weights:

```
models/
  docwain-v2-active/  →  symlink to latest
  docwain-v2-runs/
    2026-04-04/       ← latest fine-tuned GGUF
    2026-03-15/       ← previous version
```

Swap procedure:
1. GPU scheduler stops vLLM fast instance
2. `ln -sfn models/docwain-v2-runs/2026-04-04 models/docwain-v2-active`
3. Start vLLM fast instance (loads new weights)
4. Health check passes → live

### 6. Configuration

**New env vars / settings:**

```
VLLM_FAST_URL=http://localhost:8100
VLLM_SMART_URL=http://localhost:8200
VLLM_FAST_MODEL=docwain-fast
VLLM_SMART_MODEL=docwain-smart
VLLM_FAST_GPU_UTIL=0.25
VLLM_SMART_GPU_UTIL=0.65
GPU_SCHEDULER_IDLE_THRESHOLD_MINUTES=30
GPU_MODE_FILE=/tmp/docwain-gpu-mode.json
MODEL_ACTIVE_SYMLINK=models/docwain-v2-active
```

### 7. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `systemd/docwain-vllm-fast.service` | **Create** | 14B vLLM systemd unit |
| `systemd/docwain-vllm-smart.service` | **Create** | 27B vLLM systemd unit |
| `systemd/docwain-gpu-scheduler.service` | **Create** | GPU scheduler systemd unit |
| `scripts/gpu_scheduler.py` | **Create** | GPU scheduler daemon logic |
| `src/serving/vllm_manager.py` | **Refactor** | Strip subprocess mgmt, keep client + fallback |
| `src/serving/config.py` | **Modify** | Add new config vars, remove subprocess config |
| `src/api/app_lifespan.py` | **Modify** | Initialize VLLMManager, store in AppState |
| `src/api/config.py` | **Modify** | Add VLLM settings |
| `src/query/pipeline.py` | **Modify** | Ensure vllm_manager flows through clients dict |
| `src/llm/gateway.py` | **Modify** | vLLM-first fallback logic (or deprecate in favor of VLLMManager) |

### 8. Testing Strategy

1. **Unit tests:** VLLMManager fallback chain with mocked health checks
2. **Integration test:** Start vLLM fast instance, run a query through the full pipeline
3. **Fallback test:** Stop vLLM, verify Ollama Cloud picks up seamlessly
4. **GPU scheduler test:** Simulate idle period, verify training starts and serving resumes
5. **Hot-swap test:** Replace model weights, verify new model loads correctly
6. **Load test:** Concurrent queries to verify continuous batching throughput

### 9. Rollback Plan

If vLLM integration causes issues:
- Set `VLLM_ENABLED=false` in env → VLLMManager skips vLLM, goes straight to Ollama Cloud
- No code changes needed, just env var flip
- Systemd services can stay running (unused) or be stopped
