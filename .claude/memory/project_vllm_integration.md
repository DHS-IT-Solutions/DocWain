---
name: vLLM Production Integration
description: Status of vLLM dual-instance wiring (14B fast + 27B smart), systemd services, GPU scheduler, and what remains to be done
type: project
---

## vLLM Production Wiring — Status as of 2026-04-04

**Phase: Code complete. Awaiting first vLLM startup after training completes.**

### What's Done (all committed to main)
1. **Config** — `src/api/config.py` has `Config.VLLM.FAST_URL/SMART_URL/FAST_MODEL/SMART_MODEL/GPU_MODE_FILE`
2. **VLLMManager rewrite** — `src/serving/vllm_manager.py` is now client-only (no subprocess). Tiered fallback: vLLM → Ollama Cloud (397b) → Ollama local (14b). GPU mode awareness via `/tmp/docwain-gpu-mode.json`
3. **AppState wiring** — `src/api/rag_state.py` has `vllm_manager` field. `src/api/app_lifespan.py` initializes it when `Config.VLLM.ENABLED=True`
4. **Systemd units** installed and enabled at `/etc/systemd/system/`:
   - `docwain-vllm-fast.service` — 14B on port 8100 (gpu-mem 0.25)
   - `docwain-vllm-smart.service` — 27B on port 8200 (gpu-mem 0.65)
   - `docwain-gpu-scheduler.service` — manages GPU sharing between serving and training
5. **GPU scheduler** — `scripts/gpu_scheduler.py` detects idle vLLM, stops it, runs training, hot-swaps model, restarts vLLM
6. **Model symlink** — `models/docwain-v2-active` → latest fine-tuned checkpoint
7. **Tests** — 34 passed, 7 skipped (integration tests auto-skip when vLLM not running)

### What Remains
- **Start vLLM services** — `sudo systemctl start docwain-vllm-fast docwain-vllm-smart` (after training frees GPU)
- **Start GPU scheduler** — `sudo systemctl start docwain-gpu-scheduler`
- **Run integration tests** — `pytest tests/test_vllm_integration.py -v` (will run once vLLM is up)
- **Verify fallback chain** — stop smart, query smart, confirm Ollama Cloud picks up
- **Verify training mode** — set gpu-mode to training, confirm queries route to Ollama Cloud

### Key Files
- Spec: `docs/superpowers/specs/2026-04-04-vllm-production-wiring-design.md`
- Plan: `docs/superpowers/plans/2026-04-04-vllm-production-wiring.md`
- VLLMManager: `src/serving/vllm_manager.py`
- GPU Scheduler: `scripts/gpu_scheduler.py`
- Systemd units: `systemd/docwain-vllm-*.service`
- Tests: `tests/test_vllm_manager.py`, `tests/test_gpu_scheduler.py`, `tests/test_vllm_integration.py`

### Next Phase: Agentic Layer (Phase 2)
User approved building agentic capabilities AFTER vLLM is stable:
- Tool registry (`src/agents/tool_registry.py`)
- Orchestrator agent loop (`src/agents/orchestrator.py`)
- Parallel sub-agent execution (`src/agents/executor.py`)
- Tools: search_documents, extract_entities, compare_documents, generate_visualization, query_knowledge_graph, etc.

**Why:** Goal is a high-intelligence GPT-like model. vLLM = fast inference. Agentic = intelligent multi-step reasoning.

**How to apply:** When resuming, check if vLLM services are running (`systemctl status docwain-vllm-fast`). If not and training is not running, start them. Then proceed to Phase 2.
