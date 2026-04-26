# DocWain — 16 GB GPU Swap Runbook

**When:** swap deadline within 4 days of 2026-04-26.
**What changes:** GPU goes from A100 80 GB to 16 GB target; vLLM serving config switches from bfloat16 14B → AWQ-W4A16 14B with reduced `max-model-len`.
**Goal:** zero downtime to running services *before* the swap; ≤5 min downtime *during* the swap.

---

## Pre-swap (hours T-2 to T-0): nothing affects live services

These steps run while the A100 is still attached and `docwain-vllm-fast` keeps serving on it.

### 1. Verify AWQ artifact integrity
```bash
ls -lh /home/ubuntu/PycharmProjects/DocWain/models/DocWain-14B-v2-AWQ/
cat /home/ubuntu/PycharmProjects/DocWain/models/DocWain-14B-v2-AWQ/AWQ_QUANT_INFO.json
# Expect: ~7-8 GB safetensors + tokenizer files + AWQ_QUANT_INFO.json
```

### 2. Smoke-test the AWQ artifact on a parallel vLLM port (8101)
While the A100 is still here, confirm the artifact loads cleanly with `--quantization compressed-tensors`. Drop vLLM-fast's memory budget temporarily so a second instance can co-exist on the same GPU:
```bash
# Briefly reduce vLLM-fast to 0.45 (40 GB) to make room — restart blip ~30s
sudo sed -i 's/--gpu-memory-utilization 0.90/--gpu-memory-utilization 0.45/' /etc/systemd/system/docwain-vllm-fast.service
sudo systemctl daemon-reload && sudo systemctl restart docwain-vllm-fast

# Start AWQ smoke instance on port 8101 with simulated 16 GB constraint
nohup /home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/PycharmProjects/DocWain/models/DocWain-14B-v2-AWQ \
    --served-model-name docwain-fast-awq \
    --quantization compressed-tensors \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.20 \
    --tensor-parallel-size 1 \
    --port 8101 --host 0.0.0.0 \
    > /tmp/vllm_awq_smoke.log 2>&1 &

# Wait until ready, then probe
until curl -sf http://localhost:8101/v1/models > /dev/null; do sleep 5; done
curl -s -X POST http://localhost:8101/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"docwain-fast-awq","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":10}'
```
**Expected:** HTTP 200 with a coherent reply containing "OK".

### 3. Run insights-portal end-to-end against the AWQ instance
Bind a one-off `LLMGateway` pointing at port 8101 and run the smoke profile through the researcher v2 path:
```bash
DOCWAIN_LLM_FAST_BASE_URL=http://localhost:8101/v1 \
DOCWAIN_LLM_FAST_MODEL=docwain-fast-awq \
python -c "
from src.api.insights_wiring import wire_insights_portal
wire_insights_portal()
from src.tasks.researcher_v2 import run_researcher_v2_for_doc
text = open('/tmp/live_test_doc.txt').read()
import os
os.environ.update({k:'true' for k in ['INSIGHTS_TYPE_ANOMALY_ENABLED','INSIGHTS_TYPE_GAP_ENABLED','INSIGHTS_TYPE_RECOMMENDATION_ENABLED','INSIGHTS_CITATION_ENFORCEMENT_ENABLED','ADAPTER_GENERIC_FALLBACK_ENABLED']})
print(run_researcher_v2_for_doc(document_id='AWQ-SMOKE', profile_id='awq-smoke-prof', subscription_id='awq-sub', document_text=text, domain_hint='generic'))
"
```
**Expected:** `{'status': 'ok', 'written': N}` where N ≥ 2.

### 4. Quality side-by-side eval
Run `tests/sme_evalset_v1` against both the bfloat16 (port 8100) and AWQ (port 8101) instances and compare:
```bash
DOCWAIN_LLM_BASE_URL=http://localhost:8100/v1 python -m tests.sme_evalset_v1.run_eval --output /tmp/eval_bfloat16.json
DOCWAIN_LLM_BASE_URL=http://localhost:8101/v1 python -m tests.sme_evalset_v1.run_eval --output /tmp/eval_awq.json
python -c "
import json
a=json.load(open('/tmp/eval_bfloat16.json')); b=json.load(open('/tmp/eval_awq.json'))
print(f\"bfloat16 score: {a.get('score',0):.3f}\"); print(f\"AWQ score:      {b.get('score',0):.3f}\")
print(f\"Δ: {b.get('score',0) - a.get('score',0):+.3f}\")
"
```
**Pass criterion:** AWQ score ≥ bfloat16 − 0.03 (3% relative drop is the AWQ tolerance band).

### 5. Tear down the smoke instance + restore vLLM-fast memory
```bash
PID=$(pgrep -f "vllm.*--port 8101"); [ -n "$PID" ] && kill $PID
sudo sed -i 's/--gpu-memory-utilization 0.45/--gpu-memory-utilization 0.90/' /etc/systemd/system/docwain-vllm-fast.service
sudo systemctl daemon-reload && sudo systemctl restart docwain-vllm-fast
```

---

## Swap (T-0 to T+5 min): coordinated GPU change

### 6. Stop services that hold the GPU
```bash
sudo systemctl stop docwain-vllm-fast docwain-app docwain-celery-worker docwain-celery-beat
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # expect ~0 MiB used
```

### 7. Physical / virtual GPU swap (operator action)
Out of scope of this runbook — done by the platform team.

### 8. Verify new GPU is detected and has at least 14 GB free
```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```
**Expected:** the 16 GB card is listed; `memory.free` ≥ 14000 MiB.

### 9. Apply the 16 GB systemd override (already in repo, just install + reload)
```bash
sudo install -m 644 \
    /home/ubuntu/PycharmProjects/DocWain/deploy/docwain-vllm-fast.service.16gb \
    /etc/systemd/system/docwain-vllm-fast.service
sudo systemctl daemon-reload
```

### 10. Start vLLM with the AWQ config + verify
```bash
sudo systemctl start docwain-vllm-fast
until curl -sf http://localhost:8100/v1/models > /dev/null; do sleep 5; done
curl -s http://localhost:8100/v1/models | python -m json.tool | head -10
```

### 11. Start app + workers
```bash
sudo systemctl start docwain-app docwain-celery-worker docwain-celery-beat
until curl -sf http://localhost:8000/api/health > /dev/null; do sleep 5; done
curl -s http://localhost:8000/api/health | python -m json.tool
```
**Expected:** all components healthy. The Insights Portal flag drop-in (`/etc/systemd/system/docwain-app.service.d/insights-flags.conf`) is preserved across the swap; nothing flag-side changes.

### 12. Smoke the user-facing path
```bash
curl -s -X POST -H "Content-Type: application/json" -d '{
  "query": "Test ping",
  "profile_id": "smoke-profile-2026-04-25",
  "subscription_id": "smoke-sub",
  "user_id": "swap@test.com",
  "stream": false
}' http://localhost:8000/api/ask | python -c "
import sys,json
d=json.load(sys.stdin)
print(d['answer']['response'][:200])
"
```

---

## Rollback path (if AWQ on 16 GB fails)

If `vllm-fast` won't start after step 10, or `/api/ask` regresses badly after step 12, **fail over to Ollama Cloud 397B as primary**:

### A. Switch the LLM gateway primary to Cloud
```bash
# Set the gateway primary to ollama_cloud (already wired in src/llm/gateway.py)
sudo systemctl edit docwain-app --full   # add Environment="DOCWAIN_LLM_PRIMARY=ollama_cloud"
sudo systemctl edit docwain-celery-worker --full   # same env override
sudo systemctl restart docwain-app docwain-celery-worker
```

### B. Verify Cloud is responding + flag-on services keep working
```bash
curl -s http://localhost:8000/api/health   # ollama_cloud should be healthy; ollama_local may be degraded
```

### C. (later) Investigate AWQ artifact + retry
The AWQ artifact remains on disk. Re-run the smoke test (steps 2–4) once a fix is identified; rollback to local primary by reverting the env override.

---

## What this runbook does NOT change

- Insights Portal feature flags (`/etc/systemd/system/docwain-app.service.d/insights-flags.conf`) — preserved.
- Mongo `insights_index` / Qdrant `insights` / Neo4j `:Insight` — preserved.
- Adapter framework (`src/intelligence/adapters/generic.yaml` + RepoAdapterBackend) — preserved.
- Researcher v2 task code — preserved.
- /api/ask proactive injection — preserved.

The whole product shift survives the GPU swap because the engineering layer is decoupled from the model serving layer.
