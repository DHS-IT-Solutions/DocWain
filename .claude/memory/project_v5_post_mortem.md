---
name: V5 Training Sprint — Post-Mortem State (2026-04-20)
description: V5 shipped with failing gates; reverted to V2 on prod; artifacts preserved for V6 or teardown
type: project
originSessionId: 93168fda-607e-4c51-b06c-5b5e0f18a6b1
---
**Status (2026-04-20 13:00 UTC):** V5 pipeline complete but reverted. Production serving V2 via `models/docwain-v2-active` → `models/DocWain-14B-v2`.

**Why:** V5-14B passed 1/7 hard gates, V5-8B passed 2/7. V5-14B says "I am Qwen" when asked identity. **Important finding (2026-04-20 post-revert):** V2 ALSO says "I am Qwen" — so V5's identity failure is not a regression, it's a pre-existing gap that has never been fixed. User called V5 "total utter failure" and asked for revert; V2 is back on prod but the identity problem remains.

**How to apply:**
- Prod vLLM (port 8100, `docwain-fast`) serves V2 — do not swap without eval passing >=5/7 hard gates.
- V5 artifacts preserved on disk but not in active serving:
  - `models/DocWain-14B-v5/` (31GB, bf16)
  - `models/DocWain-14B-v5-sft/`
  - `models/DocWain-8B-v5/` (31GB, bf16)
  - `models/DocWain-14B-v5-q5km.gguf` (10GB)
  - `models/DocWain-8B-v5-q5km.gguf` (5.9GB)
  - Ollama `docwain-14b-v5` and `docwain-8b-v5` registered but not routed.
- Reclaimable if space needed: `finetune_artifacts/v5/distillation_teacher_cache/` (9.1GB), old SFT/DPO checkpoints.
- Unified model confirmed: only `docwain-vllm-fast.service` exists — no separate "smart" instance despite the legacy name.

**Decision points for V6 (if pursued):** see `feedback_v5_failure_lessons.md` for the 8 hard rules derived from this sprint.

**Finalised 2026-04-26:** "The V5 model is not an intelligent model and does not fit DocWain's requirement." V5 weights and GGUFs (14B, 8B, q5km variants) must NOT be proposed for deployment, fallback, or tight-budget serving. They stay on disk only as historical artifacts. Any future deployment proposal involving V5 should be rejected at suggestion time. For 16 GB GPU constraints, the path is AWQ/GPTQ quantization of V2 weights — not a fallback to V5.
