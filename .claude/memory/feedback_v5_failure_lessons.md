---
name: V5 Pipeline Failure — Key Lessons
description: What went wrong in the V5 training sprint (2026-04-18 to 2026-04-20) so the same mistakes aren't repeated in V6+
type: feedback
originSessionId: 93168fda-607e-4c51-b06c-5b5e0f18a6b1
---
The V5 training sprint (14B SFT+DPO → 8B distill → quantize → deploy) shipped with 1/7 and 2/7 hard gates passing. Reverted to V2 on 2026-04-20. What to remember:

**Why: Multi-day A100 window was burned producing a model that still answered "I am Qwen" — identity-in-weights, the central premise of V5, did not take. Distillation inherited the defect because we didn't gate on the teacher.**

**How to apply — hard rules for any future training run:**

1. **Gate distillation on the teacher passing identity.** Never distill from a teacher that fails its own identity eval. KL loss copies defects as faithfully as it copies signal. If the 14B says "I am Qwen," the 8B will too.

2. **Identity-in-weights has never actually worked** — V2 ALSO responds "I am Qwen" when asked its name (verified 2026-04-20 after revert). This isn't a V5 regression; it's a pre-existing gap that every training run has masked. 800 identity rows in SFT was insufficient; V2's count wasn't tracked but was clearly also insufficient. If this is a hard requirement, (a) budget >3000 diverse identity framings, (b) verify on a held-out probe set BEFORE the full run, (c) seriously consider a gateway prompt shim as the pragmatic fix — weights-only identity is expensive and has repeatedly failed.

3. **Validate eval scorers against hand-labeled outputs before trusting results.** In V5, schema_adherence and tool_calling scored 0.0 partly because scorers mismatched Nemotron's actual output shape (```json fences vs `<tool_call>` tags). Run scorers on 10 known-good + 10 known-bad samples before the expensive eval.

4. **Stop the line when the teacher fails its gates.** V5-14B passed 1/7 hard gates. I proceeded to 8B distillation anyway, using a "partial-pass prefer_v5" fallback. That was wrong — the downstream run compounded the wasted compute.

5. **Orchestrator must not declare "pipeline complete" on rc=0 + output_present=False.** The v5_orchestrator.py bug at scripts/v5_orchestrator.py:635 sets `distill_failed=True` but doesn't halt. Future orchestrators: failure flag must short-circuit to FAILED, not advance to QUANTIZE/DEPLOY.

6. **TIES merging is not compatible with vision-grafted Qwen3.** The seed merge in V5 scored 2.81 vs V3's 4.71. Don't attempt TIES on custom-architecture models without a small-scale pilot first.

7. **Identity baked into prompt (gateway shim) is a valid fallback.** If identity-in-weights isn't working, inject "You are DocWain..." at the vLLM proxy. Not the design ideal, but ships a working product at zero training cost — better than waiting for a perfect weights-only solution.

8. **Session isolation during training.** Ollama auto-started mid-distillation and ate 5.9GB VRAM → OOM at step 1724. Before any long GPU run: `systemctl disable --now ollama`, verify nvidia-smi clean, then launch. Add to pre-flight checklist.
