---
name: Backend Quality Audit — 2026-04-23 Result
description: Cloud 397B is materially more intelligent than local 14B; "better preprod_v01 responses" are from the model, not the code
type: project
originSessionId: dc7597b6-0d4a-464a-8305-e7a3b998992a
---
Ran a 10-query side-by-side audit on 2026-04-23 on the DocWain Ujjwal Raj resume corpus, comparing three backends on identical retrieval + prompt:

- **A_vllm_local** — DocWain-14B-v2 bf16 via `docwain-vllm-fast.service` at port 8100
- **B_ollama_local** — DocWain-14B-v2 Q5_K_M GGUF via Ollama local (`DHS/DocWain:latest`)
- **C_ollama_cloud** — `qwen3.5:397b` via Ollama Cloud (preprod_v01's current gateway primary)

Branch: `audit/backend-quality-2026-04-23`. Worktree: `~/.config/superpowers/worktrees/DocWain/audit-backend-quality-2026-04-23`. Final artifacts at `docs/audits/2026-04-23-backend-quality/{query_bank.md, results.md, results.json, verdict.md}`.

## Headline findings

- **Cloud 397B wins intelligence and grounding decisively** when it produces output — clean tables with dates/roles/metrics (Q2, Q3), insightful cross-resume reasoning (Q9), sharp interview-framing (Q10 caught that the candidate is still a student).
- **Cloud 397B returned EMPTY on 3 of 10 queries** because `num_predict=2048` was consumed entirely by `<think>` reasoning. Audit-config issue, not a model capability issue. A rerun with `num_predict≈6144` would likely reclaim those queries.
- **Ollama local (Q5_K_M GGUF) failed catastrophically on easy questions:** hallucinated `$100,000` on Q1 (name/email/phone), `$9,000.00` on Q6 (job title), and produced a **2048-token degenerate loop** on Q8 (repeating entity names ~30 times). This is production-embarrassment class, not "a bit noisy."
- **vLLM local (same weights, bf16) was shallower but safer** — never catastrophically wrong, always produced coherent output. The 14B model has limits but vLLM doesn't add failure modes the way Ollama local does.
- **Same weights, two engines → different quality.** Ollama Q5_K_M on this corpus produced real quality failures vLLM bf16 did not. Quantization precision vs engine implementation is still tangled; matched-bit rerun would isolate it.

**Why:** This audit was step 1 of the post-preprod roadmap (project_post_preprod_roadmap.md). The user's observation that "preprod_v01 responses are better than main" is CORRECT — but the mechanism is **the 397B Cloud model**, not the branch's code. Main probably routes to the same place, or whatever other routing it has is a separate question. The engineering-vs-model attribution matters for the roadmap.

## Verdict recommendation was OVERRIDDEN 2026-04-23

After reading the verdict, Muthu chose **vLLM local primary, Ollama Cloud as fallback** — the opposite of the verdict's recommendation. The trade-off was intentional and latency-driven: vLLM local avg wall-time is 4.7s vs Cloud 45s — roughly 10× faster, and user-perceived latency is the explicit product priority. The Q9/Q10 intelligence gap is acknowledged and will be mitigated at the engineering layer via the Researcher Agent (precomputed insights) and stronger prompting/retrieval — not by serving Cloud. Do NOT interpret this as the audit being wrong; the audit correctly surfaced the quality gap. The decision was to accept that gap for latency.

## Decision impact for other roadmap items

**On item #5 (serving decision — vLLM vs Ollama):**
- Do NOT switch to local Ollama. Q5_K_M + Ollama engine produces hallucinations and degenerate loops that vLLM with same weights does not produce.
- If local is the decision: use vLLM, and set expectations that local 14B trails Cloud 397B noticeably on response-intelligence queries (Q9, Q10 gap was large).
- If cost/latency of Cloud becomes the blocker: the workstream shifts from "swap engines" to "tolerate Cloud latency" (prompt compression, retrieval narrowing, streaming, caching).

**On item #4 (RAG intelligence upgrade):**
- The engineering-first rule still applies: prove prompt/retrieval/reasoning-layer changes before any retraining.
- With Cloud 397B as primary, RAG improvements compound with a capable generator. With local 14B as primary, engineering-layer improvements may bottleneck on model capacity.

**On the existing `feedback_engineering_first_model_last.md` rule:**
- Vindicated: this audit is engineering-layer work that produced a concrete, actionable finding in ~2 hours. No model retraining was required.

## How to apply

- Before any future "switch to local serving" decision, re-run this audit (or an equivalent) on a multi-document corpus with `num_predict≥6144`.
- Never use Ollama local with Q5_K_M for user-facing traffic — it can fabricate numbers on factual questions.
- The gateway's fallback chain (Cloud → GPT-4o → local) should prefer local vLLM over local Ollama — currently main gateway has no vLLM wiring; preprod_v01 likewise has local vLLM running but unused.
- If you see a retrain proposal to "close the gap to Cloud 397B with local 14B weights," push back hard — the Q9/Q10 gap here is a model-scale gap, not a training-data gap.

## Known audit limitations

- Single resume only, 10 queries — directional, not statistically robust.
- Precision asymmetry — vLLM bf16 vs Ollama Q5_K_M. Some of Ollama's errors may be quantization artifacts, not engine artifacts.
- `num_predict=2048` was too small for the cloud reasoning model. A rerun of just Cloud with a larger budget should validate Q1/Q4/Q5.
