# Backend Quality Audit — Design

**Date:** 2026-04-23
**Branch:** `preprod_v01`
**Status:** Approved for implementation
**Owner:** Muthu (audit author: pair-working session)

## 1. Purpose

Qualitative A/B/C comparison answering one question: **given the same DocWain question over the same documents, which of the three serving backends produces the most intelligent, grounded, complete response?**

The answer informs the larger serving-engine decision (directive #5 in `project_post_preprod_roadmap.md`). Today preprod_v01 serves every query from Ollama Cloud `qwen3.5:397b` — the A100 + local vLLM are idle. Before investing in a latency audit or a serving swap we need to know whether the local DocWain-14B-v2 model can match the cloud 397B model it would replace.

## 2. Scope

### In scope

- Run 10 representative queries through 3 backends.
- Hold retrieval context, prompt builder, and sampling params constant across backends.
- Produce a side-by-side response table and a written per-dimension verdict.
- Emit a single recommendation on which backend currently produces the best responses.

### Explicit non-goals

- **Not a latency audit.** Wall time is informational only; no p50/p95/p99 analysis, no concurrency curve.
- **Not a production routing change.** The harness runs on the side. No edits to `src/llm/gateway.py` or any production path.
- **Not a fine-tuning signal.** No training data is generated from the outputs.

## 3. Three backends compared

| Column | Backend        | Model                | Precision     | Endpoint                                                         |
|--------|----------------|----------------------|---------------|------------------------------------------------------------------|
| A      | vLLM local     | DocWain-14B-v2       | bf16          | `http://localhost:8100/v1` (`docwain-vllm-fast.service`, model id `docwain-fast`) |
| B      | Ollama local   | DocWain-14B-v2       | Q5_K_M GGUF   | `http://localhost:11434` (tag `DHS/DocWain:latest`, 9 GB)         |
| C      | Ollama Cloud   | `qwen3.5:397b`       | hosted        | `https://ollama.com` (preprod_v01's current gateway primary)      |

### Held constant across columns

- Same 10 queries (see §5).
- Same retrieval context per query — one Qdrant search per query, chunks reused verbatim across A/B/C.
- Same prompt — built once per query by the existing Reasoner (`src/generation/reasoner.py` → `src/generation/prompts.py`).
- Same sampling params: `temperature=0.4`, `top_p=0.85`, `max_tokens` computed per task type via the Reasoner's existing logic.

## 4. Methodology

1. **Pick one profile** with a representative spread of documents already embedded in Qdrant. Confirmed by the operator before the run; harness asserts the profile's Qdrant collection is non-empty.
2. **Retrieve once per query, reuse for all three backends.** Identical chunks, identical IDs, proven by logging chunk IDs next to each response. This is the critical fairness invariant — if each backend re-ran retrieval, we would be measuring retrieval variance in addition to generation.
3. **Build prompt once per query** via the production Reasoner code path. The only thing that changes per column is which HTTP endpoint receives the POST.
4. **Fire sequentially** (A → B → C) per query. Not concurrent — we are measuring quality, not throughput. Capture full response text, token count, wall time (wall time is informational).
5. **Record** every run into a structured table file (markdown + machine-readable JSON). Columns: `query_idx`, `query_type`, `prompt`, `retrieved_chunk_ids`, `response_A`, `response_B`, `response_C`, `tokens_A/B/C`, `wall_ms_A/B/C`, `verdict_A_B_C`, `winner`.
6. **Author the analysis** across four dimensions:
   - **Intelligence** — inferential depth, multi-document synthesis, judgment.
   - **Grounding** — every cited fact traceable to the retrieved chunks; no hallucination.
   - **Completeness** — did it actually answer the question fully.
   - **Style** — clarity, structure, formatting.

   Produce per-query verdicts, per-dimension aggregate rankings, and a single final winner (or "inconclusive" with stated reason).

## 5. Query bank (v0 — operator reviews before run)

| # | Type                  | Prompt |
|---|-----------------------|--------|
| 1 | Extraction QA         | "What is the total amount on the most recent invoice?" |
| 2 | Extraction QA         | "List every vendor that appears across the uploaded purchase orders." |
| 3 | Extraction QA         | "From the resumes, extract each candidate's most recent job title and company." |
| 4 | Cross-doc synthesis   | "Compare the payment terms between the two quotes we uploaded — which is more favorable?" |
| 5 | Cross-doc synthesis   | "Are there any duplicate line items across the invoices this month?" |
| 6 | Short factual         | "Who signed the last contract?" |
| 7 | Short factual         | "When was the earliest document uploaded?" |
| 8 | Response intelligence | "Based on the invoices and contracts together, what's our likely exposure to vendor X next quarter?" |
| 9 | Response intelligence | "Walk through what these documents collectively tell us about this project's risk profile." |
| 10| Response intelligence | "What's the smartest question I should be asking about this set of documents that I haven't asked yet?" |

**"Response intelligence" is defined as:** inferential, multi-document, goes beyond what a direct retrieval can answer; requires the model to show judgment and synthesize across documents.

The operator may rewrite any prompt (especially 4, 8, 9, 10) to match real documents in the chosen profile. Rewrites happen before the run, not after.

## 6. Deliverables

A single branch off `preprod_v01` named `audit/backend-quality-2026-04-23`, containing:

1. `scripts/backend_quality_audit.py` — the harness (~150 LOC, standalone, no edits to `src/`).
2. `docs/audits/2026-04-23-backend-quality/query_bank.md` — version-controlled query bank.
3. `docs/audits/2026-04-23-backend-quality/results.md` — side-by-side table (all 30 responses).
4. `docs/audits/2026-04-23-backend-quality/results.json` — machine-readable raw data.
5. `docs/audits/2026-04-23-backend-quality/verdict.md` — written per-dimension analysis + final recommendation.

## 7. Exit criteria

- All 10 queries ran to completion against all 3 backends (no empty cells in the table).
- Retrieval chunk IDs recorded next to each response and are **identical** across columns for each query (audit fails if they diverge).
- A verdict exists for every query, plus an aggregate per-dimension ranking.
- A clear recommendation: `ship on vLLM local` / `stay on Ollama Cloud` / `swap to Ollama local` / `inconclusive — escalate to latency audit` / `local 14B cannot match cloud 397B — need larger local model`.

## 8. Operational risks & mitigations

| Risk | Mitigation |
|---|---|
| Ollama Cloud auth / quota failure mid-run | Preflight "hello" call at harness start; fail fast with actionable error. |
| vLLM + Ollama local coexistence on A100 | Ollama Q5_K_M (~9 GB) + vLLM bf16 (~28 GB) + KV cache fits in 80 GB, but verified with a single-prompt smoke test before the full run. |
| Qdrant collection missing/empty | Harness asserts non-empty retrieval for every query. Abort if any retrieval returns zero chunks. |
| PII in real-document responses | Outputs live in `docs/audits/` on the `audit/backend-quality-2026-04-23` branch only. Not merged to main, not pushed remote without operator review, deleted after decision. |
| Cloud wins obviously but is slow | Surfaced as a finding, not a failure. Verdict explicitly states the speed/quality trade-off. |
| vLLM returns an empty response due to `max_tokens` mismatch with OpenAI-compatible API | Harness logs token counts; if vLLM returns 0 tokens it re-fires with a lower cap and flags the row. |

## 9. Time estimate

- Harness implementation: ~1 hr.
- Preflight + smoke: ~10 min.
- Full run (10 × 3 sequential): ~15–20 min (bounded by Ollama Cloud latency).
- Analysis writeup: ~30–45 min.
- **Total:** ~2.5 hrs end-to-end.

## 10. Follow-on (not part of this audit)

- If vLLM local or Ollama local wins or ties cloud: proceed to the latency audit (original step 1 in the roadmap) on the winning local candidate.
- If cloud wins decisively: the serving decision becomes "stay on cloud, optimize network latency and prompt compression instead of swapping engines" — different workstream.
- If inconclusive: run with a second profile, or expand the query bank to 20, before escalating.
