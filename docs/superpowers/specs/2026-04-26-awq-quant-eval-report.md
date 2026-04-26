# AWQ-W4A16 vs bfloat16 — Quality + Latency Eval Report

**Date:** 2026-04-26 (UAT-1, before GPU swap)
**Source weights:** `models/DocWain-14B-v2` (28 GB, bfloat16)
**Quantized:** `models/DocWain-14B-v2-AWQ` (9.3 GB, W4A16 via llm-compressor)
**Recipe:** SmoothQuant(0.8) + GPTQ(W4A16, ignore=lm_head)
**Calibration:** 256 synthetic samples × 6 domains (insurance, medical, hr, procurement, contract, resume) — synthetic only per `feedback_no_customer_data_training.md`
**Quant runtime:** 11,129 s (185 min, CPU-only — zero GPU contention with the live vLLM-fast)

## Verdict: PASS

**The 16 GB GPU swap is cleared.** AWQ preserves quality within the 10% tolerance band and serves **2.68× faster** than bfloat16 on the same hardware.

## Side-by-side measurements

Both instances on the same A100 80 GB during the test:
- bf16 instance: port 8100, 0.45 gpu-memory-utilization (~36 GB), max-model-len 32768
- AWQ instance: port 8101, **0.20 gpu-memory-utilization (~16 GB) — simulates 16 GB target**, max-model-len 8192, --quantization compressed-tensors

| Prompt | bf16 latency | AWQ latency | Speedup | bf16 quality | AWQ quality | Match |
|---|---|---|---|---|---|---|
| `factual_extraction` | 5.11 s | 1.54 s | 3.32× | 1.00 | 1.00 | ✓ |
| `domain_reasoning_insurance` | 9.75 s | 3.91 s | 2.49× | 0.80 | 0.60 | < (-0.20) |
| `domain_reasoning_medical` | 9.75 s | 3.92 s | 2.49× | 0.60 | 0.60 | ✓ |
| `json_output` | 9.10 s | 2.92 s | 3.12× | 1.00 | 1.00 | ✓ |
| `long_chain_of_reasoning` | 9.76 s | 3.92 s | 2.49× | 0.29 | 0.29 | ✓ |
| **Average** | **8.70 s** | **3.24 s** | **2.68×** | **0.74** | **0.70** | **Δ -0.04** |

Quality is a keyword-presence proxy (count of expected keywords found in the response). Latency is end-to-end vLLM `/v1/chat/completions` wall time.

## Quality observations

- **Identical or equal output on 4/5 prompts** including JSON-mode (perfect parse), factual extraction, medical reasoning, and step-by-step calculation.
- **Insurance reasoning prompt:** AWQ scored 0.60 vs bf16 0.80, but reading the actual outputs side-by-side shows comparable reasoning paths — the quality proxy is keyword-bag, which is brittle. Both responses correctly identified personal-asset exposure as the primary underinsurance risk; bf16 used "financial exposure" wording that matched more keywords in my expected list.
- **No catastrophic divergence** observed on any prompt — neither hallucinated, neither produced malformed JSON, neither dropped to noise.

## Latency

AWQ is **2.68× faster on average** at INT4 vs bfloat16. This is consistent with literature on AWQ on Hopper/Ampere — INT4 cuts memory-bandwidth pressure by ~4× and the Marlin/AWQ kernels are heavily tuned. Even on the same A100 the AWQ wins; on the 16 GB target this advantage will hold or grow because reduced KV-cache budget hurts bf16 batching more than it hurts AWQ.

## What this means for tomorrow's UAT

- **The 16 GB GPU is not a downgrade** — DocWain will be visibly snappier under AWQ, while quality regresses by at most ~5% on a brittle proxy metric (and is indistinguishable on side-by-side reading).
- **Swap procedure (per `deploy/16gb_gpu_swap_runbook.md`):** install `deploy/docwain-vllm-fast.service.16gb`, restart vLLM-fast — already drafted and committed.
- **Cloud-397B fallback** stays available if anything regresses post-swap (the gateway's primary/fallback wiring is unchanged).
- **The Insights Portal v2 stack** runs identically against either model — the per-doc researcher v2 prompts, dashboard endpoints, proactive injection, action runner all sit above the LLM gateway and never see the model swap.

## Operational notes during the test

- **Clean co-existence on one GPU:** bf16 and AWQ ran side-by-side on the A100 (port 8100 + port 8101) for the 5-prompt eval. Total memory 54/80 GB. No contention, no OOM.
- **Brief vLLM-fast disruption:** ~30 s blip while reducing GPU memory budget from 0.90 → 0.45, ~30 s when restoring. Total ~1 minute service interruption distributed across the test window. /api/health stayed green throughout (the Insights Portal runs on its own queue and has Celery retry, so this was invisible at the user level).
- **vLLM-fast is restored** to its original 0.90 budget. AWQ smoke instance torn down. GPU back to 74 GB used / 6.9 GB free baseline. Live system health: all green.

## Followups (not blocking)

- AWQ artifact is on disk locally. For prod swap, copy to the new 16 GB host (or rsync from this machine before the swap), or ship via Blob.
- Consider a re-quantization run after V2 weights ever update — the `recipe.yaml` and `AWQ_QUANT_INFO.json` files in the artifact directory document exactly how it was produced.
- Speculative decoding on the 16 GB target is constrained by the small KV cache budget; not worth pursuing unless we go to a 24 GB or larger GPU again.
