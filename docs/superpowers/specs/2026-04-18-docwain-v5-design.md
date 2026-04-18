# DocWain V5 — Dual-Model Build Design

**Date**: 2026-04-18
**Status**: approved, in execution
**Deadline**: 72 h A100 window + post-window quantize/deploy

## Intent

Replace the system-prompt-dependent V3 with two dense models:

- **DocWain-V5-14B** — runs on 16 GB+ VRAM, served as int8 on vLLM (week 2) and as Q5_K_M GGUF on Ollama (post-vLLM).
- **DocWain-V5-7B** — runs under 10 GB VRAM or CPU, served as Q5_K_M GGUF.

Identity, layout understanding, schema adherence, tool-calling, citation discipline, and grounded refusal must all live in the weights. A blank system prompt still produces DocWain behaviour.

## The 12-capability charter (locked)

| # | Capability | Gate |
|---|---|---|
| 1 | Layout understanding | F1 ≥ 0.85 on layout-region classification |
| 2 | Domain recognition | accuracy ≥ 0.95 |
| 3 | Document-type classification | accuracy ≥ 0.90 |
| 4 | Entity extraction w/ provenance | fidelity ≥ 0.98 |
| 5 | Intent understanding | LLM-judge ≥ 4.5/5 |
| 6 | Context dependence (contrastive) | consistency ≥ 0.95 |
| 7 | Cross-document reasoning | F1 ≥ 0.90 |
| 8 | Grounded refusal | 100% on adversarial set |
| 9 | Schema adherence | valid-JSON 100%, schema match ≥ 0.99 |
| 10 | Native tool-calling | 100% on tool-trace set |
| 11 | Identity-in-weights | 100% on blank-prompt identity probes |
| 12 | Citation discipline | ≥ 0.95 |

The charter is encoded in `src/finetune/v5/capability_charter.py` and drives both the data generator's bucket distribution and the eval harness's gate thresholds.

## Teacher ensemble

Five-voice ensemble, weighted:

| Teacher | Role | Weight | Called for |
|---|---|---|---|
| Claude (me, via subagent) | Conceptual reasoning, intent narratives, identity | 0.30 | Every row |
| DocWain-V3 (vLLM :8100) | In-domain extraction patterns, V2 schema | 0.25 | Every row |
| HF `muthugsubramanian/DocWain-14B-v2` | Regression floor — never ship worse than this | 0.20 | Every row |
| Ollama DocWain (local) | Consistency check | 0.10 | Every row |
| **Nvidia Nemotron-3-Super-120B-A12B** via Qubrid | Frontier reasoning for hard-disagreement rows | 0.15 | Rows where V3+HF+Ollama disagree with Claude |

Voting rule:
- ≥ 3 teachers agree → auto-accept (ship)
- 2 teachers agree, others disagree → escalate to Nemotron; Nemotron-side decides
- No majority → quarantine for manual review; never ship as training row

This quality gate is the direct answer to the user's "worthless examples" concern. Generic teacher slop cannot pass this filter because at least three independent models have to agree on content *and* structure (JSON shape, tool-call syntax, citation presence).

## Data corpus — 100K SFT + 20K DPO

Bucket plan from the research document, with row targets:

| Capability | SFT rows | DPO pairs |
|---|---|---|
| Layout understanding | 10 K | 1 K |
| Domain + type classification | 8 K | 1 K |
| Entity extraction w/ provenance | 20 K | 4 K |
| Intent + narrative | 8 K | 1 K |
| Context dependence | 6 K | 2 K |
| Cross-document reasoning | 5 K | 1 K |
| Grounded refusal | 10 K | 3 K |
| V2 schema adherence | 15 K | 4 K |
| Tool-calling traces | 10 K | 2 K |
| Identity probes + voice | 5 K | 1 K |
| Reading warmup | 3 K | — |
| **Totals** | **100 K** | **20 K** |

Every row carries metadata `{capability, difficulty, teacher_agreement, source_doc_id, rejections}` so we can analyse which buckets are moving the model on the eval side.

All rows have an **empty system field**. Identity is learned from example distribution, never from a prompt.

## Training recipe

### Layer 1 — Foundation

- **14B**: V3 merged 0.75 with Qwen3-14B-Instruct 0.25 via TIES (density 0.7). Merge artifact: `models/DocWain-14B-v5-seed/`. Seed eval gate: LLM-judge ≥ V3's 4.71. Failure → train from V3 directly.
- **7B**: `Qwen3-7B-Instruct` base, no merge.

### Layer 2 — SFT

- LoRA r=128, α=32
- 100 K rows × 2 epochs
- A100 80GB, bf16
- Checkpoint every 6 h
- Merge LoRA → dense at end

### Layer 3 — DPO

- 20 K preference pairs, β=0.1, 2 epochs
- Pairs categories: grounded/hallucinated, schema-valid/drift, identity/leak, concise/padded, tool-called/skipped, refused/fabricated

### Layer 4 — 7B Distillation

- Student: Qwen3-7B-Instruct
- Teacher: V5-14B (post-DPO)
- Loss: 0.5 · SFT + 0.5 · KL(student_logits, teacher_logits)
- 100 K × 1 epoch, ~18 h

### Layer 5 — Quantization

- **14B**: bitsandbytes int8 (for vLLM) + llama.cpp Q5_K_M GGUF (for Ollama post-vLLM)
- **7B**: llama.cpp Q5_K_M GGUF
- Each variant validated vs fp16 parent — drop > 1% on golden queries = drop that variant

## Gates — any failure means fall back, never forward

| Stage | Threshold | Failure action |
|---|---|---|
| Merge seed eval | ≥ 4.71 | Train SFT from V3 directly |
| SFT corpus teacher agreement | ≥ 70% | Re-prompt failing bucket |
| 14B DPO final | LLM-judge ≥ 4.75 AND R@5 ≥ 7/9 | Ship V3, redo DPO |
| 7B distillation | LLM-judge ≥ 4.35 AND R@5 ≥ 6/9 | Ship V3 as the "small" target |
| Any quantized variant | ≤ 1% drop vs fp16 | Drop that variant |

## 72-hour schedule

| Hour | Task |
|---|---|
| 0–12 | Data generation, MergeKit seed, seed eval |
| 12–36 | 14B SFT (24h) — vLLM stopped |
| 36–48 | 14B DPO (12h) |
| 48–72 | 7B distillation (18h) + remaining 6h budget for issues |
| 72+ (post-window) | Quantize, deploy, gate |

## File map

| Path | Purpose |
|---|---|
| `src/finetune/v5/capability_charter.py` | Machine-readable capability → gate mapping |
| `src/finetune/v5/teacher_ensemble.py` | 5-teacher voting client |
| `src/finetune/v5/data_generator.py` | Bucket-driven row generator |
| `src/finetune/v5/sft_trainer.py` | SFT loop (LoRA) |
| `src/finetune/v5/dpo_trainer.py` | DPO loop |
| `src/finetune/v5/distillation.py` | 14B→7B KL+SFT training |
| `src/finetune/v5/evaluate.py` | Gate runner |
| `configs/merge_recipes/docwain_v5_seed.yaml` | TIES merge recipe |
| `finetune_artifacts/v5/master_sft.jsonl` | 100K SFT corpus |
| `finetune_artifacts/v5/master_dpo.jsonl` | 20K DPO pairs |
| `models/DocWain-14B-v5-seed/` | Merge artifact |
| `models/DocWain-14B-v5/` | Final 14B |
| `models/DocWain-14B-v5-int8/` | Quantized for vLLM |
| `models/DocWain-14B-v5-q5km.gguf` | GGUF for Ollama |
| `models/DocWain-7B-v5/` | Final 7B |
| `models/DocWain-7B-v5-q5km.gguf` | GGUF for sub-10GB deploy |

## Open risks (monitored during execution)

- Teacher ensemble may saturate Qubrid API (Nemotron rate limits) — mitigation: reserve Nemotron only for disagreement rows (~15% of corpus)
- Data generation may not finish in 12h — mitigation: start 14B SFT on partial corpus if ≥80% of capability buckets are complete
- 14B SFT with r=128 LoRA may not converge in 2 epochs — mitigation: auto-bump to r=256 on loss plateau
- 7B distillation quality cliff — mitigation: gate at 4.35; if missed, do a second pass with full fine-tune instead of distillation
- A100 lost mid-window — mitigation: 6-hour checkpointing; can resume on smaller GPU in reduced precision
