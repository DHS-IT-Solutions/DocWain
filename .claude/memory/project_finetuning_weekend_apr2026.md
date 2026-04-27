---
name: Weekend Finetuning April 2026 Results
description: V3 training results, datasets, architecture decisions, and next steps for converting DocWain to base model
type: project
---

## Weekend Finetuning (Apr 10-12, 2026)

### Training Progression
| Version | SFT Loss | DPO Loss | SFT Examples | DPO Pairs |
|---------|----------|----------|-------------|-----------|
| Baseline (iter_3) | 1.034 | N/A | 863 | 0 |
| V1 Weekend Round 2 | 0.768 | 0.422 | 922 | 35 |
| V2 Distilled Round 2 | 0.260 | 0.089 | 8,162 | 1,319 |
| **V3 Final Round 2** | **0.127** | **0.096** | **28,808** | **3,814** |
| V4 Prepared (not yet trained) | - | - | 31,050 | 4,314 |

### Datasets Location
- V3 master SFT: `finetune_artifacts/teacher_data/master_v3.jsonl` (28,808)
- V3 master DPO: `finetune_artifacts/teacher_data/master_dpo_v3.jsonl` (3,814)
- V4 master SFT: `finetune_artifacts/teacher_data/master_v4.jsonl` (31,050)
- V4 master DPO: `finetune_artifacts/teacher_data/master_dpo_v4.jsonl` (4,314)
- Element extraction: `finetune_artifacts/teacher_data/element_sft.jsonl` (2,300)
- Element DPO: `finetune_artifacts/teacher_data/element_dpo.jsonl` (500)
- Advanced reasoning: `finetune_artifacts/teacher_data/advanced_sft.jsonl` (20,646)
- Advanced DPO: `finetune_artifacts/teacher_data/advanced_dpo.jsonl` (2,495)

### Data Generator Modules
- `src/finetune/distillation/generators.py` — basic extraction/analytical/boundary/reasoning
- `src/finetune/distillation/advanced_generators.py` — long-context, numerical, table, temporal, legal, comparison (20K+ examples)
- `src/finetune/distillation/document_element_generators.py` — invoice/PO/quote/Excel element extraction (2,800 examples)

### Infrastructure Built
- `scripts/weekend_finetune_loop.py` — autonomous training loop with state persistence
- `scripts/distill_from_claude.py` — Qdrant harvesting + data generation
- `scripts/test_extraction_accuracy.py` — real document extraction testing
- `scripts/evaluate_docwain.py` — LLM judge evaluation CLI
- `src/finetune/evaluation/llm_judge.py` — LLM-based scoring pipeline
- `src/serving/metrics.py` — latency tracking (p50/p95/p99)
- EAGLE3 speculative decoding configured in `systemd/docwain-vllm-fast.service`

### Extraction Test Results (14 real documents)
- 12/14 passed (86%), raw extraction 14/14 (100%)
- 2 failures due to generation pipeline returning empty (not model capability)
- Invoices: 100%, POs: 75%, Quotes: 86%, Excel: 75%, JPEG: 100%

### Best Checkpoint
- Path: `finetune_artifacts/weekend_loop/round_2/merged_16bit`
- GGUF: `finetune_artifacts/weekend_loop/round_2/gguf/model-Q4_K_M.gguf`
- Ollama: `DHS/DocWain:latest` (pushed to ollama.com/DHS/DocWain)
- vLLM symlink: `models/docwain-v2-active`

### Base Model Aggressive Round (Apr 13-14, 2026)
| Metric | Value |
|--------|-------|
| SFT Loss | 0.1405 |
| DPO Loss | 0.0945 |
| SFT Examples | 31,828 (with layout/OCR) |
| DPO Pairs | 4,319 |
| Training Time | 11.7 hours |
| Extraction Test | **12/12 (100% pass), 89.2% overall** |
| Format | Base model ("You are DocWain." minimal system prompt) |
| Checkpoint | `finetune_artifacts/weekend_loop/aggressive_round/merged_16bit` |
| Ollama | `DHS/DocWain:latest` pushed to ollama.com/DHS/DocWain |

### Key Insight for Next Run
**Convert DocWain to base model** — bake the system prompt and identity directly into model weights instead of passing as system prompt at inference time. This will:
- Reduce inference latency (no system prompt processing)
- Make behavior more consistent (identity is in weights, not prompt)
- Improve accuracy (model IS DocWain, doesn't need to be told to be DocWain)

**Why:** Current approach passes a 200+ line system prompt on every request. This wastes context window and the model sometimes ignores parts of it. Baking identity into weights via training makes it intrinsic.

**How to apply:** Include system prompt content as part of training examples WITHOUT the system prompt wrapper — train the model to respond as DocWain by default. Use chat template with empty/minimal system field.
