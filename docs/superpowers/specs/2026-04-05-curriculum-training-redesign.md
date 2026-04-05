# DocWain V2 Curriculum Training Redesign

**Date:** 2026-04-05
**Status:** Approved
**Supersedes:** Sequential per-track autonomous trainer (`autonomous_trainer.py`)

## Context

The original V2 training pipeline trained 6 tracks sequentially (excel_csv → layout → ocr_vision → reasoning → kg → visualization), each with its own synthetic data generator, programmatic rubric, and test bank. After 8 evaluated iterations on the first track (excel_csv), scores plateaued at 3.76/4.0 due to:

- **Low-diversity synthetic data** — 2500 examples from ~15 template functions
- **Train/eval format mismatch** — training used `<spreadsheet>` XML, eval used markdown tables
- **Mechanical reasoning** — template-generated chain-of-thought lacked analytical depth
- **No effective DPO** — disabled because synthetic pairs caused regression
- **Identical data per iteration** — same templates with different random seeds

## Design: Claude-Guided Unified Curriculum

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Claude-Guided Curriculum                │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Generate  │───>│  Train   │───>│  Eval    │          │
│  │ (subagent)│    │ (Unsloth)│    │(subagent)│          │
│  └──────────┘    └──────────┘    └─────┬────┘          │
│       ^                                │               │
│       │          ┌──────────┐          │               │
│       └──────────│ Analyze  │<─────────┘               │
│                  │ Failures │                           │
│                  │(subagent)│                           │
│                  └──────────┘                           │
│                                                         │
│  Gate 1: basics (3.5 avg, no dim < 3.0)                │
│  Gate 2: production (4.0 avg, no dim < 3.5)            │
│  Max iterations: 10, then escalate                      │
└─────────────────────────────────────────────────────────┘
```

**Key change:** All 6 capability areas train together in a single unified run. No sequential per-track progression. This eliminates catastrophic forgetting between tracks.

### Components

1. **Orchestrator** (`curriculum_trainer.py`) — manages pipeline state, launches subagents, triggers training, checks tiered gates, handles resume
2. **Data Generator** (`curriculum_generator.py`) — dispatches Claude Code subagents (via the Agent tool with `subagent_type="general-purpose"`) to generate diverse, realistic training examples in Qwen3 chat template format. Each subagent writes JSONL batches to disk.
3. **Trainer** — modified `train_track.py` with curriculum sampling and multi-checkpoint saving
4. **Evaluator** (`curriculum_evaluator.py`) — runs direct LoRA inference, dispatches subagents to judge responses
5. **Failure Analyzer** — subagent that clusters failures, hypothesizes root causes, produces generation briefs for targeted augmentation

### Data Generation

**Initial dataset (iteration 1): ~5K examples**

| Area | Count | Coverage |
|------|-------|----------|
| Excel/CSV Intelligence | 900 | Tabular QA, multi-sheet reasoning, formulas, aggregation, data types |
| Layout Intelligence | 800 | Field extraction, spatial relationships, nested structures, noise filtering |
| OCR/Vision | 700 | OCR error handling, confidence scoring, multi-language, handwriting context |
| Reasoning | 900 | Multi-hop inference, contradiction detection, evidence synthesis, confidence calibration |
| Knowledge Graph | 800 | Entity extraction, relationship mapping, cross-document linking |
| Visualization | 900 | Chart selection, data-to-insight, spec generation |

**Difficulty distribution:** 20% easy / 50% medium / 30% hard

**Generation rules:**
- Subagent receives a generation brief per capability area describing document types, complexity levels, and exact output format (Qwen3 `<|im_start|>` template with `<think>` blocks)
- Batches of 50 with diversity constraint — no two examples share same domain + document type + question pattern
- Each example includes: realistic document context, natural user question, deep CoT (5-15 steps), precise answer with confidence/citations, reference dict for pre-screening
- Training examples use the same format as the test bank (markdown tables, `[SPREADSHEET: ...]` headers)
- All data is purely synthetic — no customer documents, only patterns and metadata

**Subsequent iterations:** Failure analyzer generates 500-1500 targeted examples for weak areas per iteration. Maximum total dataset: 15K examples.

### Training Configuration

- **Base model:** `unsloth/Qwen3-14B-bnb-4bit`
- **LoRA:** rank=64, alpha=128, dropout=0.05, targets: q/k/v/o/gate/up/down
- **Epochs:** 3 (iteration 1), 2 (subsequent iterations)
- **Learning rate:** 2e-5, cosine decay, 10% warmup
- **Batch:** 4 per device, gradient accumulation 8 (effective batch 32)
- **Max seq length:** 4096
- **Curriculum sampling:** examples ordered easy → medium → hard within each epoch
- **Checkpoints:** save every 25% of training steps. Eval all checkpoints, pick best (not necessarily final).
- **Merge:** only merge winning checkpoint to FP16. No Ollama/GGUF export until production gate passes.

### Evaluation

**Method:** Direct inference via transformers + LoRA adapter. No merge/GGUF/Ollama overhead.

**Test bank:** Existing frozen 300-example test bank (50 per track).

**Judging:** Claude Code subagents score each response on 4 dimensions per area (1.0-5.0 scale):
- Factual correctness — right values, correct calculations
- Reasoning quality — logical, complete, non-mechanical CoT
- Completeness — all parts of the question addressed
- Grounding — cites specific sources/cells/sheets from context

**Batching:** 10 examples per subagent invocation. Structured JSON scores returned.

**Tiered gates:**
- **Gate 1 (basics):** avg >= 3.5, no dimension below 3.0. Must pass within 5 iterations.
- **Gate 2 (production):** avg >= 4.0, no dimension below 3.5. Must pass within 10 total iterations.
- **Escalation:** stop training, report scores and failure analysis, wait for human input.

### Failure Analysis & Adaptive Augmentation

After each eval, the failure analyzer subagent receives all 300 scores with model responses and produces:

1. **Failure taxonomy** — clusters failures into patterns
2. **Root cause hypothesis** — data gap vs. model capacity issue
3. **Generation brief** — specific instructions for next round of targeted examples

**Rules:**
- Never regenerate full dataset — only augment
- New examples tagged with iteration number
- Same failure pattern persisting 3 iterations → flagged as capacity issue, not data issue
- Dataset hygiene scan before each training: dedup, contradiction check, format validation

### File Structure

```
finetune_artifacts/v2_curriculum/
├── state.json
├── training.log
├── dataset/
│   ├── iter_1_base.jsonl
│   ├── iter_N_augment.jsonl
│   └── combined.jsonl
├── eval/
│   ├── iter_N_results.json
│   └── iter_N_analysis.json
├── checkpoints/
│   └── iter_N/
│       ├── checkpoint-25pct/
│       ├── checkpoint-50pct/
│       ├── checkpoint-75pct/
│       └── checkpoint-final/
└── merged/
    └── final_fp16/
```

### Code Changes

- **New:** `src/finetune/v2/curriculum_trainer.py` — orchestrator
- **New:** `src/finetune/v2/curriculum_generator.py` — subagent dispatch for data generation
- **New:** `src/finetune/v2/curriculum_evaluator.py` — subagent dispatch for eval + judging
- **Modify:** `src/finetune/v2/train_track.py` — add curriculum sampling mode, multi-checkpoint saving
- **Keep:** `eval/test_bank.py` (frozen eval set), `eval/rubrics.py` (reference for subagent judge)
- **Deprecate:** `autonomous_trainer.py`, per-track data generators (stay in tree, no longer invoked)

### Cleanup

Delete all contents of `finetune_artifacts/v2_upgrade/` (117GB). Old checkpoints trained on misaligned data are not useful as starting points. Train from base model.

### Constraints

- All training data must be purely synthetic — no customer documents
- GPU priority: vLLM serving > training. Training scavenges idle GPU time.
- No Co-Authored-By, Claude, or Anthropic references in code/commits
