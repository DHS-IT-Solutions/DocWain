# DocWain V2 Training Retrospective

**Date:** 2026-04-05
**Pipeline:** Curriculum Training (replaced sequential per-track trainer)

## Results

| Metric | Old Pipeline | New Pipeline |
|--------|-------------|--------------|
| Iterations | 8 (1 track only) | 3 (all 6 tracks) |
| Training examples | 20,000+ (templates) | 863 (subagent-generated) |
| Best score | 3.76/5.0 (excel_csv only) | 4.71/5.0 (all tracks) |
| Time to best | ~8 hours | ~2 hours |
| Production gate | Never passed | Passed at iter 3 |

### Per-Track Scores (Iteration 3)

| Track | Score | Key Dimensions |
|-------|-------|---------------|
| excel_csv | 5.00 | Perfect factual accuracy, strong grounding |
| layout | 5.00 | Complete field extraction, spatial reasoning |
| visualization | 4.80 | Chart specs, trend detection, Vega-Lite output |
| ocr_vision | 4.59 | OCR error correction, confidence handling |
| kg | 4.55 | Entity extraction, relationship mapping |
| reasoning | 4.31 | Multi-hop inference, evidence synthesis |

## What Worked

### 1. Unified Curriculum > Sequential Tracks
Training all 6 capability areas together prevented catastrophic forgetting.
The old pipeline trained one track at a time, and each track's training
degraded the previous ones. Curriculum sampling (easy→medium→hard) within
a single run maintained all capabilities simultaneously.

### 2. Quality Over Quantity
603 subagent-generated examples outperformed 20,000+ template examples.
The key difference: diversity, realistic document contexts, natural questions,
and deep chain-of-thought reasoning. Templates recycled the same 15 patterns
with different random values — the model memorized patterns without generalizing.

### 3. Targeted Augmentation from Failure Analysis
The adaptive loop was highly effective:
- Layout completeness: 3.45 → 5.00 with just 30 targeted examples
- Visualization: 2.88 → 4.80 with 120 targeted examples (chart specs)
- OCR: 2.34 → 4.59 after fixing test bank format alignment

### 4. Subagent Judges > Programmatic Rubrics
Subagent judges identified issues that regex matching couldn't:
- Response truncation (completeness issue, not accuracy)
- Hallucination vs legitimate refusal (OCR with empty context)
- Reasoning depth vs mechanical step enumeration

### 5. Direct LoRA Inference for Fast Eval
Skipping the Ollama round-trip (merge→GGUF→load→query) saved 30 minutes
per eval cycle. Direct LoRA inference via Unsloth took 15 minutes.

## What Didn't Work

### 1. Template Data Generators
Low diversity: 15 templates × random values = structurally identical examples.
The model learned to pattern-match templates rather than understand documents.

### 2. Train/Eval Format Mismatch
Training used `<spreadsheet>` XML format but eval used markdown tables.
This single mismatch accounted for a significant portion of the score gap.

### 3. DPO with Synthetic Pairs
Disabled because both chosen/rejected responses came from templates.
Could work with subagent-generated preference pairs in future.

### 4. Ollama-Based Eval Loop
30-minute overhead per cycle (merge + GGUF export + Ollama load + 300 queries).
The bottleneck was GGUF conversion, not inference.

### 5. Insufficient max_new_tokens for Eval
Using 256-512 tokens caused artificial truncation, especially for
visualization (chart specs need 500+ tokens) and reasoning (long CoT).
1024 tokens is the minimum for reliable evaluation.

## Key Design Decisions

### LoRA Configuration
- **Rank 64** (not 128): Higher rank showed minimal improvement for 2x memory.
- **Dropout 0.05** (not 0.0): Prevents overfitting on small datasets.
- **All projections**: q, k, v, o, gate, up, down — maximum expressiveness.

### Training Schedule
- **3 epochs initial, 2 epochs augmentation**: Sufficient for convergence.
- **Cosine LR with 10% warmup**: Standard, works well.
- **Batch 32**: Stable gradients without excessive memory.

### Evaluation Design
- **60 examples** (10 per track): Sufficient for signal, fast iteration.
- **4 dimensions**: factual_correctness, reasoning_quality, completeness, grounding.
- **Tiered gates**: basics (3.5/3.0) then production (4.0/3.5).

## Automated Pipeline Design

The production pipeline (`auto_curriculum.py`) integrates:

1. **Pattern Collection**: Harvest anonymized metadata from MongoDB (doc types,
   entity patterns, quality signals) — no customer content.
2. **Pattern-Enhanced Generation**: Inject real-world patterns into subagent briefs
   so training data reflects actual document distributions.
3. **Feedback Integration**: Low-confidence queries and failure categories from
   `learning_signals` inform augmentation targeting.
4. **Model Promotion**: Automatic symlink update and vLLM restart on gate pass.

### When to Retrain
- Monthly scheduled run (patterns evolve as customers upload new doc types)
- Triggered when low-confidence rate exceeds 30% (from feedback tracker)
- Triggered when new document types are detected in pattern collection
- Manual trigger via `python -m src.finetune.v2.auto_curriculum`

### Data Budget
- Initial dataset: ~600-900 examples (1-2 hours generation)
- Augmentation per iteration: 50-200 targeted examples
- Maximum dataset: 15,000 examples
- Typical convergence: 2-4 iterations

## File Reference

| File | Purpose |
|------|---------|
| `src/finetune/v2/auto_curriculum.py` | Production entry point |
| `src/finetune/v2/curriculum_trainer.py` | Core orchestrator |
| `src/finetune/v2/curriculum_generator.py` | Data generation briefs |
| `src/finetune/v2/curriculum_evaluator.py` | LoRA eval + judging |
| `src/finetune/v2/pattern_collector.py` | MongoDB pattern harvesting |
| `src/finetune/v2/train_track.py` | Unified SFT trainer |
| `finetune_artifacts/v2_curriculum/` | Artifacts directory |
