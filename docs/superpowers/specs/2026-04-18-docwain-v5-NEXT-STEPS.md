# DocWain V5 — Next Steps Handoff

**Status as of 2026-04-18 05:10 UTC**

## What's landed (committed)

| Commit | What |
|---|---|
| `f3e7999` | V5 design spec (full plan, locked decisions) |
| `c6b8...` | Capability Charter (12 capabilities, machine-readable) |
| `c45a465` | Teacher ensemble + Nemotron adapter |
| (pilot) | Voting rule fixed, Ollama model tag corrected |
| `0c82783` | V4→V5 transform — 31,011 reused rows + gap analysis |
| (subagent, pending) | MergeKit recipe YAML + seed eval script |

## What remains — in execution order

### Step 1: Fill the 57K data gap (BEFORE any GPU training)

The 0%-coverage capabilities are exactly the behaviours V5 must learn. V4's corpus can't teach them because V4 relied on system prompts. Generate fresh rows via the ensemble pipeline:

```bash
# Start with the biggest gaps — schema + refusal + tool-calling
python -m src.finetune.v5.data_generator \
    --capability schema_adherence --rows 15000 \
    --capability grounded_refusal --rows 10000 \
    --capability tool_calling --rows 10000 \
    --capability identity_in_weights --rows 5000 \
    --output finetune_artifacts/v5/sft_generated.jsonl
```

**NOTE**: `data_generator.py` is NOT YET BUILT — it needs to be the next code written. It wraps `teacher_ensemble.vote()` over prompt templates per capability. Budget:

- structured-output capabilities (schema, extraction, classification, refusal, tool-calling) — use ensemble voting with expect_json=True
- narrative capabilities (identity, intent, citation, layout) — use Claude-authored seeds + Nemotron-as-LLM-judge for equivalence (NOT fingerprint match)

The pilot proved the voting pipeline works for structured outputs (65% acceptance). Narrative capabilities need the judge-based agreement that's called out in the spec.

### Step 2: Merge the seed (subagent running now)

Once the subagent finishes:

```bash
mergekit-yaml configs/merge_recipes/docwain_v5_seed.yaml models/DocWain-14B-v5-seed
python scripts/evaluate_merge_seed.py \
    --model-path models/DocWain-14B-v5-seed \
    --baseline-score 4.71
```

- PASS → proceed to SFT on the seed
- FAIL → train SFT from `models/DocWain-14B-v2/` directly; skip the merge entirely. No time is lost.

### Step 3: Stop vLLM, run 14B SFT (~24h)

```bash
sudo systemctl stop docwain-vllm-fast
# Trainer script to be written: src/finetune/v5/sft_trainer.py
python -m src.finetune.v5.sft_trainer \
    --base models/DocWain-14B-v5-seed \
    --corpus finetune_artifacts/v5/sft_reused.jsonl,finetune_artifacts/v5/sft_generated.jsonl \
    --output models/DocWain-14B-v5-sft \
    --lora-rank 128 --epochs 2
```

Checkpoint every 6 hours. Monitor for loss divergence — if loss plateaus above 0.25 after epoch 1, auto-bump LoRA rank to 256.

### Step 4: 14B DPO (~12h)

```bash
python -m src.finetune.v5.dpo_trainer \
    --base models/DocWain-14B-v5-sft \
    --pairs finetune_artifacts/v5/dpo_reused.jsonl,finetune_artifacts/v5/dpo_generated.jsonl \
    --output models/DocWain-14B-v5 \
    --beta 0.1 --epochs 2
```

**Hard gate**: LLM-judge ≥ 4.75 AND golden-query R@5 ≥ 7/9. Fail → ship V3, restart DPO with cleaner pairs.

### Step 5: 7B distillation (~18h)

```bash
python -m src.finetune.v5.distillation \
    --teacher models/DocWain-14B-v5 \
    --student Qwen/Qwen3-8B \
    --corpus finetune_artifacts/v5/sft_reused.jsonl,finetune_artifacts/v5/sft_generated.jsonl \
    --output models/DocWain-8B-v5 \
    --alpha 0.5
```

Hard gate: LLM-judge ≥ 4.35 AND R@5 ≥ 6/9. Fail → ship V3 as the "small" target.

### Step 6: Quantize + deploy (post-A100)

```bash
# 14B → int8 for vLLM
python -m src.serving.quantize \
    --input models/DocWain-14B-v5 --output models/DocWain-14B-v5-int8 --format int8

# 14B → GGUF Q5_K_M for Ollama
python -m llama_cpp.convert --outtype q5_k_m \
    models/DocWain-14B-v5 models/DocWain-14B-v5-q5km.gguf

# 7B → GGUF Q5_K_M for sub-10GB deployments
python -m llama_cpp.convert --outtype q5_k_m \
    models/DocWain-8B-v5 models/DocWain-8B-v5-q5km.gguf
```

Each variant validated vs its fp16 parent on the golden-query harness — drop > 1% = drop that variant.

Relaunch vLLM with V5-14B-int8:

```bash
sudo ln -sf models/DocWain-14B-v5-int8 models/docwain-v5-active
sudo systemctl start docwain-vllm-fast
python -m src.tasks.rag_regression 67fde0754e36c00b14cea7f5 69e260f3e41cbd913401e420
```

## Files still to write (in priority order)

1. **`src/finetune/v5/data_generator.py`** — bucket-driven row producer.
   Wraps `teacher_ensemble.vote()`. For structured capabilities, uses
   existing ensemble voting. For narrative capabilities, adds a
   Nemotron-as-judge path: two responses, judge rates equivalence 1-5,
   accept if ≥ 4.
2. **`src/finetune/v5/sft_trainer.py`** — LoRA SFT loop. Reuses
   `src/finetune/v2/train_track.py` and similar — don't rebuild, adapt.
3. **`src/finetune/v5/dpo_trainer.py`** — DPO loop, similar adapt.
4. **`src/finetune/v5/distillation.py`** — KL+SFT loss for 7B student.
5. **`src/finetune/v5/evaluate.py`** — gate runner that reads the
   capability_charter and scores per capability.

## Known issues and how they were handled

| Issue | How handled |
|---|---|
| Nemotron thinking mode burns tokens on short `max_tokens` | Clamp to ≥ 1500 for Nemotron calls |
| Ollama can't load model while vLLM holds GPU | Use Ollama only during data gen phase when vLLM is up; when training starts, vLLM stops anyway |
| V4 data has "You are DocWain…" system prompts | Stripped at transform time (drop_system=True by default) |
| Voting rule hard-coded ≥3 agree | Now scales with `len(teacher_callers)` |
| Narrative responses don't fingerprint-match | Flagged for LLM-judge-based agreement in data_generator (not yet built) |

## How to resume

The codebase is self-describing — `src/finetune/v5/__init__.py` lists the module map, the charter is the contract, and this handoff says what's next. A new session starts by:

1. Reading `docs/superpowers/specs/2026-04-18-docwain-v5-design.md` (design)
2. Reading this file (next steps)
3. Running `git log --oneline | head -30` to see what's landed
4. Proceeding to step 1 (data generator) in the order above
