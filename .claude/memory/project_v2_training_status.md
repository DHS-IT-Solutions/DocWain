---
name: V2 Training Pipeline Status
description: Production model live on vLLM, codebase cleaned (15K lines removed), automated pipeline with pattern capture ready
type: project
---

## V2 Training Pipeline — Status as of 2026-04-05

**Phase: PRODUCTION. Model serving via vLLM on port 8100.**

### Current Model
- **Checkpoint:** `finetune_artifacts/v2_curriculum/checkpoints/iter_3/merged_16bit`
- **Symlink:** `models/docwain-v2-active` → iter_3 checkpoint
- **Serving:** vLLM 0.19.0, bfloat16, A100 80GB, port 8100
- **Score:** 4.71/5.0 overall (production gate passed)
- **Dataset:** 863 examples, 3 iterations

### Automated Pipeline
- **Entry point:** `python -m src.finetune.v2.auto_curriculum`
- **Pattern collector:** `src/finetune/v2/pattern_collector.py` — harvests anonymized doc metadata from MongoDB
- **Triggers for retraining:**
  - Monthly scheduled run
  - Low-confidence rate >30% (from feedback tracker)
  - New document types detected
  - Manual trigger

### Codebase State
- 15,030 lines of dead code removed (old phase scripts, template generators, deprecated trainer)
- 77GB of old training artifacts cleaned up
- 13 active Python files in `src/finetune/v2/`
- 47 tests passing
- V1 finetune code (`src/finetune/`) kept separately — used by API for user-feedback-driven finetuning

### Key Files
| File | Purpose |
|------|---------|
| `auto_curriculum.py` | Production pipeline entry point |
| `curriculum_trainer.py` | Core orchestrator (generate→train→eval→analyze) |
| `curriculum_generator.py` | Subagent brief generation |
| `curriculum_evaluator.py` | LoRA inference + judging |
| `pattern_collector.py` | MongoDB pattern harvesting |
| `train_track.py` | Unified SFT trainer |

### vLLM NVML Workaround
The host has an NVML driver/library version mismatch. vLLM's bundled pynvml was replaced with a shim at `.venv/lib/python3.12/site-packages/vllm/third_party/pynvml.py`. The real file is backed up as `pynvml_real.py`. This allows vLLM to detect CUDA platform correctly.

**How to apply:** Model is live. For retraining, run `auto_curriculum.py`. For serving issues, check the NVML shim.
