---
name: V2 Weights — HF Recovery Path
description: Local v2 merged weights were deleted 2026-04-17. Public copy at muthugsubramanian/DocWain-14B-v2 on HF was downloaded to models/DocWain-14B-v2/ and is now the active symlink target.
type: project
originSessionId: 9a15d065-2784-44cb-8db5-07a885ea1f7c
---
On 2026-04-17 the user discovered `models/docwain-v2-active` was a broken symlink (local `finetune_artifacts/{weekend_loop,v2_curriculum,v2_upgrade}/` checkpoints wiped by prior cleanup). vLLM-fast (PID 81277, up since 2026-04-16 16:37) held the only live copy in GPU RAM. Later that day the user confirmed a public backup at `https://huggingface.co/muthugsubramanian/DocWain-14B-v2` (6 safetensor shards, Unsloth-trained Qwen3 14B bfloat16, last modified 2026-04-16 06:18 UTC).

Recovery executed same day: downloaded the repo to `models/DocWain-14B-v2/` (28 GB), repointed `models/docwain-v2-active` symlink there, wrote `models/docwain-v2-active.score.json` as the "always the best" baseline (score 0.0 placeholder — any eval-gated retrain will beat it).

**Why:** Without a restartable copy on disk, DocWain was one GPU fault away from extinction. The HF backup makes restarts recoverable and lets us schedule maintenance without panic.

**How to apply:**
- `models/DocWain-14B-v2/` is the canonical on-disk weights dir. `models/docwain-v2-active` symlinks there.
- vLLM can now be restarted safely — it will load from the symlink target. PID 81277 is still the live serving process; do NOT restart gratuitously, but a restart no longer equals disaster.
- Re-train path (`autonomous_trainer` → weekend_finetune_loop) rebuilds iter_1→iter_3 from `finetune_artifacts/teacher_data/` (661M, intact) + `unsloth/qwen3-14b-bnb-4bit` base (9.3G, cached). `scripts/weekend_finetune_loop.py` hardcodes `ITER3_CHECKPOINT = finetune_artifacts/v2_curriculum/checkpoints/iter_3/merged_16bit` — that path no longer exists, so starting there requires running autonomous_trainer first.
- Any future training must beat the active score (enforced by `models/docwain-v2-active.score.json` — currently 0.0 placeholder). After retraining, update the score file with a real eval value and repoint the symlink via the promotion logic in `src/finetune/v2/auto_curriculum.py:promote_model` or the deploy phase of `scripts/weekend_finetune_loop.py`.
- After any future retrain that clearly surpasses the HF backup, push the merged checkpoint back to `muthugsubramanian/DocWain-14B-v2` (or a new version tag) so the off-site copy stays current.
