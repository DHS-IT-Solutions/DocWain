"""DocWain V5 training pipeline — dual 14B + 7B build.

See docs/superpowers/specs/2026-04-18-docwain-v5-design.md for the
full design. Entry points:
    capability_charter  — machine-readable capability → gate contract
    teacher_ensemble    — 5-voice teacher voting for corpus generation
    data_generator      — 100K SFT + 20K DPO row producer
    sft_trainer         — LoRA SFT loop
    dpo_trainer         — preference DPO loop
    distillation        — 14B-teacher → 7B-student KL+SFT
    evaluate            — gate runner (LLM-judge + golden-query eval)
"""
