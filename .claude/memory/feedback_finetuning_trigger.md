---
name: Finetuning Pipeline Trigger
description: When user says "initiate finetuning pipeline", start the iterative fine-tuning loop for the given requirement
type: feedback
---

When the prompt contains "initiate finetuning pipeline", immediately begin the recursive fine-tuning process for whatever requirement is specified.

**Why:** User wants a shorthand command to trigger the full SFT → DPO → evaluate → retrain loop without re-explaining the process each time.

**How to apply:** On seeing "initiate finetuning pipeline" in the prompt:
1. Identify the target behavior/requirement from the rest of the prompt
2. Generate synthetic training data (SFT + DPO) for the requirement
3. Run the recursive loop: `python -m src.finetune.viz_finetune_loop` (or equivalent for the requirement)
4. The loop generates fresh data each iteration, trains SFT then DPO, evaluates, and retrains until the threshold is met or max iterations reached
5. Prod GPU machine: A100-SXM4-80GB at `/home/ubuntu/PycharmProjects/DocWain`
6. Base model: `unsloth/Qwen3-14B-bnb-4bit` on A100; `unsloth/Qwen3-4B-bnb-4bit` for T4 fallback
