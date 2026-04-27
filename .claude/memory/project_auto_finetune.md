---
name: Auto Fine-Tune Daily Schedule
description: Daily auto fine-tune loop — feedback drives model improvement, push to MuthuSubramanian/DocWain on Ollama
type: project
---

Daily scheduled fine-tune pipeline triggered by feedback loop data.
Model: MuthuSubramanian/DocWain (base: qwen3:8b on Ollama registry)

Focus areas:
1. Response structure quality (markdown headers, tables, bold values)
2. Context understanding intelligence (cross-document reasoning)
3. Evidence grounding accuracy

**Why:** Incremental improvements didn't work over a month. Continuous fine-tuning from real user feedback is the path to enterprise-grade intelligence.

**How to apply:** The fine-tune pipeline collects feedback signals from Redis (low confidence queries, grounding failures, task type distribution), generates training pairs, runs Unsloth LoRA training, and pushes the updated model to Ollama registry daily.
