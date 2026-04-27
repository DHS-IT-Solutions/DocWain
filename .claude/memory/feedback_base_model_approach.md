---
name: DocWain Base Model Approach
description: User wants DocWain converted to a base model with identity baked into weights, no system prompt dependency
type: feedback
---

DocWain must be trained as a base model — its identity, behavior, and document intelligence should be baked into the model weights, NOT passed as a system prompt at inference time.

**Why:** The user wants higher accuracy and intelligence. The current 200+ line system prompt wastes context window, adds latency, and the model sometimes ignores parts of it. A base model approach makes DocWain intrinsically intelligent.

**How to apply:** 
- Training data should have minimal/empty system fields
- The model should respond as DocWain by default without being told
- All behavioral rules (grounding, formatting, identity) must be learned from examples, not prompted
- The Ollama Modelfile and vLLM config should use empty or minimal system prompts
- This is the approach for the next weekend training run (Apr 19-20, 2026)
