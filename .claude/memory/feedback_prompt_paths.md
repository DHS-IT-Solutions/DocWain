---
name: LLM Prompt Code Paths
description: The actual code path for LLM prompts is generation/prompts.py via Reasoner — NOT intelligence/generator.py
type: feedback
---

The main query response path is:
  core_agent.py → Reasoner (generation/reasoner.py) → build_system_prompt + build_reason_prompt (generation/prompts.py)

NOT:
  intelligence/generator.py (IntelligentGenerator) — this is a secondary/unused path

**Why:** Spent hours putting formatting fixes in the wrong file. The LLM never saw them.

**How to apply:** Any change to response formatting, system prompt, or task instructions MUST go in `src/generation/prompts.py`. Check `_SYSTEM_PROMPT`, `TASK_FORMATS`, and `_OUTPUT_FORMATS` in that file.
