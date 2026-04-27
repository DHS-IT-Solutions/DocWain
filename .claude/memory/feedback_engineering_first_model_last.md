---
name: Engineering First, Model Training Only After Pattern Capture
description: Sequencing rule for DocWain intelligence upgrades — prove behavior changes at the engineering layer (prompts, retrieval, reasoning stages, KG traversal, evaluation) first, instrument to capture patterns from real behavior, and only invest in model retraining after engineering has demonstrably moved metrics and surfaced stable, recurring patterns the weights should encode.
type: feedback
originSessionId: 56b70947-9824-48b4-9a97-b3d2d50b0d88
---
When the user asks for DocWain intelligence improvements (SME behavior, reasoning, cross-document synthesis, insights, URL crawling, etc.), default to engineering-layer changes. Only propose training-data generation, SFT, DPO, or fine-tuning after engineering-layer work has (a) moved the relevant metrics and (b) captured patterns that justify teaching them to the weights.

**Why:** Stated by the user on 2026-04-20 while scoping the Profile-SME reasoning redesign: "this is not about model improvement, this is about engineering first and then capture patterns and only perform model improvement when there is a high confidence." This aligns with the V5 training failure (2026-04-20), where training work advanced without gated evidence. Engineering changes are reversible and cheap to measure; training is expensive and hard to unwind.

**How to apply:**
- When proposing options for intelligence work, lead with prompt/retrieval/reasoning-stage/evaluation changes; treat retraining as a later option contingent on evidence, never the first move.
- Include an instrumentation/pattern-capture step in any intelligence redesign so that future training decisions are data-driven rather than assumption-driven.
- If the user explicitly asks to start from model training, verify they have the engineering-layer signal first — otherwise flag the sequence risk and reference this rule.
- Edge case: bug fixes to the training pipeline itself are fine without engineering-first gating (they're not intelligence upgrades).
