# DocWain as an In-House Base Model — Research & Practical Roadmap

**Date**: 2026-04-17
**Status**: Research complete, plan ready for approval
**Goal**: Make DocWain a true base model whose document intelligence, identity, and behaviour live in the weights — no system-prompt crutch at inference.

---

## What "base model" really has to mean for DocWain

A model called "base" only matters if it behaves this way end-to-end:

1. **Identity is implicit.** No `"You are DocWain, a document intelligence assistant..."` preamble. The model already knows.
2. **Grounded extraction is a default.** When given a document the model returns structured output because that's what every training example did, not because a system prompt demanded it.
3. **Tool-calling is native.** `<tool_call>` syntax is a tokenizer-level pattern, learned through the same data path as ordinary text.
4. **Behaviour contracts are weights.** "Never fabricate numbers", "cite spans", "use the V2 schema" — each survives a blank system prompt.

Every one of those is missing today. The 200+ line system prompt is where identity + contracts live, and the model ignores pieces of it under load. That's the gap.

---

## Current assets (what we don't have to rebuild)

| Asset | Where | Value |
|---|---|---|
| V3 Qwen3-14B + vision graft | `models/DocWain-14B-v2/`, served on vLLM:8100 | Strong base — 4.71/5.0 gate, SFT loss 0.127 |
| 31K multi-task SFT corpus | `finetune_artifacts/teacher_data/master_v4.jsonl` | Extraction + reasoning + tables + legal — ready |
| 4.3K DPO preference pairs | `master_dpo_v4.jsonl` | Can teach behaviour preferences |
| Autonomous training loop | `scripts/weekend_finetune_loop.py` + `src/finetune/v2/auto_curriculum.py` | SFT → DPO → eval → retrain automation |
| Distillation tooling | `src/finetune/distillation/{generators,advanced_generators,document_element_generators}.py` | Produces more data on demand |
| Eval harness | `src/finetune/evaluation/llm_judge.py` + `scripts/evaluate_docwain.py` | LLM-judge scoring, real-doc testing |
| EAGLE3 speculative decoding | `systemd/docwain-vllm-fast.service` | Serving throughput headroom |

The infra to *train* is built. What's missing is the *recipe* that produces a base-model-quality result.

---

## Candidate techniques — honest assessment

### 1. Continued Pretraining (CPT) on document corpora

**What**: Resume Qwen3-14B pretraining on billions of raw document tokens (SEC filings, EDGAR, PubMed PDFs, invoice datasets).
**Pros**: Deeply bakes document-world priors into the model.
**Cons**: Needs **1–10B tokens** for measurable effect. At Qwen3-14B scale that's 40–400 A100-hours just for the token-pass. No large public corpus of *structured-doc-with-labels* exists — we'd spend more on data curation than training.
**Fit**: **Not recommended as primary lever.** The marginal gain per dollar is worse than option 3, and our existing V3 already has strong document priors from its SFT.

### 2. Massive instruction SFT at scale (100K–1M examples)

**What**: Scale the weekend SFT approach from 31K to 300K–1M examples, keeping the distillation generators as the source.
**Pros**: Directly moves the behaviour we care about into weights. Fully compatible with existing infra. Every example can omit the system prompt so identity is implicit.
**Cons**: Labour: need to generate + quality-gate ~10× more data. Throughput: at A100 80GB + LoRA rank-64, a full pass on 300K examples is ~36 hours.
**Fit**: **Primary lever.** This is how we put behaviour into weights at our scale. Already 60% built.

### 3. DPO / preference optimization

**What**: Generate chosen/rejected pairs and run DPO so the model learns the behaviour contracts as *preferences*, not rules.
**Pros**: Far more token-efficient than SFT for *behaviour* (citation style, refusal-when-ungrounded, schema adherence). V3 already runs DPO on 3.8K pairs with loss 0.096 — working.
**Cons**: Needs high-quality preference data. The 4.3K V4 DPO pairs are a floor, not ceiling — probably need 10–20K for a true behavioural shift.
**Fit**: **Co-primary lever alongside SFT.** DPO teaches *style/taste*; SFT teaches *knowledge + format*.

### 4. Model merging / souping (**the user's specific question**)

**What**: Take several Qwen3-14B-based models trained for complementary skills and merge their weights into a single 14B model. No GPU training needed — just weight arithmetic.
**How**: [MergeKit](https://github.com/arcee-ai/mergekit) supports several algorithms:
- **SLERP** (spherical linear interpolation): 2-way blend
- **Linear / Task Arithmetic**: add/subtract deltas from a shared base
- **TIES-Merging**: resolve sign conflicts by keeping the dominant direction per parameter
- **DARE** (Drop And REscale): randomly zero 70–90% of delta params, rescale the rest; remarkably effective when combining several specialised models
- **Breadcrumbs**: sparse merging that keeps only salient deltas

**Pros**:
- **No training cost** — runs on CPU in 30–90 min for 14B
- When source models are complementary (reasoning + instruction + domain), the merge often beats any single source on a mixed evaluation. This is well-documented in the [MergeKit paper](https://arxiv.org/abs/2403.13257) and confirmed by public leaderboards.
- Reproducible and auditable — a merge recipe is a YAML file checked into the repo.

**Cons**:
- Requires **shared architecture + tokenizer**. All source models must be Qwen3-14B with identical layer counts/widths. This rules out merging a Llama model in.
- Merging **can't invent** capabilities the source models don't have. It re-weights what exists.
- Public Qwen3-14B fine-tunes are fewer than for Llama/Mistral — the ecosystem is thinner.

**Candidate merge for DocWain** (all Qwen3-14B-compatible):
| Source | Contributes | Weight |
|---|---|---|
| Qwen3-14B-Base | Broad factual knowledge, fluent language | 0.25 |
| Qwen3-14B-Instruct | Instruction following, chat format | 0.25 |
| DocWain-V3 (our own) | Document extraction, identity priors | 0.35 |
| Qwen3-14B-Coder (HF: `Qwen/Qwen3-14B-Coder` if available, else our code-tuned variant) | Tool-call / structured output syntax | 0.10 |
| Qwen3-14B math or reasoning variant (`Qwen/Qwen3-14B-Math` or similar) | Arithmetic accuracy on invoice totals | 0.05 |

Use TIES-Merging with density 0.7 and our own V3 as the "base" in task-arithmetic mode, so DocWain's document behaviour dominates while the others provide complementary strengths.

**Fit**: **Cheap leveraged multiplier.** Run it once before the next SFT to give training a better starting point. Not a replacement for SFT+DPO — but it raises the floor for both.

### 5. Knowledge distillation from frontier models

**What**: Query Claude 4 / GPT-4 / Gemini 2 with our document tasks, record their outputs, and SFT DocWain against those.
**Pros**: Pulls frontier reasoning into our 14B. Already proven in our V3→V4 jump (SFT loss 1.034 → 0.127). The distillation tooling exists.
**Cons**: API cost (large dataset = real money). License review on using proprietary outputs as training data — **Anthropic's TOS forbids training a competing LLM on Claude outputs; OpenAI's is stricter. For an in-house enterprise model used internally, read the TOS with legal before scaling this.** Qwen's distillation license is permissive.
**Fit**: **Complementary, with the licence caveat.** Already in the V3 pipeline. Scale with care; lean more on synthetic data generated by Qwen/DocWain self-play.

### 6. LoRA / adapter stacking

**What**: Keep base frozen, train separate LoRA adapters per capability (extraction-LoRA, reasoning-LoRA, identity-LoRA), load the right set per query.
**Pros**: Modular, cheap to train, easy to A/B.
**Cons**: Runtime switching has latency; adapters don't compose cleanly (interference is a well-known problem). Creates operational complexity we don't need if the goal is "one unified model".
**Fit**: **Not recommended as core strategy.** The user explicitly wants a *unified* model. LoRA during training fine, but the final artifact should be merged to a single dense model.

### 7. Retrieval-augmented training (RAT)

**What**: Train examples include retrieved chunks in the context, so the model *learns to use* retrieval rather than treating it as external.
**Pros**: Aligns training and inference — what the model sees in training matches what it sees in prod.
**Cons**: Requires a stable retrieval pipeline for training. We *now* have that (the 7-point refactor), so this is newly feasible.
**Fit**: **Yes, as a V5 addition.** A fraction of the SFT corpus should include retrieved chunks. The instruction format is: "Given these documents from the KG: [chunks] — answer...".

### 8. Mixture of Experts (MoE)

**What**: Replace dense Qwen3-14B with a MoE architecture (8 experts × 2B each, 2 active per token).
**Pros**: Specialised experts per doc type.
**Cons**: MoE is an **architecture change**, not a training change. Would require rebuilding vLLM serving, Ollama support, EAGLE3 drafting. Multi-month effort and Qwen3 doesn't have a public MoE variant of the same scale. 
**Fit**: **Out of scope** for this sprint. Revisit after V5 if evaluation plateaus.

---

## Is weight cherry-picking feasible? Direct answer

**Yes, with qualifications:**

✅ **Feasible now**:
- MergeKit handles Qwen3-14B merges on CPU in under 90 minutes
- Our V3 + public Qwen3 fine-tunes share architecture + tokenizer
- The operation is deterministic and reproducible — the merge YAML becomes a build artifact
- We can evaluate each merge candidate on our existing eval harness before promoting

⚠️ **Limits to be honest about**:
- The pool of **high-quality Qwen3-14B fine-tunes** is smaller than Llama's. Most of the "ecosystem weight souping" magic people see on leaderboards happens in Llama/Mistral land
- Merging **won't give us capabilities nobody in the pool has** — e.g., no public Qwen3-14B excels at stamped-PDF OCR, so we can't merge that in
- "Cherry-picking best weights" is a misleading mental model. We're not picking *individual weights* per se — we're blending whole models with per-tensor weighting schemes
- Merges **can hurt** if sources disagree strongly. A merge that includes a chat-heavily-safety-tuned variant can reduce our extraction confidence — needs evaluation before promoting

❌ **Not feasible**:
- Picking weights from **different architectures** (e.g. a Llama reasoning model into Qwen3) — architecture-level merges are an active research area, nothing battle-tested
- Picking **individual neurons** as "best" across models — the interpretability to do that at 14B scale isn't there yet for us

**Conclusion on cherry-picking**: it's one lever among several, and the cheapest of all of them. Use it as a *starting point* for training, not a substitute for training.

---

## Practical roadmap — what to actually do

A 4-phase plan, each phase yielding a measurable improvement, each building on the prior V3 checkpoint.

### Phase A — Baseline + Merge-Start (1 day)

**Goal**: Lift the starting point before we spend GPU hours.

1. Enumerate merge candidates (confirm availability of Qwen3-14B-Instruct, code-tuned, math-tuned variants on HuggingFace).
2. Write `configs/merge_recipes/docwain_v5_seed.yaml` — a TIES-Merging YAML with our V3 as the base and 3–4 complementary Qwen3-14Bs at the weights above.
3. Run MergeKit on CPU (no training); artifact: `models/DocWain-14B-v5-seed/`.
4. Score the merged seed on the existing LLM-judge eval suite. Gate: must not regress V3's 4.71/5.0.
5. Commit the recipe + benchmark result.

**Cost**: ~2 hours compute, mostly CPU. No training yet.
**Risk**: low — if merge regresses, we keep V3 as the SFT base.

### Phase B — Identity & Behaviour SFT at scale (2 days GPU)

**Goal**: Put identity, grounding, citation, refusal-when-ungrounded into weights. No system prompt in training data.

1. Regenerate the V4 master SFT corpus (31K) with:
   - Every example's system field set to empty (or a consistent short "Respond as DocWain." token sequence — nothing more).
   - Identity patterns distributed across examples (some have "Who are you?" → DocWain self-description; most just do their task with DocWain's voice).
   - Tool-use examples with `<tool_call>` tags at tokenizer level, not prose-style.
2. Generate an additional 30–50K examples through `src/finetune/distillation/advanced_generators.py` + document-element generators to reach **80K**.
3. Train SFT from the merge seed (Phase A) on the 80K corpus: 2 epochs, LoRA rank 128, A100 80GB.
4. Merge LoRA → dense. Artifact: `models/DocWain-14B-v5-sft/`.
5. Eval gate: identity (blank system prompt still answers "I'm DocWain") + extraction recall ≥ V3.

**Cost**: ~24 A100-hours (one weekend run).
**Risk**: medium. Mitigated by the eval gate.

### Phase C — Preference DPO for behaviour contracts (1 day GPU)

**Goal**: Teach taste — when to refuse, when to cite, when to just answer briefly.

1. Expand DPO pairs from 4.3K → 15K. Categories:
   - **Grounded vs hallucinated** (chosen cites, rejected invents)
   - **Schema-valid vs drifting** (chosen follows V2 schema, rejected breaks it)
   - **Identity-stable vs identity-leaking** (chosen says "I'm DocWain", rejected says "I'm an AI assistant")
   - **Concise vs padded** (chosen gives the answer, rejected gives a lecture)
2. Run DPO from the Phase-B checkpoint. β=0.1, 2 epochs.
3. Artifact: `models/DocWain-14B-v5/`.
4. Eval gate: behaviour contracts pass + no regression on extraction/reasoning.

**Cost**: ~12 A100-hours.
**Risk**: medium. DPO can degrade capability if preference pairs are noisy — gate tightly.

### Phase D — Retrieval-augmented SFT & Self-Play (V6 — later sprint)

Once V5 is in production and we have feedback-loop data:
1. Subset the SFT corpus where retrieved chunks are **inlined** into the instruction — trains the model to read retrieved context.
2. Self-play: DocWain answers a query, an LLM judge ranks the answer, the ranked pair becomes a new DPO example. Loop automates.
3. This is where DocWain gets unique — it learns **from its own production traffic** (with PII masking), becoming a model whose behaviour encodes DocWain's specific document fleet.

**Cost**: ongoing, amortised. Worth scoping after V5 ships.

### What gets SHIPPED at the end of Phase C — V5

A 14B dense model that:
- Identifies as DocWain without prompting
- Returns V2-schema-structured output as default when handed a document
- Does native tool-calls with `<tool_call>` (inherited from V2 graft)
- Cites spans, refuses to fabricate, uses contextualised-retrieval output cleanly
- Serves via the same vLLM instance (no serving change)
- Beats V3 on the LLM-judge eval, R@5/MRR on the golden query set, and blank-system-prompt identity tests

---

## What makes this result *one-of-a-kind*

The user asked for a unique model, not a model souped up from existing recipes. What makes V5 unique:

1. **DocWain's own V3 is the dominant merge source** (weight 0.35). No other public 14B has our document priors.
2. **The SFT corpus is 80K examples generated from our own Qdrant/Neo4j corpus + frontier distillation** — no one else has access to the same data mix.
3. **DPO preferences encode DocWain's exact behaviour contracts** (grounded citations, V2 schema, DocWain voice). Those aren't in any public checkpoint.
4. **Vision-graft is preserved across the pipeline** — still sees document images via SigLIP-SO400M. Most open 14Bs are text-only.
5. **Identity in weights, not prompts** — a blank system prompt still produces DocWain behaviour. Every public Qwen3 fine-tune requires a system prompt.

The model is *derivative* of Qwen3-14B the same way every good fine-tune is derivative. What makes it one-of-a-kind is the *combination* of merge + our data + our DPO + our vision graft, and the fact that identity is intrinsic.

---

## What we're NOT doing (and why)

- **Not training from scratch.** 14B from scratch is 10^22+ FLOPs. Not practical on one A100.
- **Not architectural changes (MoE, new attention).** Out of scope, breaks serving.
- **Not Llama/Mistral cross-merges.** Architecture mismatch.
- **Not massive CPT.** Marginal utility doesn't justify the GPU hours vs. scaled SFT.
- **Not frontier-model fine-tuning.** TOS concerns and we don't have their weights anyway.

---

## Decision requested

Shall I proceed with Phase A (merge recipe + seed model) this week, then hand off Phases B/C to the autonomous training loop for the next weekend run? Or do you want adjustments to the merge candidates / SFT scale / DPO categories before I start?

If approved as-is, I'll:
1. Write `configs/merge_recipes/docwain_v5_seed.yaml`
2. Run MergeKit and produce `models/DocWain-14B-v5-seed/`
3. Score it on the existing eval suite
4. Report numbers — then you decide whether to commit to Phases B/C.
