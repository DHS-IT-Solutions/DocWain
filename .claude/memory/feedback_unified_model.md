---
name: Unified DocWain Model (No Fast/Smart Split)
description: DocWain is one unified model — no separate fast/smart vLLM instances or routing. Always promote the best-scoring checkpoint to active.
type: feedback
originSessionId: 9a15d065-2784-44cb-8db5-07a885ea1f7c
---
DocWain must be a single unified model. The fast (14B)/smart (27B) split is rejected. All serving, routing, and prompt paths should assume one served model.

**Why:** User design decision stated 2026-04-17. Simpler ops, consistent quality, no routing logic to maintain, no second 52G model to keep in sync, and the v2 grafted 14B is already intended as "one model to see/read/reason/act" (per project memory on V2).

**How to apply:**
- Do not reintroduce `docwain-vllm-smart.service`, `--served-model-name docwain-fast`, or a `model_router` that picks between fast/smart. Rename the fast service to `docwain-vllm.service` when safe (requires restart).
- When `src/serving/model_router.py`, `src/serving/fast_path.py`, or any fast/smart branching is touched, consolidate to a single path rather than preserving the split.
- The active served weights (`models/docwain-v2-active` symlink) must always point to the best-scoring checkpoint. Any finetune loop that produces a better checkpoint (per eval gate) must repoint the symlink; never downgrade to preserve fast/smart balance.
- When promoting a new best checkpoint, also quantize/convert it into any formats the unified server needs (e.g., AWQ) so there is still one canonical active model.
