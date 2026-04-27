---
name: DocWain V2 Unified Model
description: Vision-grafted Qwen3-14B with native tool-calling — architecture, training phases, and module locations
type: project
---

DocWain V2 grafts SigLIP-SO400M vision encoder onto existing Qwen3-14B (V1) via a trainable projection MLP. The model gains native vision + tool-calling while preserving all V1 knowledge.

**Why:** Single unified model for document intelligence — see, read, reason, and act. No separate OCR model needed.

**How to apply:**
- V2 infrastructure in `src/finetune/v2/` (14 modules, 38 tests)
- 4-phase training: projection pre-train → doc intelligence SFT → tool-calling SFT → merge + promote
- V1 preserved as `DHS/DocWain:v1`, V2 becomes `DHS/DocWain:latest` after gate
- 9 core tools with native `<tool_call>` format, 7 auto-invoked
- Evolving pipeline (`/finetune`) targets V2 after promotion
- GPU upgrade needed for full training (A100 40GB ideal, T4 16GB tight for 14B)
