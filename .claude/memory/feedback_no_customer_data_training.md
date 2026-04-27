---
name: No customer data in finetuning
description: Training data must be purely synthetic - no customer documents, only patterns and metadata
type: feedback
---

Never use customer data for finetuning. Only patterns and metadata are allowed.

**Why:** Privacy/compliance boundary - customer documents in Azure Blob/Qdrant are for inference only, not training.

**How to apply:** All training data must be synthetically generated. Can use document structure patterns and metadata schemas as inspiration, but never actual customer content.
