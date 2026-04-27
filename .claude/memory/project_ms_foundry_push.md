---
name: Push DocWain to Microsoft Foundry
description: After finetuning, push DocWain model to Microsoft Azure AI Foundry as a new model
type: project
---

After the base model finetuning is complete, push DocWain to Microsoft Azure AI Foundry as a new model.

**Why:** Make DocWain available as a hosted model on Azure for enterprise customers.

**How to apply:** After GGUF conversion and Ollama push, also convert to safetensors/ONNX format compatible with Azure AI Foundry and push using Azure CLI or SDK.

**When:** After the current base model training loop completes and extraction tests pass.
