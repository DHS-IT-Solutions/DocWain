---
name: A100 80GB GPU Migration
description: Prod GPU machine config — A100-SXM4-80GB, Ubuntu, training stack installed
type: project
---

Prod DocWain GPU machine at `/home/ubuntu/PycharmProjects/DocWain`:
- GPU: NVIDIA A100-SXM4-80GB, Driver 550.144, CUDA 12.4
- RAM: 113GB, Disk: 424GB free on /dev/vda1
- Python: 3.12.9 in .venv
- Training stack: PyTorch 2.10+cu128, Unsloth 2026.4.1, TRL 0.24.0, Transformers 5.5.0
- Ollama: installed at /usr/local/bin/ollama
- Models: DHS/DocWain:latest (9.3GB), DHS/DocWain:v1 (frozen backup)
- Git: user.email=muthu.subramanian@dhsit.co.uk, user.name=Muthu Subramanian G
- HF: logged in as Muthu88

**Why:** New prod machine replacing old dev setup. All training happens here.

**How to apply:** Use `unsloth/Qwen3-14B-bnb-4bit` as base (fits A100). No CUDA_HOME needed — Unsloth handles it.
