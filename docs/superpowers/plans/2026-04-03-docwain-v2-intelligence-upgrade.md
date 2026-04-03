# DocWain V2 Intelligence Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fully autonomous iterative training pipeline that upgrades DHS/DocWain from V1 to V2 across 6 capability tracks (Excel/CSV, Layout, OCR/Vision, Context/Reasoning, KG-Augmented, Visualization), using programmatic evaluation and strategy evolution with no user intervention.

**Architecture:** A Python orchestrator (`autonomous_trainer.py`) drives 6 sequential tracks. Each track runs an iterative loop: generate synthetic data → train LoRA with Unsloth → evaluate model outputs programmatically → analyze weaknesses → generate targeted data → retrain until gates pass. Evaluation uses rubric-based scoring against structured test prompts via Ollama inference. The V1 model is preserved; V2 is promoted to `latest` only after all gates pass and regression confirms no capability loss.

**Tech Stack:** Unsloth + QLoRA on A100-80GB, Ollama for inference/evaluation, PyTorch 2.x, TRL (SFTTrainer/DPOTrainer), datasets, bitsandbytes, CUDA 12.4

**Spec:** `docs/superpowers/specs/2026-04-03-docwain-v2-intelligence-upgrade-design.md`

---

### Task 1: Environment Setup — Install Training Stack

**Files:**
- Create: `scripts/setup_training_env.sh`

- [ ] **Step 1: Install CUDA toolkit and training dependencies**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing CUDA Toolkit 12.4 ==="
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -qq
sudo apt-get install -y cuda-toolkit-12-4
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

echo "=== Installing Python training dependencies ==="
cd /home/ubuntu/PycharmProjects/DocWain
pip install --upgrade pip
pip install "unsloth[cu124] @ git+https://github.com/unslothai/unsloth.git"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets trl peft bitsandbytes accelerate
pip install openpyxl xlrd  # Excel support
pip install sentencepiece protobuf

echo "=== Verifying GPU access ==="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"
python3 -c "from unsloth import FastLanguageModel; print('Unsloth OK')"

echo "=== Environment ready ==="
```

- [ ] **Step 2: Run the setup script**

Run: `bash scripts/setup_training_env.sh`
Expected: All dependencies install, GPU verification passes.

- [ ] **Step 3: Commit**

```bash
git add -f scripts/setup_training_env.sh
git commit -m "infra: add training environment setup script for A100"
```

---

### Task 2: Preserve V1 Model

**Files:**
- None (Ollama CLI operations only)

- [ ] **Step 1: Tag current model as V1**

```bash
ollama cp DHS/DocWain:latest DHS/DocWain:v1
ollama list | grep DocWain
```

Expected: Both `DHS/DocWain:latest` and `DHS/DocWain:v1` listed.

- [ ] **Step 2: Export V1 Modelfile for backup**

```bash
mkdir -p /home/ubuntu/PycharmProjects/DocWain/models/v1_backup
ollama show DHS/DocWain:v1 --modelfile > models/v1_backup/Modelfile.v1
```

---

### Task 3: Data Generator — Base Infrastructure Update

**Files:**
- Modify: `src/finetune/v2/data_generator/base.py`

- [ ] **Step 1: Extend base.py with chart_spec schema, spreadsheet format, and KG context format**

Add constants for the new output format (`CHART_SPEC_SCHEMA`, `SPREADSHEET_CONTEXT_TEMPLATE`, `KG_CONTEXT_TEMPLATE`) and a `format_sft_with_chart` helper that produces examples with `<response>` + `<chart_spec>` sections. Add `format_spreadsheet_context()` and `format_kg_context()` helpers.

- [ ] **Step 2: Commit**

```bash
git add src/finetune/v2/data_generator/base.py
git commit -m "feat: extend base data generator with chart_spec, spreadsheet, and KG formats"
```

---

### Task 4: Track 1 Data Generator — Excel/CSV Intelligence

**Files:**
- Create: `src/finetune/v2/data_generator/track1_excel_csv.py`

- [ ] **Step 1: Write data generator**

Generate 2.5K examples covering:
- Single-sheet tabular QA (400)
- Multi-sheet reasoning (350)
- Formula-aware understanding (300)
- Merged cell & named range handling (250)
- CSV delimiter & encoding detection (200)
- Large spreadsheet chunking (200)
- Data type inference (250)
- Spreadsheet-to-insight (300)
- Negatives & edge cases (250)

Each example uses the `<spreadsheet>` input format and produces SFT examples with reasoning chains.

- [ ] **Step 2: Commit**

```bash
git add src/finetune/v2/data_generator/track1_excel_csv.py
git commit -m "feat: add Track 1 Excel/CSV intelligence data generator (2.5K examples)"
```

---

### Task 5: Track 2 Data Generator — Layout Intelligence

**Files:**
- Create: `src/finetune/v2/data_generator/track2_layout.py`

- [ ] **Step 1: Write data generator**

Generate 2.5K examples covering nested tables, merged cells, multi-column, mixed form+prose, page-spanning structures, hierarchical headings, document type adaptation, completeness verification, and edge cases. Each example includes a completeness check comparing extracted fields against expected schema. DPO pairs target silent omission as rejected behavior.

- [ ] **Step 2: Commit**

```bash
git add src/finetune/v2/data_generator/track2_layout.py
git commit -m "feat: add Track 2 layout intelligence data generator (2.5K examples)"
```

---

### Task 6: Track 3 Data Generator — OCR & Vision

**Files:**
- Create: `src/finetune/v2/data_generator/track3_ocr_vision.py`

- [ ] **Step 1: Write data generator**

Generate 2.5K examples covering printed text (clean+degraded), handwritten text (block+cursive), mixed print+handwriting, diagram understanding, chart-in-image extraction, table-in-image reconstruction, stamps/watermarks, and caption/label extraction. Each OCR example includes confidence scoring per region.

- [ ] **Step 2: Commit**

```bash
git add src/finetune/v2/data_generator/track3_ocr_vision.py
git commit -m "feat: add Track 3 OCR & vision intelligence data generator (2.5K examples)"
```

---

### Task 7: Track 4 Data Generator — Context & Reasoning

**Files:**
- Create: `src/finetune/v2/data_generator/track4_reasoning.py`

- [ ] **Step 1: Write data generator**

Generate 2K examples covering multi-doc contradiction resolution, temporal reasoning, implicit intent decomposition, causal chain reasoning, quantitative reasoning, counterfactual analysis, uncertainty calibration, and refusal with explanation. DPO pairs target shallow summarization as rejected behavior vs deep reasoning as chosen.

- [ ] **Step 2: Commit**

```bash
git add src/finetune/v2/data_generator/track4_reasoning.py
git commit -m "feat: add Track 4 context & reasoning data generator (2K examples)"
```

---

### Task 8: Track 5 Data Generator — KG-Augmented Knowledge

**Files:**
- Create: `src/finetune/v2/data_generator/track5_kg.py`

- [ ] **Step 1: Write data generator**

Generate 2K examples covering entity-aware answering, relationship traversal, cross-doc entity linking, KG-grounded fact checking, missing relationship detection, ontology-aware reasoning, and KG context format training. All examples include `<kg_context>` blocks in the input.

- [ ] **Step 2: Commit**

```bash
git add src/finetune/v2/data_generator/track5_kg.py
git commit -m "feat: add Track 5 KG-augmented knowledge data generator (2K examples)"
```

---

### Task 9: Track 6 Data Generator — Visualization Intelligence

**Files:**
- Create: `src/finetune/v2/data_generator/track6_visualization.py`

- [ ] **Step 1: Write data generator**

Generate 2K examples covering single-series charts, multi-series comparison, auto-detect triggers, explicit request handling, no-chart negatives (400 — critical for judgment), annotation intelligence, and chart type selection reasoning. Output format uses `<response>` + `<chart_spec>` JSON blocks.

- [ ] **Step 2: Commit**

```bash
git add src/finetune/v2/data_generator/track6_visualization.py
git commit -m "feat: add Track 6 visualization intelligence data generator (2K examples)"
```

---

### Task 10: Evaluation Infrastructure

**Files:**
- Create: `src/finetune/v2/eval/rubrics.py`
- Create: `src/finetune/v2/eval/evaluator.py`
- Create: `src/finetune/v2/eval/test_bank.py`
- Create: `src/finetune/v2/eval/__init__.py`

- [ ] **Step 1: Write rubrics.py — scoring dimensions per track**

Define programmatic scoring functions for each track:
- Track 1: `score_tabular_qa()` — checks data accuracy, aggregation correctness, type inference
- Track 2: `score_layout()` — structure F1, completeness ratio, relationship extraction
- Track 3: `score_ocr()` — character accuracy, diagram description quality, handwriting recognition
- Track 4: `score_reasoning()` — reasoning depth (step count), evidence grounding, synthesis coherence
- Track 5: `score_kg()` — entity usage rate, relationship traversal accuracy, citation correctness
- Track 6: `score_visualization()` — trigger precision/recall, JSON validity, data accuracy, type selection

Each returns a dict of dimension scores (1-5 scale).

- [ ] **Step 2: Write test_bank.py — frozen eval examples per track**

50 evaluation prompts per track (300 total) with expected outputs, covering the full range of each track's categories. These are NEVER used in training.

- [ ] **Step 3: Write evaluator.py — orchestrates evaluation via Ollama**

`evaluate_track(track_name, model_name)` → queries the model via Ollama HTTP API, scores responses with rubrics, returns aggregate scores per dimension and pass/fail verdict.

- [ ] **Step 4: Commit**

```bash
git add src/finetune/v2/eval/
git commit -m "feat: add evaluation infrastructure — rubrics, test bank, evaluator"
```

---

### Task 11: Strategy Evolver

**Files:**
- Create: `src/finetune/v2/strategy_evolver.py`

- [ ] **Step 1: Write strategy evolver**

`StrategyEvolver` class that:
- Receives iteration history (scores per dimension across rounds)
- Identifies persistent failure patterns (dimensions that stay below threshold)
- Recommends strategy changes:
  - Increase data for weak categories
  - Add DPO pairs from actual model failures
  - Adjust hyperparameters (LR, LoRA rank, epochs)
  - Restructure curriculum order
- Returns a `StrategyUpdate` with adjusted data generation params and training config

- [ ] **Step 2: Commit**

```bash
git add src/finetune/v2/strategy_evolver.py
git commit -m "feat: add strategy evolver for iterative training loop"
```

---

### Task 12: Track Training Script

**Files:**
- Create: `src/finetune/v2/train_track.py`

- [ ] **Step 1: Write unified track training script**

Single training script that handles any track:
- Loads base model (Qwen3-14B via Unsloth) or previous track's checkpoint
- Applies LoRA (r=64, alpha=128, target all projection modules)
- Trains SFT on track-specific JSONL data
- Optionally trains DPO on preference pairs
- Saves merged checkpoint + GGUF Q4_K_M
- Updates Ollama model for evaluation

Key function: `train_track(track_name, data_path, base_checkpoint, config)` → returns checkpoint path.

- [ ] **Step 2: Commit**

```bash
git add src/finetune/v2/train_track.py
git commit -m "feat: add unified track training script with LoRA + GGUF export"
```

---

### Task 13: Autonomous Training Orchestrator

**Files:**
- Create: `src/finetune/v2/autonomous_trainer.py`

- [ ] **Step 1: Write the main orchestrator**

`AutonomousTrainer` class that runs the entire pipeline end-to-end:

```python
class AutonomousTrainer:
    TRACKS = ["excel_csv", "layout", "ocr_vision", "reasoning", "kg", "visualization"]

    def run(self):
        self.preserve_v1()
        checkpoint = None
        for track in self.TRACKS:
            checkpoint = self.run_track(track, base_checkpoint=checkpoint)
        self.run_cross_track_eval()
        self.run_regression_vs_v1()
        self.merge_and_promote()

    def run_track(self, track, base_checkpoint):
        """Iterative loop for a single track — no iteration cap."""
        iteration = 0
        strategy = default_strategy(track)
        while True:
            iteration += 1
            data = self.generate_data(track, strategy)
            checkpoint = self.train(track, data, base_checkpoint, strategy)
            scores = self.evaluate(track)
            if self.gate_passed(track, scores):
                break
            strategy = self.evolver.evolve(track, iteration, scores, strategy)
        return checkpoint
```

State persistence: saves progress to `finetune_artifacts/v2_upgrade/state.json` after each iteration so it can resume if interrupted.

Logging: writes detailed logs to `finetune_artifacts/v2_upgrade/training.log`.

- [ ] **Step 2: Add CLI entry point**

```python
if __name__ == "__main__":
    trainer = AutonomousTrainer()
    trainer.run()
```

Run: `nohup python -m src.finetune.v2.autonomous_trainer > training_output.log 2>&1 &`

- [ ] **Step 3: Commit**

```bash
git add src/finetune/v2/autonomous_trainer.py
git commit -m "feat: add autonomous training orchestrator — 6-track iterative pipeline"
```

---

### Task 14: Merge, Promote & Launch

**Files:**
- Modify: `src/finetune/v2/merge_promote.py`

- [ ] **Step 1: Update merge_promote.py with real merge/quantize/promote logic**

Replace the "In production:" stubs with actual Unsloth merge + GGUF export + Ollama create commands. Add regression test runner that queries V1 and V2 with the same prompts and compares scores.

- [ ] **Step 2: Launch the autonomous pipeline**

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
cd /home/ubuntu/PycharmProjects/DocWain
nohup python -m src.finetune.v2.autonomous_trainer > training_output.log 2>&1 &
echo "Pipeline launched. PID: $!"
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: launch DocWain V2 autonomous training pipeline"
```
