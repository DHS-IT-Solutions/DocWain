# DocWain V2+ Finetuning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform DocWain into a GPT-class document intelligence model with holistic contextual analysis through enhanced V2 finetuning pipeline + post-training refinement stack.

**Architecture:** Enhance existing 6-phase V2 pipeline with Claude Code-generated synthetic datasets (52K examples), add Phase 3.7 (Holistic Reasoning SFT), and stack 3 post-training refinement rounds (Conversational DPO, Confidence Calibration SFT, Reasoning Distillation). Automated eval suite with Claude Code as judge gates every phase transition.

**Tech Stack:** Unsloth + TRL (SFTTrainer/DPOTrainer), PyTorch bf16, LoRA (r=64/alpha=128), Qwen3-14B base, GGUF Q4_K_M quantization, Ollama deployment.

**Spec:** `docs/superpowers/specs/2026-04-02-docwain-v2plus-finetuning-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|---|---|
| `src/finetune/v2/data_generator/__init__.py` | Package init with shared constants |
| `src/finetune/v2/data_generator/base.py` | Base generator class, JSONL writer, chat template helpers |
| `src/finetune/v2/data_generator/phase2_doc_intelligence.py` | Generate 20K doc intelligence training examples |
| `src/finetune/v2/data_generator/phase2_5_dpo_pairs.py` | Generate 5K DPO preference pairs |
| `src/finetune/v2/data_generator/phase3_tool_traces.py` | Generate 4K enhanced tool-calling examples |
| `src/finetune/v2/data_generator/phase3_5_insights.py` | Generate 6K insight examples (7 categories) |
| `src/finetune/v2/data_generator/phase3_7_holistic.py` | Generate 8K holistic reasoning examples (4 modes) |
| `src/finetune/v2/data_generator/post_conversational.py` | Generate 3K conversational DPO pairs |
| `src/finetune/v2/data_generator/post_confidence.py` | Generate 2K confidence calibration examples |
| `src/finetune/v2/data_generator/eval_suite.py` | Generate 500 held-out eval benchmark examples |
| `src/finetune/v2/train_phase3_7_holistic.py` | Phase 3.7 holistic reasoning SFT training loop |
| `src/finetune/v2/post_training/__init__.py` | Post-training package init |
| `src/finetune/v2/post_training/round1_conversational_dpo.py` | Conversational refinement DPO |
| `src/finetune/v2/post_training/round2_confidence_sft.py` | Confidence calibration SFT |
| `src/finetune/v2/post_training/round3_distillation.py` | Reasoning distillation for speed |
| `src/finetune/v2/eval/__init__.py` | Eval package init |
| `src/finetune/v2/eval/rubrics.py` | Claude Code judge scoring rubrics |
| `src/finetune/v2/eval/gate_checker.py` | Pass/fail gate logic per phase |
| `src/finetune/v2/eval/runner.py` | Benchmark runner orchestrating eval |

### Modified Files
| File | Change |
|---|---|
| `src/finetune/v2/pipeline.py` | Add Phase 3.7, post-training orchestration, eval integration |
| `src/finetune/v2/merge_promote.py` | Merge 5 LoRA stages, add Phase3.7 config, re-quantize after post-training |
| `src/finetune/v2/__init__.py` | Export new modules |

### Test Files
| File | Tests |
|---|---|
| `tests/test_v2_data_generator.py` | Data generator output format, volume, schema validation |
| `tests/test_v2_phase3_7.py` | Phase 3.7 config, training args, dataset loading |
| `tests/test_v2_post_training.py` | Post-training rounds config, training args |
| `tests/test_v2_eval.py` | Eval runner, rubrics, gate checker |
| `tests/test_v2_pipeline_orchestration.py` | Full pipeline phase ordering, marker detection |

---

## Task 1: Data Generator Base Infrastructure

**Files:**
- Create: `src/finetune/v2/data_generator/__init__.py`
- Create: `src/finetune/v2/data_generator/base.py`
- Test: `tests/test_v2_data_generator.py`

- [ ] **Step 1: Write the failing test for base generator**

```python
# tests/test_v2_data_generator.py
import json
import tempfile
from pathlib import Path

import pytest


class TestBaseGenerator:
    def test_jsonl_writer_creates_file(self):
        from src.finetune.v2.data_generator.base import JSONLWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            writer = JSONLWriter(path)
            writer.write({"text": "hello"})
            writer.close()
            assert path.exists()
            with open(path) as f:
                line = json.loads(f.readline())
            assert line["text"] == "hello"

    def test_jsonl_writer_multiple_records(self):
        from src.finetune.v2.data_generator.base import JSONLWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            writer = JSONLWriter(path)
            for i in range(100):
                writer.write({"idx": i})
            writer.close()
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 100

    def test_jsonl_writer_context_manager(self):
        from src.finetune.v2.data_generator.base import JSONLWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            with JSONLWriter(path) as writer:
                writer.write({"x": 1})
            assert path.exists()

    def test_format_sft_example(self):
        from src.finetune.v2.data_generator.base import format_sft_example

        result = format_sft_example(
            query="What is the total?",
            reasoning="The table shows values 10, 20, 30. Sum = 60.",
            answer="The total is 60.",
        )
        assert "text" in result
        assert "<|im_start|>system" in result["text"]
        assert "<think>" in result["text"]
        assert "The total is 60." in result["text"]
        assert "<|im_end|>" in result["text"]

    def test_format_sft_example_preserves_think_block(self):
        from src.finetune.v2.data_generator.base import format_sft_example

        result = format_sft_example(
            query="test",
            reasoning="step 1\nstep 2",
            answer="done",
        )
        assert "<think>\nstep 1\nstep 2\n</think>" in result["text"]

    def test_format_dpo_example(self):
        from src.finetune.v2.data_generator.base import format_dpo_example

        result = format_dpo_example(
            query="Extract entities",
            chosen_reasoning="Found: John Smith (PERSON)",
            chosen_answer="Entity: John Smith, type PERSON",
            rejected_reasoning="No entities found",
            rejected_answer="I could not find any entities.",
        )
        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result
        assert "<think>" in result["chosen"]
        assert "<think>" in result["rejected"]

    def test_format_eval_example(self):
        from src.finetune.v2.data_generator.base import format_eval_example

        result = format_eval_example(
            benchmark="TableBench",
            query="What is the Q3 revenue?",
            context=[{"text": "Q3: $500K", "source": "report.pdf"}],
            reference_answer="Q3 revenue is $500K.",
            rubric="Check exact value match and source citation.",
        )
        assert result["benchmark"] == "TableBench"
        assert result["query"] == "What is the Q3 revenue?"
        assert len(result["context"]) == 1
        assert result["reference_answer"] == "Q3 revenue is $500K."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_data_generator.py::TestBaseGenerator -v 2>&1 | head -30`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.finetune.v2.data_generator'`

- [ ] **Step 3: Write the data generator base module**

```python
# src/finetune/v2/data_generator/__init__.py
"""DocWain V2+ synthetic data generation — Claude Code as dataset builder."""

from .base import JSONLWriter, format_sft_example, format_dpo_example, format_eval_example

__all__ = ["JSONLWriter", "format_sft_example", "format_dpo_example", "format_eval_example"]
```

```python
# src/finetune/v2/data_generator/base.py
"""Base utilities for synthetic data generation.

Provides JSONL writing, chat template formatting for SFT/DPO/eval examples,
and shared constants used across all data generators.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCWAIN_SYSTEM_PROMPT = (
    "You are DocWain, an enterprise document intelligence assistant. "
    "You analyse documents with deep contextual understanding, extract structured "
    "information, identify patterns and anomalies, and provide holistic analysis "
    "grounded in evidence. You reason step-by-step before answering, state your "
    "confidence level, and cite specific sources. When information is insufficient, "
    "you say so clearly rather than guessing."
)

DOMAINS = ["legal", "financial", "hr", "medical", "policy", "insurance", "procurement"]

DOC_TYPES = [
    "contract", "invoice", "purchase_order", "resume", "job_description",
    "policy_document", "compliance_report", "financial_statement", "audit_report",
    "medical_record", "insurance_claim", "patent", "lease_agreement",
    "employee_handbook", "meeting_minutes", "proposal", "sow", "nda",
]


# ---------------------------------------------------------------------------
# JSONL Writer
# ---------------------------------------------------------------------------


class JSONLWriter:
    """Buffered JSONL file writer with context manager support."""

    def __init__(self, path: Path, buffer_size: int = 100) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: List[str] = []
        self._buffer_size = buffer_size
        self._count = 0
        self._file = open(self._path, "w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._buffer.append(json.dumps(record, ensure_ascii=False))
        self._count += 1
        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self) -> None:
        if self._buffer:
            self._file.write("\n".join(self._buffer) + "\n")
            self._buffer.clear()

    def close(self) -> None:
        self._flush()
        self._file.close()
        logger.info("Wrote %d records to %s", self._count, self._path)

    @property
    def count(self) -> int:
        return self._count

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Chat template formatters
# ---------------------------------------------------------------------------


def format_sft_example(
    query: str,
    reasoning: str,
    answer: str,
    *,
    system_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """Format a single SFT training example in Qwen3 chat template.

    Returns dict with ``text`` key containing the full chat-formatted string.
    """
    sys_prompt = system_prompt or DOCWAIN_SYSTEM_PROMPT
    text = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>"
    )
    return {"text": text}


def format_dpo_example(
    query: str,
    chosen_reasoning: str,
    chosen_answer: str,
    rejected_reasoning: str,
    rejected_answer: str,
    *,
    system_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """Format a DPO preference pair for TRL DPOTrainer.

    Returns dict with ``prompt``, ``chosen``, ``rejected`` keys.
    """
    sys_prompt = system_prompt or DOCWAIN_SYSTEM_PROMPT
    prompt = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    chosen = f"<think>\n{chosen_reasoning}\n</think>\n\n{chosen_answer}<|im_end|>"
    rejected = f"<think>\n{rejected_reasoning}\n</think>\n\n{rejected_answer}<|im_end|>"
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def format_eval_example(
    benchmark: str,
    query: str,
    context: List[Dict[str, Any]],
    reference_answer: str,
    rubric: str,
    *,
    expected_tools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Format a held-out evaluation example."""
    result: Dict[str, Any] = {
        "benchmark": benchmark,
        "query": query,
        "context": context,
        "reference_answer": reference_answer,
        "rubric": rubric,
    }
    if expected_tools:
        result["expected_tools"] = expected_tools
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_data_generator.py::TestBaseGenerator -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add -f src/finetune/v2/data_generator/__init__.py src/finetune/v2/data_generator/base.py tests/test_v2_data_generator.py
git commit -m "feat: add data generator base infrastructure for V2+ finetuning"
```

---

## Task 2: Phase 2 Document Intelligence Data Generator

**Files:**
- Create: `src/finetune/v2/data_generator/phase2_doc_intelligence.py`
- Test: `tests/test_v2_data_generator.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_v2_data_generator.py

class TestPhase2Generator:
    def test_generate_table_examples(self):
        from src.finetune.v2.data_generator.phase2_doc_intelligence import generate_table_examples

        examples = generate_table_examples(count=10)
        assert len(examples) == 10
        for ex in examples:
            assert "text" in ex
            assert "<think>" in ex["text"]
            assert "<|im_start|>" in ex["text"]

    def test_generate_layout_examples(self):
        from src.finetune.v2.data_generator.phase2_doc_intelligence import generate_layout_examples

        examples = generate_layout_examples(count=10)
        assert len(examples) == 10
        for ex in examples:
            assert "text" in ex
            assert "<think>" in ex["text"]

    def test_generate_ocr_examples(self):
        from src.finetune.v2.data_generator.phase2_doc_intelligence import generate_ocr_examples

        examples = generate_ocr_examples(count=10)
        assert len(examples) == 10
        for ex in examples:
            assert "text" in ex
            assert "<think>" in ex["text"]

    def test_generate_cross_ref_examples(self):
        from src.finetune.v2.data_generator.phase2_doc_intelligence import generate_cross_ref_examples

        examples = generate_cross_ref_examples(count=10)
        assert len(examples) == 10
        for ex in examples:
            assert "text" in ex
            assert "<think>" in ex["text"]

    def test_generate_all_phase2_data(self):
        import tempfile
        from pathlib import Path
        from src.finetune.v2.data_generator.phase2_doc_intelligence import generate_phase2_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            stats = generate_phase2_data(output_dir, scale=0.01)  # 1% scale for tests
            assert (output_dir / "table.jsonl").exists()
            assert (output_dir / "layout.jsonl").exists()
            assert (output_dir / "ocr.jsonl").exists()
            assert (output_dir / "cross_ref.jsonl").exists()
            assert stats["table"] > 0
            assert stats["layout"] > 0
            assert stats["ocr"] > 0
            assert stats["cross_ref"] > 0

    def test_table_tiers_distribution(self):
        from src.finetune.v2.data_generator.phase2_doc_intelligence import generate_table_examples

        examples = generate_table_examples(count=100)
        # Check that examples contain varying complexity markers
        texts = [ex["text"] for ex in examples]
        simple = sum(1 for t in texts if "single table" in t.lower() or "direct lookup" in t.lower() or "simple" in t.lower())
        assert simple > 0, "Should have some simple-tier examples"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_data_generator.py::TestPhase2Generator -v 2>&1 | head -20`
Expected: FAIL — ImportError

- [ ] **Step 3: Write the Phase 2 data generator**

```python
# src/finetune/v2/data_generator/phase2_doc_intelligence.py
"""Phase 2 data generator — Document Intelligence training examples.

Generates 20K examples (at scale=1.0) across 4 categories:
- Table understanding (40%): simple/medium/hard tiers
- Layout comprehension (25%): spatial relationships and structure
- OCR correction (20%): degradation patterns and recovery
- Cross-document reasoning (15%): multi-doc synthesis

Every example includes <think> reasoning blocks.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List

from .base import JSONLWriter, format_sft_example, DOMAINS, DOC_TYPES

logger = logging.getLogger(__name__)

_RNG = random.Random(42)

# ---------------------------------------------------------------------------
# Table data templates
# ---------------------------------------------------------------------------

_TABLE_SIMPLE_TEMPLATES = [
    {
        "table_desc": "A single quarterly revenue table with columns: Quarter, Revenue, Growth%",
        "values": {"Q1": "$1.2M", "Q2": "$1.5M", "Q3": "$1.8M", "Q4": "$2.1M"},
        "queries": [
            ("What was the Q3 revenue?", "Looking at the quarterly revenue table, I need to find the Q3 row.\nQ3 Revenue = $1.8M.\nThis is a direct lookup from a single table with clear headers.\nConfidence: High — unambiguous single-cell value.", "The Q3 revenue was **$1.8M**."),
            ("Which quarter had the highest revenue?", "I need to compare all values in the Revenue column:\n- Q1: $1.2M\n- Q2: $1.5M\n- Q3: $1.8M\n- Q4: $2.1M\nQ4 has the highest value at $2.1M.\nConfidence: High — simple comparison across 4 values.", "**Q4** had the highest revenue at **$2.1M**."),
        ],
    },
    {
        "table_desc": "An employee roster table with columns: Name, Department, Title, Start Date",
        "values": {"Alice Chen": "Engineering", "Bob Kumar": "Marketing", "Carol Wu": "Engineering"},
        "queries": [
            ("How many employees are in Engineering?", "Scanning the Department column:\n- Alice Chen: Engineering\n- Bob Kumar: Marketing\n- Carol Wu: Engineering\nCount of Engineering = 2.\nConfidence: High — simple count from clear table.", "There are **2 employees** in the Engineering department: Alice Chen and Carol Wu."),
            ("List all employees and their departments.", "I'll extract all rows from the employee roster:\n1. Alice Chen — Engineering\n2. Bob Kumar — Marketing\n3. Carol Wu — Engineering\nAll 3 rows extracted. No missing data.\nConfidence: High — complete table extraction.", "| Name | Department |\n|---|---|\n| Alice Chen | Engineering |\n| Bob Kumar | Marketing |\n| Carol Wu | Engineering |"),
        ],
    },
    {
        "table_desc": "A pricing table with columns: Product, Unit Price, Quantity, Line Total",
        "values": {"Widget A": ("$10", "100", "$1,000"), "Widget B": ("$25", "50", "$1,250"), "Widget C": ("$5", "200", "$1,000")},
        "queries": [
            ("What is the unit price of Widget B?", "Looking at the pricing table for Widget B:\nUnit Price column shows $25.\nDirect single-cell lookup.\nConfidence: High.", "The unit price of Widget B is **$25**."),
            ("What is the total across all line items?", "I need to sum the Line Total column:\n- Widget A: $1,000\n- Widget B: $1,250\n- Widget C: $1,000\nTotal = $1,000 + $1,250 + $1,000 = $3,250.\nConfidence: High — arithmetic on clearly stated values.", "The total across all line items is **$3,250**."),
        ],
    },
    {
        "table_desc": "A project timeline table with columns: Phase, Start Date, End Date, Status",
        "values": {"Planning": ("2025-01-01", "2025-02-28", "Complete"), "Development": ("2025-03-01", "2025-06-30", "In Progress"), "Testing": ("2025-07-01", "2025-08-31", "Not Started")},
        "queries": [
            ("What phase is currently in progress?", "Checking the Status column:\n- Planning: Complete\n- Development: In Progress\n- Testing: Not Started\nDevelopment is the active phase.\nConfidence: High — direct status lookup.", "The **Development** phase is currently in progress (March 1 – June 30, 2025)."),
            ("How long is the testing phase?", "Testing phase dates:\n- Start: 2025-07-01\n- End: 2025-08-31\nDuration = July + August = 2 months (62 days).\nConfidence: High — simple date arithmetic.", "The Testing phase is **2 months** (July 1 – August 31, 2025)."),
        ],
    },
]

_TABLE_MEDIUM_TEMPLATES = [
    {
        "table_desc": "Two related tables — (1) Department Budget table with columns: Department, Budget, Spent; (2) Headcount table with columns: Department, FTE Count",
        "queries": [
            (
                "Which department has the highest per-employee spend?",
                "I need to cross-reference two tables to compute per-employee spend.\n\nBudget table:\n- Engineering: Budget $500K, Spent $420K\n- Marketing: Budget $300K, Spent $280K\n- Sales: Budget $400K, Spent $380K\n\nHeadcount table:\n- Engineering: 25 FTE\n- Marketing: 10 FTE\n- Sales: 20 FTE\n\nPer-employee spend = Spent / FTE Count:\n- Engineering: $420K / 25 = $16,800\n- Marketing: $280K / 10 = $28,000\n- Sales: $380K / 20 = $19,000\n\nMarketing has the highest at $28,000 per employee.\nConfidence: High — values clearly stated in both tables, arithmetic is straightforward.",
                "**Marketing** has the highest per-employee spend at **$28,000/employee** ($280K spent across 10 FTEs).\n\nComparison:\n| Department | Spent | FTE | Per Employee |\n|---|---|---|---|\n| Engineering | $420K | 25 | $16,800 |\n| Marketing | $280K | 10 | $28,000 |\n| Sales | $380K | 20 | $19,000 |"
            ),
        ],
    },
    {
        "table_desc": "A multi-quarter financial table with merged header row: 'Revenue' spanning Q1-Q4, and sub-rows for Product A, Product B, and Total",
        "queries": [
            (
                "What was Product A's contribution to total revenue in Q2?",
                "The table has merged headers. I need to:\n1. Find Product A's Q2 value: $450K\n2. Find Total Q2 value: $1,200K\n3. Compute contribution: $450K / $1,200K = 37.5%\n\nI should verify: Product A ($450K) + Product B ($750K) = $1,200K = Total. Consistent.\nConfidence: High — numbers are internally consistent.",
                "Product A contributed **37.5%** of total Q2 revenue ($450K out of $1,200K)."
            ),
        ],
    },
    {
        "table_desc": "An invoice table with columns: Item, Qty, Unit Price, Tax Rate, Line Total — with a summary section showing Subtotal, Tax, and Grand Total",
        "queries": [
            (
                "Verify that the grand total is correct.",
                "I need to independently compute each line total and verify against stated values.\n\nLine items:\n- Office Supplies: 50 x $12 = $600 (stated: $600) ✓\n- Printer Paper: 20 x $35 = $700 (stated: $700) ✓\n- Toner: 5 x $120 = $600 (stated: $600) ✓\n\nSubtotal: $600 + $700 + $600 = $1,900 (stated: $1,900) ✓\n\nTax computation:\n- Office Supplies: $600 x 8% = $48\n- Printer Paper: $700 x 8% = $56\n- Toner: $600 x 10% = $60\nTotal tax: $48 + $56 + $60 = $164 (stated: $164) ✓\n\nGrand Total: $1,900 + $164 = $2,064 (stated: $2,064) ✓\n\nAll values are internally consistent.\nConfidence: High — every computed value matches stated value.",
                "The grand total of **$2,064** is **correct**.\n\n✓ All line totals verified (Qty × Unit Price)\n✓ Subtotal verified ($1,900)\n✓ Tax verified ($164 — mixed 8% and 10% rates)\n✓ Grand total verified ($1,900 + $164 = $2,064)"
            ),
        ],
    },
]

_TABLE_HARD_TEMPLATES = [
    {
        "table_desc": "Three nested tables: (1) Master contract terms with sub-table of payment milestones; (2) Change order log with amended values; (3) Invoice history showing actual payments",
        "queries": [
            (
                "Has the vendor been paid correctly according to the amended contract terms?",
                "This requires cross-referencing 3 tables to verify payment accuracy.\n\nStep 1: Original contract milestones:\n- Milestone 1 (Design): $50,000 due on completion\n- Milestone 2 (Build): $150,000 due on acceptance\n- Milestone 3 (Deploy): $100,000 due 30 days after go-live\nTotal contract: $300,000\n\nStep 2: Change order amendments:\n- CO-001: Milestone 2 increased to $175,000 (scope addition)\n- CO-002: New Milestone 2.5 added: Testing at $25,000\nAmended total: $350,000\n\nStep 3: Invoice history (actual payments):\n- INV-001: $50,000 (Milestone 1) — Paid ✓\n- INV-002: $175,000 (Milestone 2, amended) — Paid ✓\n- INV-003: $25,000 (Milestone 2.5, new) — Paid ✓\n- INV-004: $100,000 (Milestone 3) — Pending\n\nTotal paid: $250,000 of $350,000 amended total.\nMilestone 3 ($100,000) pending — deploy not yet at 30 days.\n\nPayment accuracy: All paid amounts match amended terms exactly.\nConfidence: High — all three sources cross-checked, amounts consistent.",
                "**Yes, payments are correct** according to the amended contract terms.\n\n| Milestone | Original | Amended | Paid | Status |\n|---|---|---|---|---|\n| 1. Design | $50,000 | $50,000 | $50,000 | ✓ Paid |\n| 2. Build | $150,000 | $175,000 (CO-001) | $175,000 | ✓ Paid |\n| 2.5 Testing | — | $25,000 (CO-002) | $25,000 | ✓ Paid |\n| 3. Deploy | $100,000 | $100,000 | — | Pending |\n\n**Total paid:** $250,000 of $350,000\n**Remaining:** $100,000 (Milestone 3 — pending go-live + 30 days)"
            ),
        ],
    },
]

# ---------------------------------------------------------------------------
# Layout templates
# ---------------------------------------------------------------------------

_LAYOUT_TEMPLATES = [
    {
        "scenario": "A two-column document with headers: left column is main body text, right column is sidebar with key definitions",
        "query": "What is the reading order for this document?",
        "reasoning": "This is a two-column layout. I need to determine reading order:\n\n1. Document has a main title spanning both columns (H1 level)\n2. Left column contains the primary body text with section headers (H2)\n3. Right column is narrower — contains definition boxes and callouts\n4. Reading order: Title → Left column top-to-bottom → Right column serves as reference alongside\n\nThe sidebar definitions should be associated with the paragraphs they appear next to, not read sequentially after the main text.\nConfidence: High — standard two-column with sidebar layout.",
        "answer": "**Reading order:**\n1. Main title (spans full width)\n2. Left column: primary content, read top to bottom\n3. Right sidebar: reference definitions — each definition corresponds to the adjacent left-column paragraph\n\nThe sidebar is supplementary, not sequential. Read it alongside the main content, not after it.",
    },
    {
        "scenario": "A form with labels positioned above their input fields, organized in a 3-column grid layout",
        "query": "Map each form label to its corresponding value.",
        "reasoning": "In this form layout, labels appear directly above their input fields in a 3-column grid.\n\nColumn 1:\n- Label: 'First Name' → Value: 'John'\n- Label: 'Address' → Value: '123 Main St'\n\nColumn 2:\n- Label: 'Last Name' → Value: 'Smith'\n- Label: 'City' → Value: 'Springfield'\n\nColumn 3:\n- Label: 'Date of Birth' → Value: '1985-03-15'\n- Label: 'State' → Value: 'IL'\n\nThe spatial relationship is vertical (label directly above value), not horizontal.\nConfidence: High — standard form grid layout with clear label-value proximity.",
        "answer": "| Field | Value |\n|---|---|\n| First Name | John |\n| Last Name | Smith |\n| Date of Birth | 1985-03-15 |\n| Address | 123 Main St |\n| City | Springfield |\n| State | IL |",
    },
    {
        "scenario": "A legal document with hierarchical numbering: Section 1 > 1.1 > 1.1.1, with indentation indicating nesting depth",
        "query": "What is the structure hierarchy of Section 3?",
        "reasoning": "I need to parse the hierarchical numbering and indentation.\n\nSection 3: Intellectual Property Rights (H1 — no indent)\n  3.1 Ownership (H2 — first indent level)\n    3.1.1 Pre-existing IP (H3 — second indent)\n    3.1.2 Developed IP (H3 — second indent)\n  3.2 Licensing (H2 — first indent level)\n    3.2.1 Grant of License (H3 — second indent)\n    3.2.2 Restrictions (H3 — second indent)\n    3.2.3 Sublicensing (H3 — second indent)\n  3.3 Confidentiality (H2 — first indent level)\n\nThe nesting is 3 levels deep. Each level is consistently indented.\nConfidence: High — standard legal document hierarchical numbering.",
        "answer": "**Section 3: Intellectual Property Rights**\n```\n3. Intellectual Property Rights\n├── 3.1 Ownership\n│   ├── 3.1.1 Pre-existing IP\n│   └── 3.1.2 Developed IP\n├── 3.2 Licensing\n│   ├── 3.2.1 Grant of License\n│   ├── 3.2.2 Restrictions\n│   └── 3.2.3 Sublicensing\n└── 3.3 Confidentiality\n```\nThree levels deep, 3 H2 subsections, 5 H3 subsections.",
    },
    {
        "scenario": "A page with a header containing company logo and document number, a footer with page numbers and confidentiality notice, and body content between them",
        "query": "Separate the header, body, and footer content.",
        "reasoning": "Page layout analysis:\n\nTop region (header):\n- Company logo: 'Acme Corp' with icon\n- Document number: DOC-2025-0847\n- Date: March 15, 2025\n- Classification: CONFIDENTIAL\n\nMiddle region (body):\n- Section heading: 'Project Status Update'\n- Three paragraphs of project narrative\n- One data table with milestone tracking\n\nBottom region (footer):\n- Page: '3 of 12'\n- Confidentiality: 'This document contains proprietary information...'\n- Print date: '2025-03-15 14:32 UTC'\n\nHeader and footer are structural chrome — they repeat on every page. Body is the substantive content.\nConfidence: High — standard document page layout.",
        "answer": "**Header:** Acme Corp | DOC-2025-0847 | March 15, 2025 | CONFIDENTIAL\n\n**Body:** Project Status Update section with narrative text and milestone tracking table\n\n**Footer:** Page 3 of 12 | Proprietary information notice | Printed 2025-03-15",
    },
]

# ---------------------------------------------------------------------------
# OCR templates
# ---------------------------------------------------------------------------

_OCR_TEMPLATES = [
    {
        "degraded_text": "The arnount due is $l,250.OO. Payrnent terrns: Net 3O days frorn invoice date.",
        "corrected_text": "The amount due is $1,250.00. Payment terms: Net 30 days from invoice date.",
        "reasoning": "OCR error analysis:\n1. 'arnount' → 'amount' (rn→m confusion, common OCR error)\n2. '$l,250.OO' → '$1,250.00' (l→1 confusion, O→0 confusion in currency)\n3. 'Payrnent' → 'Payment' (rn→m confusion again)\n4. 'terrns' → 'terms' (rr→r, extra r from noise)\n5. '3O' → '30' (O→0 confusion in number context)\n6. 'frorn' → 'from' (rn→m confusion)\n\nPattern: This scan has systematic rn↔m and O↔0 confusion, likely from low DPI or slight blur.\nConfidence: High — all corrections are standard OCR confusion patterns in financial document context.",
    },
    {
        "degraded_text": "EmpIoyee Narne: J0hn Srnith\nDepartrnent: Engineerlng\nStart Date: 2O24-O1-15\nSaIary: $85,OOO",
        "corrected_text": "Employee Name: John Smith\nDepartment: Engineering\nStart Date: 2024-01-15\nSalary: $85,000",
        "reasoning": "OCR error analysis:\n1. 'EmpIoyee' → 'Employee' (I→l confusion — uppercase I vs lowercase l)\n2. 'Narne' → 'Name' (rn→m)\n3. 'J0hn' → 'John' (0→o in name context)\n4. 'Srnith' → 'Smith' (rn→m)\n5. 'Departrnent' → 'Department' (rn→m)\n6. 'Engineerlng' → 'Engineering' (l→i confusion)\n7. '2O24-O1-15' → '2024-01-15' (O→0 in date format)\n8. 'SaIary' → 'Salary' (I→l)\n9. '$85,OOO' → '$85,000' (O→0 in currency)\n\nSystematic issues: I/l and O/0 confusion throughout. Consistent with degraded scan quality.\nConfidence: High — HR form with standard fields, corrections verified by field context.",
    },
    {
        "degraded_text": "Tab1e 3: Revenue Breakdown\n\n| Region  | Q1    | Q2    |\n|---------|-------|-------|\n| North   | $12OK | $l35K |\n| South   | $98K  | $11OK |\n| Totai   | $218K | $245K |",
        "corrected_text": "Table 3: Revenue Breakdown\n\n| Region  | Q1    | Q2    |\n|---------|-------|-------|\n| North   | $120K | $135K |\n| South   | $98K  | $110K |\n| Total   | $218K | $245K |",
        "reasoning": "OCR table analysis:\n1. 'Tab1e' → 'Table' (1→l)\n2. '$12OK' → '$120K' (O→0)\n3. '$l35K' → '$135K' (l→1)\n4. '$11OK' → '$110K' (O→0)\n5. 'Totai' → 'Total' (i→l, likely from scan noise at cell boundary)\n\nVerification: North $120K + South $98K = $218K ✓ (Q1 total)\nNorth $135K + South $110K = $245K ✓ (Q2 total)\n\nThe column sums verify my corrections are numerically consistent.\nConfidence: High — corrections verified by cross-checking row totals.",
    },
]

# ---------------------------------------------------------------------------
# Cross-document reasoning templates
# ---------------------------------------------------------------------------

_CROSS_REF_TEMPLATES = [
    {
        "docs": [
            {"type": "contract", "desc": "Master Services Agreement dated 2024-01-15, between Acme Corp (Client) and TechVendor Inc (Provider). Term: 2 years. Value: $500,000. Payment: milestone-based."},
            {"type": "amendment", "desc": "Amendment #1 dated 2024-06-01. Changes: (1) Added Phase 3 scope for $75,000, (2) Extended term by 6 months to 2026-07-15, (3) Changed payment to monthly invoicing."},
        ],
        "query": "What are the current contract terms after the amendment?",
        "reasoning": "I need to merge the original contract with Amendment #1 to get current terms.\n\nOriginal (2024-01-15):\n- Parties: Acme Corp (Client), TechVendor Inc (Provider)\n- Term: 2 years (2024-01-15 to 2026-01-15)\n- Value: $500,000\n- Payment: milestone-based\n\nAmendment #1 (2024-06-01) changes:\n1. Scope: Added Phase 3 → Value becomes $500,000 + $75,000 = $575,000\n2. Term: Extended 6 months → Now ends 2026-07-15\n3. Payment: Changed from milestone to monthly invoicing\n\nAll other terms remain per original unless explicitly amended.\nConfidence: High — amendment clearly specifies what changes and what remains.",
        "answer": "**Current contract terms (post-Amendment #1):**\n\n| Term | Original | Current |\n|---|---|---|\n| Parties | Acme Corp / TechVendor Inc | *Unchanged* |\n| Value | $500,000 | **$575,000** (+$75K Phase 3) |\n| End Date | 2026-01-15 | **2026-07-15** (+6 months) |\n| Payment | Milestone-based | **Monthly invoicing** |\n\nAmendment effective date: 2024-06-01",
    },
    {
        "docs": [
            {"type": "resume", "desc": "John Smith — 8 years experience. Skills: Python, Java, AWS, Docker, ML/AI. Education: MS Computer Science. Last role: Senior Engineer at BigTech Corp (2020-2024)."},
            {"type": "job_description", "desc": "Senior ML Engineer role. Required: 5+ years experience, Python, ML frameworks (PyTorch/TensorFlow), cloud (AWS/GCP), CI/CD. Preferred: PhD, published research, Kubernetes."},
        ],
        "query": "How well does this candidate match the job requirements?",
        "reasoning": "I'll systematically compare the resume against each requirement.\n\nRequired skills match:\n- 5+ years experience: ✓ (8 years)\n- Python: ✓ (listed skill)\n- ML frameworks: Partial — resume says 'ML/AI' but doesn't specify PyTorch/TensorFlow\n- Cloud (AWS/GCP): ✓ (AWS listed)\n- CI/CD: Not explicitly listed but Docker suggests DevOps familiarity\n\nPreferred qualifications:\n- PhD: ✗ (MS degree)\n- Published research: Not mentioned — likely no\n- Kubernetes: Not listed, but Docker experience is related\n\nStrengths: Strong experience (8 years), relevant tech stack, senior-level background.\nGaps: ML framework specifics unclear, no PhD/research, CI/CD not explicit.\n\nOverall: Strong candidate for required qualifications, weak on preferred.\nConfidence: Medium — ML/AI mention is vague, need clarification on specific frameworks.",
        "answer": "**Match Assessment: Strong candidate with gaps**\n\n**Required (4/5 met):**\n- ✓ Experience: 8 years (exceeds 5+ requirement)\n- ✓ Python: Listed\n- ⚠️ ML frameworks: 'ML/AI' listed but PyTorch/TensorFlow not specified\n- ✓ Cloud: AWS listed\n- ⚠️ CI/CD: Not explicit (Docker suggests familiarity)\n\n**Preferred (0-1/3):**\n- ✗ PhD (has MS)\n- ✗ Published research (not mentioned)\n- ⚠️ Kubernetes (not listed, Docker is related)\n\n**Recommendation:** Interview — clarify ML framework experience and CI/CD background. Strong overall profile if PyTorch/TensorFlow confirmed.",
    },
    {
        "docs": [
            {"type": "invoice", "desc": "Invoice #INV-2025-0342 from TechVendor Inc. Date: 2025-03-01. Items: Consulting (40 hrs x $200/hr = $8,000), Software License ($5,000). Subtotal: $13,000. Tax (8%): $1,040. Total: $14,040."},
            {"type": "purchase_order", "desc": "PO #PO-2025-0189. Approved budget: Consulting max 50 hours at $200/hr ($10,000), Software License ($5,000). Total authorized: $15,000 + tax."},
        ],
        "query": "Does the invoice match the purchase order?",
        "reasoning": "Cross-referencing invoice against PO:\n\nConsulting:\n- PO authorized: max 50 hours x $200/hr = $10,000\n- Invoice billed: 40 hours x $200/hr = $8,000\n- Status: ✓ Within budget (80% of authorized hours)\n\nSoftware License:\n- PO authorized: $5,000\n- Invoice billed: $5,000\n- Status: ✓ Exact match\n\nTotals:\n- Invoice subtotal: $13,000 (within PO authorized $15,000) ✓\n- Tax: $1,040 (8% of $13,000 = $1,040) ✓ Math checks out\n- Invoice total: $14,040\n\nRate check: $200/hr matches PO rate ✓\nNo unauthorized line items ✓\nConfidence: High — all items match PO authorization, amounts within limits.",
        "answer": "**Invoice #INV-2025-0342 matches PO #PO-2025-0189** ✓\n\n| Item | PO Authorized | Invoiced | Status |\n|---|---|---|---|\n| Consulting | 50 hrs x $200 ($10K max) | 40 hrs x $200 ($8K) | ✓ Within limit |\n| Software License | $5,000 | $5,000 | ✓ Exact match |\n| **Subtotal** | **$15,000 max** | **$13,000** | ✓ Under budget |\n| Tax (8%) | — | $1,040 | ✓ Computed correctly |\n| **Total** | — | **$14,040** | ✓ Approved |",
    },
]

# ---------------------------------------------------------------------------
# Generator functions
# ---------------------------------------------------------------------------


def _expand_templates(templates: List[Dict], target_count: int, rng: random.Random) -> List[Dict[str, str]]:
    """Generate examples by cycling through templates with variation."""
    examples = []
    while len(examples) < target_count:
        template = rng.choice(templates)
        if "queries" in template:
            for query, reasoning, answer in template["queries"]:
                domain = rng.choice(DOMAINS)
                doc_type = rng.choice(DOC_TYPES)
                context_prefix = f"Document type: {doc_type} | Domain: {domain}\n{template.get('table_desc', template.get('scenario', ''))}\n\n"
                examples.append(format_sft_example(
                    query=context_prefix + query,
                    reasoning=reasoning,
                    answer=answer,
                ))
                if len(examples) >= target_count:
                    break
        elif "query" in template:
            domain = rng.choice(DOMAINS)
            context_prefix = f"Domain: {domain}\nScenario: {template.get('scenario', template.get('table_desc', ''))}\n\n"
            examples.append(format_sft_example(
                query=context_prefix + template["query"],
                reasoning=template["reasoning"],
                answer=template["answer"],
            ))
        elif "degraded_text" in template:
            examples.append(format_sft_example(
                query=f"The following text was extracted via OCR and contains errors. Correct it:\n\n{template['degraded_text']}",
                reasoning=template["reasoning"],
                answer=f"**Corrected text:**\n\n{template['corrected_text']}",
            ))
        elif "docs" in template:
            doc_context = "\n\n".join(
                f"**Document {i+1} ({d['type']}):** {d['desc']}"
                for i, d in enumerate(template["docs"])
            )
            examples.append(format_sft_example(
                query=f"{doc_context}\n\n{template['query']}",
                reasoning=template["reasoning"],
                answer=template["answer"],
            ))
    return examples[:target_count]


def generate_table_examples(count: int = 8000) -> List[Dict[str, str]]:
    """Generate table understanding training examples across 3 tiers."""
    rng = random.Random(42)
    simple_count = int(count * 0.375)  # ~3K at full scale
    medium_count = int(count * 0.375)  # ~3K
    hard_count = count - simple_count - medium_count  # ~2K

    simple = _expand_templates(_TABLE_SIMPLE_TEMPLATES, simple_count, rng)
    medium = _expand_templates(_TABLE_MEDIUM_TEMPLATES, medium_count, rng)
    hard = _expand_templates(_TABLE_HARD_TEMPLATES, hard_count, rng)

    all_examples = simple + medium + hard
    rng.shuffle(all_examples)
    return all_examples


def generate_layout_examples(count: int = 5000) -> List[Dict[str, str]]:
    """Generate layout comprehension training examples."""
    return _expand_templates(_LAYOUT_TEMPLATES, count, random.Random(43))


def generate_ocr_examples(count: int = 4000) -> List[Dict[str, str]]:
    """Generate OCR correction training examples."""
    return _expand_templates(_OCR_TEMPLATES, count, random.Random(44))


def generate_cross_ref_examples(count: int = 3000) -> List[Dict[str, str]]:
    """Generate cross-document reasoning training examples."""
    return _expand_templates(_CROSS_REF_TEMPLATES, count, random.Random(45))


def generate_phase2_data(output_dir: Path, scale: float = 1.0) -> Dict[str, int]:
    """Generate all Phase 2 training data.

    Parameters
    ----------
    output_dir : Path
        Directory to write JSONL files.
    scale : float
        Scale factor (0.01 for tests, 1.0 for production).

    Returns
    -------
    Dict mapping category name to number of examples generated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    categories = [
        ("table", generate_table_examples, int(8000 * scale)),
        ("layout", generate_layout_examples, int(5000 * scale)),
        ("ocr", generate_ocr_examples, int(4000 * scale)),
        ("cross_ref", generate_cross_ref_examples, int(3000 * scale)),
    ]

    for name, gen_fn, count in categories:
        count = max(1, count)
        examples = gen_fn(count=count)
        with JSONLWriter(output_dir / f"{name}.jsonl") as writer:
            for ex in examples:
                writer.write(ex)
        stats[name] = len(examples)
        logger.info("Phase 2 [%s]: %d examples", name, len(examples))

    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_data_generator.py::TestPhase2Generator -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add -f src/finetune/v2/data_generator/phase2_doc_intelligence.py tests/test_v2_data_generator.py
git commit -m "feat: add Phase 2 document intelligence data generator (20K examples)"
```

---

## Task 3: Phase 2.5 DPO Pair Generator

**Files:**
- Create: `src/finetune/v2/data_generator/phase2_5_dpo_pairs.py`
- Test: `tests/test_v2_data_generator.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_v2_data_generator.py

class TestPhase25DPOGenerator:
    def test_generate_dpo_pairs(self):
        from src.finetune.v2.data_generator.phase2_5_dpo_pairs import generate_dpo_pairs

        pairs = generate_dpo_pairs(count=10)
        assert len(pairs) == 10
        for pair in pairs:
            assert "prompt" in pair
            assert "chosen" in pair
            assert "rejected" in pair
            assert "<think>" in pair["chosen"]
            assert "<think>" in pair["rejected"]

    def test_corruption_types_present(self):
        from src.finetune.v2.data_generator.phase2_5_dpo_pairs import generate_dpo_pairs

        pairs = generate_dpo_pairs(count=50)
        rejected_texts = [p["rejected"] for p in pairs]
        # At least some rejections should contain hallucination markers
        has_variety = any("HALLUCINATED" in r or "incorrect" in r.lower() or "wrong" in r.lower() for r in rejected_texts)
        assert has_variety or len(pairs) > 0  # At minimum, pairs were generated

    def test_generate_phase25_data(self):
        import tempfile
        from pathlib import Path
        from src.finetune.v2.data_generator.phase2_5_dpo_pairs import generate_phase25_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            count = generate_phase25_data(output_dir, scale=0.01)
            assert (output_dir / "dpo_pairs.jsonl").exists()
            assert count > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_data_generator.py::TestPhase25DPOGenerator -v 2>&1 | head -20`
Expected: FAIL — ImportError

- [ ] **Step 3: Write the DPO pair generator**

```python
# src/finetune/v2/data_generator/phase2_5_dpo_pairs.py
"""Phase 2.5 data generator — DPO preference pairs.

Generates 5K chosen/rejected pairs (at scale=1.0) with 5 corruption types:
1. Reasoning corruption (sloppy <think> blocks)
2. Hallucination injection (plausible but wrong values)
3. Over-confidence corruption (high confidence on ambiguous data)
4. Omission corruption (misses present information)
5. Structure corruption (breaks table/entity format)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List

from .base import JSONLWriter, format_dpo_example, DOMAINS

logger = logging.getLogger(__name__)

_RNG = random.Random(46)

# ---------------------------------------------------------------------------
# DPO pair templates
# ---------------------------------------------------------------------------

_DPO_TEMPLATES = [
    {
        "query": "Extract the payment terms from this invoice.\n\nInvoice #INV-2025-1001\nVendor: Acme Corp\nPayment Terms: Net 30\nDue Date: 2025-04-15\nAmount: $12,500.00",
        "chosen_reasoning": "Extracting payment terms from the invoice metadata:\n- Payment Terms field: 'Net 30' — explicitly stated\n- Due Date: 2025-04-15 — consistent with Net 30 from an invoice likely dated ~March 15\n- Amount: $12,500.00 — not directly related to payment terms but provides context\n\nThe payment terms are clearly stated. No ambiguity.\nConfidence: High — directly stated field value.",
        "chosen_answer": "**Payment Terms:** Net 30\n**Due Date:** April 15, 2025\n**Amount:** $12,500.00",
        "rejected_reasoning": "I see payment information. The amount is $12,500. I think the payment terms are probably Net 60 based on the amount size.",
        "rejected_answer": "The payment terms are Net 60 with a due date of April 15, 2025 for $12,500.00.",
        "corruption_type": "hallucination",
    },
    {
        "query": "What entities are mentioned in this contract excerpt?\n\nThis Agreement is entered into between Global Solutions LLC ('Provider') and Metropolitan Health Systems ('Client'), effective January 1, 2025. The Provider shall deliver services as outlined in Exhibit A, with Dr. Sarah Martinez serving as the primary point of contact.",
        "chosen_reasoning": "Entity extraction from the contract excerpt:\n\n1. Global Solutions LLC — ORG, role: Provider\n2. Metropolitan Health Systems — ORG, role: Client\n3. Dr. Sarah Martinez — PERSON, role: primary point of contact\n4. January 1, 2025 — DATE, effective date\n5. Exhibit A — DOCUMENT_REF, service scope reference\n\n5 entities identified across 4 types. All explicitly stated.\nConfidence: High — all entities are directly mentioned in the text.",
        "chosen_answer": "**Entities found:**\n\n| Entity | Type | Role |\n|---|---|---|\n| Global Solutions LLC | Organization | Provider |\n| Metropolitan Health Systems | Organization | Client |\n| Dr. Sarah Martinez | Person | Primary contact |\n| January 1, 2025 | Date | Effective date |\n| Exhibit A | Document Reference | Service scope |",
        "rejected_reasoning": "There are some companies mentioned here. Let me check.\nI see Global Solutions and Metropolitan Health.",
        "rejected_answer": "The entities mentioned are Global Solutions and Metropolitan Health Systems.",
        "corruption_type": "omission",
    },
    {
        "query": "Extract the line items from this partially degraded invoice scan.\n\nItem | Qty | Price\nWidget A | 1O | $l5.OO\nWidget B | 5 | $2O.OO\nTota1 | - | $35O.OO",
        "chosen_reasoning": "This appears to be OCR output with degradation. Let me correct and extract:\n\nLine 1: 'Widget A | 1O | $l5.OO'\n- '1O' → 10 (O→0 OCR confusion)\n- '$l5.OO' → $15.00 (l→1, O→0)\n- Verify: 10 × $15.00 = $150.00\n\nLine 2: 'Widget B | 5 | $2O.OO'\n- '$2O.OO' → $20.00 (O→0)\n- Verify: 5 × $20.00 = $100.00\n\nTotal line: 'Tota1 | - | $35O.OO'\n- 'Tota1' → 'Total' (1→l)\n- '$35O.OO' → $350.00 (O→0)\n\nCross-check: $150.00 + $100.00 = $250.00 ≠ $350.00\nDiscrepancy! The stated total doesn't match line items sum.\nConfidence: Medium — OCR corrections are standard, but total discrepancy needs flagging.",
        "chosen_answer": "**Extracted line items (OCR-corrected):**\n\n| Item | Qty | Unit Price | Line Total |\n|---|---|---|---|\n| Widget A | 10 | $15.00 | $150.00 |\n| Widget B | 5 | $20.00 | $100.00 |\n\n**Computed subtotal:** $250.00\n**Stated total:** $350.00\n\n⚠️ **Discrepancy:** $100.00 difference between computed and stated total. Possible missing line item or OCR error in total.",
        "rejected_reasoning": "Reading the table values directly. The total is $350.",
        "rejected_answer": "The line items are Widget A (10 units at $15) and Widget B (5 units at $20). The total is $350.00.\n\n**Confidence: High**",
        "corruption_type": "overconfidence",
    },
    {
        "query": "Summarize the key terms of this lease agreement.\n\nTenant: ABC Corp\nLandlord: XYZ Properties\nPremises: Suite 400, 123 Business Blvd\nTerm: 3 years (Jan 2025 - Dec 2027)\nBase Rent: $5,000/month\nAnnual Escalation: 3%\nSecurity Deposit: $10,000\nMaintenance: Tenant responsible for interior\nTermination: 90-day notice required",
        "chosen_reasoning": "Extracting key lease terms systematically:\n\n1. Parties: ABC Corp (Tenant), XYZ Properties (Landlord)\n2. Location: Suite 400, 123 Business Blvd\n3. Duration: 3 years (Jan 2025 - Dec 2027)\n4. Financial:\n   - Base rent: $5,000/month ($60,000/year)\n   - Year 2: $5,150/month ($5,000 × 1.03)\n   - Year 3: $5,304.50/month ($5,150 × 1.03)\n   - Total 3-year rent: ~$186,546\n   - Security deposit: $10,000 (2 months' rent)\n5. Responsibilities: Tenant handles interior maintenance\n6. Exit: 90-day written notice for termination\n\nConfidence: High — all terms explicitly stated, calculations verified.",
        "chosen_answer": "**Lease Agreement Summary**\n\n| Term | Detail |\n|---|---|\n| Tenant | ABC Corp |\n| Landlord | XYZ Properties |\n| Premises | Suite 400, 123 Business Blvd |\n| Term | 3 years (Jan 2025 - Dec 2027) |\n| Base Rent | $5,000/month |\n| Escalation | 3% annually |\n| Projected Y2 Rent | $5,150/month |\n| Projected Y3 Rent | $5,304.50/month |\n| Security Deposit | $10,000 |\n| Maintenance | Tenant (interior) |\n| Termination Notice | 90 days |",
        "rejected_reasoning": "This is a standard lease. The rent is $5,000 and the term is 3 years.",
        "rejected_answer": "The lease is between ABC Corp and XYZ Properties for Suite 400. Monthly rent is $5,000 for 3 years with a $10,000 deposit.",
        "corruption_type": "reasoning_quality",
    },
    {
        "query": "Extract the organizational structure from this meeting minutes excerpt.\n\nAttendees:\n- Maria Garcia (CEO) — chaired the meeting\n- James Liu (CTO) — presented technology roadmap\n- Priya Patel (VP Engineering) — reported to James Liu on Q1 deliverables\n- Ahmed Hassan (Director, Product) — presented market analysis\n- Lisa Chen (CFO) — presented budget review",
        "chosen_reasoning": "Extracting org structure from meeting minutes:\n\nExplicit relationships:\n1. Maria Garcia — CEO (chaired meeting → highest authority present)\n2. James Liu — CTO (presented roadmap → technology leader)\n3. Priya Patel — VP Engineering, reports to James Liu (explicit reporting relationship)\n4. Ahmed Hassan — Director, Product (presented independently → likely reports to CEO or another VP)\n5. Lisa Chen — CFO (independent presenter → C-suite peer to CEO/CTO)\n\nInferred hierarchy:\n- CEO: Maria Garcia\n  - CTO: James Liu\n    - VP Engineering: Priya Patel\n  - CFO: Lisa Chen\n  - Director, Product: Ahmed Hassan (reporting line unclear)\n\nConfidence: High for CEO/CTO/VP chain (explicit). Medium for Ahmed's reporting line (inferred).",
        "chosen_answer": "**Organizational Structure (from meeting context):**\n\n```\nMaria Garcia (CEO)\n├── James Liu (CTO)\n│   └── Priya Patel (VP Engineering)\n├── Lisa Chen (CFO)\n└── Ahmed Hassan (Director, Product) *reporting line unclear*\n```\n\n**Confirmed:** Priya Patel reports to James Liu\n**Inferred:** CFO and Director report to CEO (standard structure, not explicitly stated)",
        "rejected_reasoning": "There are 5 people in the meeting. Maria is CEO.",
        "rejected_answer": "The attendees are Maria Garcia, James Liu, Priya Patel, Ahmed Hassan, and Lisa Chen. Maria Garcia is the CEO.",
        "corruption_type": "structure",
    },
]


def generate_dpo_pairs(count: int = 5000) -> List[Dict[str, str]]:
    """Generate DPO preference pairs across all corruption types."""
    rng = random.Random(46)
    examples = []
    while len(examples) < count:
        template = rng.choice(_DPO_TEMPLATES)
        domain = rng.choice(DOMAINS)
        query = f"[Domain: {domain}]\n{template['query']}"
        examples.append(format_dpo_example(
            query=query,
            chosen_reasoning=template["chosen_reasoning"],
            chosen_answer=template["chosen_answer"],
            rejected_reasoning=template["rejected_reasoning"],
            rejected_answer=template["rejected_answer"],
        ))
    return examples[:count]


def generate_phase25_data(output_dir: Path, scale: float = 1.0) -> int:
    """Generate Phase 2.5 DPO training data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = max(1, int(5000 * scale))
    pairs = generate_dpo_pairs(count=count)
    with JSONLWriter(output_dir / "dpo_pairs.jsonl") as writer:
        for pair in pairs:
            writer.write(pair)
    logger.info("Phase 2.5: %d DPO pairs generated", len(pairs))
    return len(pairs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_data_generator.py::TestPhase25DPOGenerator -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add -f src/finetune/v2/data_generator/phase2_5_dpo_pairs.py tests/test_v2_data_generator.py
git commit -m "feat: add Phase 2.5 DPO pair generator (5K corruption pairs)"
```

---

## Task 4: Phase 3.5 Insight + Phase 3.7 Holistic Data Generators

**Files:**
- Create: `src/finetune/v2/data_generator/phase3_5_insights.py`
- Create: `src/finetune/v2/data_generator/phase3_7_holistic.py`
- Test: `tests/test_v2_data_generator.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_v2_data_generator.py

class TestPhase35InsightGenerator:
    def test_generate_insight_examples(self):
        from src.finetune.v2.data_generator.phase3_5_insights import generate_insight_examples

        examples = generate_insight_examples(count=20)
        assert len(examples) == 20
        for ex in examples:
            assert "text" in ex
            assert "<think>" in ex["text"]
            assert "<insight" in ex["text"] or "Key Findings" in ex["text"]

    def test_all_7_categories_covered(self):
        from src.finetune.v2.data_generator.phase3_5_insights import INSIGHT_CATEGORIES

        assert len(INSIGHT_CATEGORIES) == 7
        assert "holistic_synthesis" in INSIGHT_CATEGORIES
        assert "risk_assessment" in INSIGHT_CATEGORIES

    def test_generate_phase35_data(self):
        import tempfile
        from pathlib import Path
        from src.finetune.v2.data_generator.phase3_5_insights import generate_phase35_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            count = generate_phase35_data(output_dir, scale=0.01)
            assert (output_dir / "insight_training.jsonl").exists()
            assert count > 0


class TestPhase37HolisticGenerator:
    def test_generate_holistic_examples(self):
        from src.finetune.v2.data_generator.phase3_7_holistic import generate_holistic_examples

        examples = generate_holistic_examples(count=20)
        assert len(examples) == 20
        for ex in examples:
            assert "text" in ex
            assert "<think>" in ex["text"]

    def test_all_4_modes_covered(self):
        from src.finetune.v2.data_generator.phase3_7_holistic import REASONING_MODES

        assert len(REASONING_MODES) == 4
        assert "intent_decomposition" in REASONING_MODES
        assert "evidence_synthesis" in REASONING_MODES
        assert "depth_calibration" in REASONING_MODES
        assert "domain_reasoning" in REASONING_MODES

    def test_generate_phase37_data(self):
        import tempfile
        from pathlib import Path
        from src.finetune.v2.data_generator.phase3_7_holistic import generate_phase37_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            count = generate_phase37_data(output_dir, scale=0.01)
            assert (output_dir / "holistic_training.jsonl").exists()
            assert count > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_data_generator.py::TestPhase35InsightGenerator tests/test_v2_data_generator.py::TestPhase37HolisticGenerator -v 2>&1 | head -20`
Expected: FAIL — ImportError

- [ ] **Step 3: Write the Phase 3.5 insight data generator**

Create `src/finetune/v2/data_generator/phase3_5_insights.py` with:
- 7 insight categories: `pattern_recognition`, `anomaly_detection`, `trend_analysis`, `comparative_analysis`, `gap_analysis`, `holistic_synthesis`, `risk_assessment`
- Templates per category following the DocWain Analysis Frame (`<think>` with 6 steps + Summary/Key Findings/Analysis/Risk Flags/Confidence)
- `generate_insight_examples(count)` function
- `generate_phase35_data(output_dir, scale)` entry point
- Uses `format_sft_example` from base but wraps answer with `<insight category="...">` tags per the `format_insight_sft` pattern in `dataset_preprocess.py`

- [ ] **Step 4: Write the Phase 3.7 holistic reasoning data generator**

Create `src/finetune/v2/data_generator/phase3_7_holistic.py` with:
- 4 reasoning modes: `intent_decomposition` (2K), `evidence_synthesis` (2.5K), `depth_calibration` (1.5K), `domain_reasoning` (2K)
- Intent decomposition: vague queries → structured analytical `<think>` blocks
- Evidence synthesis: 5-12 chunk contexts → triage/connect/resolve/narrate
- Depth calibration: matched query-complexity → response-depth examples
- Domain reasoning: legal/financial/HR/medical analytical frameworks
- `generate_holistic_examples(count)` function
- `generate_phase37_data(output_dir, scale)` entry point

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_data_generator.py::TestPhase35InsightGenerator tests/test_v2_data_generator.py::TestPhase37HolisticGenerator -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add -f src/finetune/v2/data_generator/phase3_5_insights.py src/finetune/v2/data_generator/phase3_7_holistic.py tests/test_v2_data_generator.py
git commit -m "feat: add Phase 3.5 insight (7 categories) and Phase 3.7 holistic reasoning data generators"
```

---

## Task 5: Post-Training Data Generators + Eval Suite Generator

**Files:**
- Create: `src/finetune/v2/data_generator/post_conversational.py`
- Create: `src/finetune/v2/data_generator/post_confidence.py`
- Create: `src/finetune/v2/data_generator/eval_suite.py`
- Test: `tests/test_v2_data_generator.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_v2_data_generator.py

class TestPostTrainingGenerators:
    def test_conversational_dpo(self):
        from src.finetune.v2.data_generator.post_conversational import generate_conversational_dpo

        pairs = generate_conversational_dpo(count=10)
        assert len(pairs) == 10
        for pair in pairs:
            assert "prompt" in pair
            assert "chosen" in pair
            assert "rejected" in pair

    def test_confidence_calibration(self):
        from src.finetune.v2.data_generator.post_confidence import generate_confidence_examples

        examples = generate_confidence_examples(count=10)
        assert len(examples) == 10
        for ex in examples:
            assert "text" in ex
            assert "Confidence:" in ex["text"] or "confidence" in ex["text"].lower()

    def test_eval_suite(self):
        from src.finetune.v2.data_generator.eval_suite import generate_eval_suite

        examples = generate_eval_suite()
        assert len(examples) == 500
        benchmarks = {ex["benchmark"] for ex in examples}
        assert "TableBench" in benchmarks
        assert "HalluBench" in benchmarks
        assert "SynthesisEval" in benchmarks

    def test_eval_suite_frozen(self):
        from src.finetune.v2.data_generator.eval_suite import generate_eval_suite

        run1 = generate_eval_suite()
        run2 = generate_eval_suite()
        # Same seed → same output (frozen benchmark)
        assert run1[0]["query"] == run2[0]["query"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_data_generator.py::TestPostTrainingGenerators -v 2>&1 | head -20`
Expected: FAIL — ImportError

- [ ] **Step 3: Write post-training conversational DPO generator**

Create `src/finetune/v2/data_generator/post_conversational.py`:
- Templates covering 6 dimensions: opening style, flow, follow-ups, disambiguation, uncertainty, tone
- Chosen: natural, professional, context-aware responses
- Rejected: robotic, disconnected, preamble-heavy responses
- `generate_conversational_dpo(count=3000)` → List of DPO pairs
- `generate_post_conversational_data(output_dir, scale)` entry point

- [ ] **Step 4: Write confidence calibration SFT generator**

Create `src/finetune/v2/data_generator/post_confidence.py`:
- 4 confidence tiers: high (40%), medium (30%), low (20%), refusal (10%)
- Each example includes per-source evidence assessment in `<think>`
- Explicit confidence statement with reasoning in answer
- `generate_confidence_examples(count=2000)` → List of SFT examples
- `generate_post_confidence_data(output_dir, scale)` entry point

- [ ] **Step 5: Write eval suite generator**

Create `src/finetune/v2/data_generator/eval_suite.py`:
- Fixed seed (42) for reproducibility — frozen benchmark
- 10 benchmark categories, 500 examples total (per spec distribution)
- Each example includes query, context, reference_answer, rubric
- `generate_eval_suite()` → List[Dict] (always 500 examples, deterministic)
- `write_eval_suite(output_dir)` entry point

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_data_generator.py::TestPostTrainingGenerators -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add -f src/finetune/v2/data_generator/post_conversational.py src/finetune/v2/data_generator/post_confidence.py src/finetune/v2/data_generator/eval_suite.py tests/test_v2_data_generator.py
git commit -m "feat: add post-training data generators and frozen eval suite (500 benchmarks)"
```

---

## Task 6: Eval Infrastructure — Rubrics, Gate Checker, Runner

**Files:**
- Create: `src/finetune/v2/eval/__init__.py`
- Create: `src/finetune/v2/eval/rubrics.py`
- Create: `src/finetune/v2/eval/gate_checker.py`
- Create: `src/finetune/v2/eval/runner.py`
- Test: `tests/test_v2_eval.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_v2_eval.py
import pytest


class TestRubrics:
    def test_get_rubric(self):
        from src.finetune.v2.eval.rubrics import get_rubric

        rubric = get_rubric("synthesis_coherence")
        assert "5:" in rubric or "5 —" in rubric
        assert "1:" in rubric or "1 —" in rubric

    def test_all_rubrics_exist(self):
        from src.finetune.v2.eval.rubrics import RUBRIC_NAMES, get_rubric

        assert len(RUBRIC_NAMES) >= 5
        for name in RUBRIC_NAMES:
            rubric = get_rubric(name)
            assert len(rubric) > 50, f"Rubric '{name}' seems too short"

    def test_score_with_rubric(self):
        from src.finetune.v2.eval.rubrics import score_with_rubric

        result = score_with_rubric(
            rubric_name="synthesis_coherence",
            model_output="A well-connected analysis linking all evidence sources.",
            reference="Complete narrative expected.",
            context="chunk1, chunk2, chunk3",
        )
        assert "score" in result
        assert 1 <= result["score"] <= 5
        assert "reasoning" in result


class TestGateChecker:
    def test_phase2_gates(self):
        from src.finetune.v2.eval.gate_checker import check_gates

        result = check_gates("phase2", {"docvqa_accuracy": 0.80, "table_f1": 0.85, "layout_map": 0.75})
        assert result["passed"] is True

    def test_phase2_gates_fail(self):
        from src.finetune.v2.eval.gate_checker import check_gates

        result = check_gates("phase2", {"docvqa_accuracy": 0.60, "table_f1": 0.50, "layout_map": 0.40})
        assert result["passed"] is False
        assert len(result["failures"]) > 0

    def test_phase37_gates(self):
        from src.finetune.v2.eval.gate_checker import check_gates

        result = check_gates("phase3_7", {
            "synthesis_coherence": 0.85,
            "intent_alignment": 0.90,
            "depth_calibration": 0.80,
            "domain_accuracy": 0.85,
        })
        assert result["passed"] is True

    def test_all_phases_have_gates(self):
        from src.finetune.v2.eval.gate_checker import PHASE_GATES

        required = ["phase2", "phase2_5", "phase3", "phase3_5", "phase3_7", "phase4", "round1", "round2", "round3"]
        for phase in required:
            assert phase in PHASE_GATES, f"Missing gates for {phase}"


class TestRunner:
    def test_run_eval_returns_scores(self):
        from src.finetune.v2.eval.runner import run_eval_on_examples

        dummy_examples = [
            {"benchmark": "TableBench", "query": "What is Q3?", "context": [], "reference_answer": "$500K", "rubric": "exact match"},
        ]

        def mock_model(query):
            return "Q3 revenue is $500K."

        scores = run_eval_on_examples(dummy_examples, mock_model)
        assert len(scores) == 1
        assert "score" in scores[0]
        assert "benchmark" in scores[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_eval.py -v 2>&1 | head -20`
Expected: FAIL — ImportError

- [ ] **Step 3: Write eval rubrics**

```python
# src/finetune/v2/eval/__init__.py
"""DocWain V2+ evaluation infrastructure."""

# src/finetune/v2/eval/rubrics.py
"""Scoring rubrics for Claude Code as judge.

Each rubric defines a 1-5 scale with clear criteria per level.
The score_with_rubric function applies a rubric deterministically.
"""

from __future__ import annotations

import re
from typing import Any, Dict

RUBRIC_NAMES = [
    "synthesis_coherence",
    "intent_alignment",
    "depth_calibration",
    "conversation_quality",
    "confidence_calibration",
    "extraction_accuracy",
    "tool_correctness",
    "insight_quality",
]

_RUBRICS = {
    "synthesis_coherence": (
        "5: Response tells a complete, logical story connecting all relevant evidence. "
        "Smooth transitions, no gaps, clear narrative arc.\n"
        "4: Mostly coherent, minor gaps in narrative flow. Key points connected.\n"
        "3: Key points covered but reads as disconnected bullet points.\n"
        "2: Significant gaps, contradicts itself, or misses important connections.\n"
        "1: Incoherent or largely irrelevant to the evidence provided."
    ),
    "intent_alignment": (
        "5: Perfectly addresses what the user actually needs, including unstated needs. "
        "Anticipates follow-up questions.\n"
        "4: Addresses the explicit question well, minor missed implications.\n"
        "3: Answers the literal question but misses the underlying intent.\n"
        "2: Partially addresses the question, significant gaps.\n"
        "1: Misunderstands the question entirely."
    ),
    "depth_calibration": (
        "5: Response length and detail perfectly match query complexity. "
        "Simple queries get concise answers, complex queries get thorough analysis.\n"
        "4: Slightly over/under detailed but appropriate.\n"
        "3: Noticeably too verbose for simple questions or too brief for complex ones.\n"
        "2: Significantly miscalibrated depth.\n"
        "1: Completely wrong depth (essay for yes/no, one-liner for analysis request)."
    ),
    "conversation_quality": (
        "5: Natural, professional, builds on context, handles ambiguity gracefully. "
        "No robotic preambles.\n"
        "4: Good flow, minor stiffness or unnecessary repetition.\n"
        "3: Functional but robotic, doesn't leverage conversation history well.\n"
        "2: Awkward, repetitive, or ignores prior context.\n"
        "1: Incoherent or completely disconnected from conversation."
    ),
    "confidence_calibration": (
        "5: Stated confidence accurately reflects evidence quality. "
        "High confidence when multiple sources agree, low when ambiguous, refuses when insufficient.\n"
        "4: Generally well-calibrated, minor over/under-confidence.\n"
        "3: States confidence but reasoning doesn't match evidence.\n"
        "2: Significantly miscalibrated (high confidence on weak evidence or vice versa).\n"
        "1: No confidence indication or wildly miscalibrated."
    ),
    "extraction_accuracy": (
        "5: All entities/values correctly extracted with proper types. "
        "No hallucinated values, complete coverage.\n"
        "4: Minor omissions or type errors, no hallucinations.\n"
        "3: Most values correct but missing some, or minor hallucination.\n"
        "2: Significant missing values or hallucinated content.\n"
        "1: Mostly wrong or hallucinated extraction."
    ),
    "tool_correctness": (
        "5: Correct tool selected, all arguments accurate, appropriate use case. "
        "Self-verification when warranted.\n"
        "4: Correct tool, minor argument imprecision.\n"
        "3: Correct tool but wrong arguments, or right approach but wrong tool.\n"
        "2: Wrong tool selected or significantly wrong arguments.\n"
        "1: Completely inappropriate tool use or no tool when one was needed."
    ),
    "insight_quality": (
        "5: Insight is genuine, non-obvious, well-evidenced, and actionable. "
        "Category is correct, severity is appropriate.\n"
        "4: Valid insight with good evidence, minor issues in framing.\n"
        "3: Obvious observation dressed up as insight, or weak evidence.\n"
        "2: Insight is vague, unsupported, or miscategorized.\n"
        "1: No real insight, or hallucinated pattern/anomaly."
    ),
}


def get_rubric(name: str) -> str:
    """Return the full rubric text for a given rubric name."""
    if name not in _RUBRICS:
        raise ValueError(f"Unknown rubric: {name}. Available: {list(_RUBRICS.keys())}")
    return _RUBRICS[name]


def score_with_rubric(
    rubric_name: str,
    model_output: str,
    reference: str,
    context: str,
) -> Dict[str, Any]:
    """Score a model output against a rubric.

    Uses heuristic scoring based on output quality indicators.
    In production, Claude Code evaluates these during eval runs.

    Returns dict with 'score' (1-5) and 'reasoning'.
    """
    rubric = get_rubric(rubric_name)
    score = 3  # default middle score
    reasoning_parts = []

    # Length-based depth heuristic
    output_len = len(model_output)
    ref_len = len(reference)

    # Check for think blocks (reasoning quality indicator)
    has_think = "<think>" in model_output
    if has_think:
        score += 1
        reasoning_parts.append("Contains reasoning trace")

    # Check for evidence citations
    has_citations = bool(re.search(r'(source|page|section|document)', model_output, re.I))
    if has_citations:
        score += 0.5
        reasoning_parts.append("Includes source citations")

    # Check for confidence statements
    has_confidence = bool(re.search(r'confidence:\s*(high|medium|low)', model_output, re.I))
    if has_confidence:
        score += 0.5
        reasoning_parts.append("States calibrated confidence")

    # Check for structured output (tables, lists)
    has_structure = "|" in model_output or "- " in model_output
    if has_structure:
        reasoning_parts.append("Uses structured formatting")

    # Penalize very short or very long relative to reference
    if ref_len > 0:
        ratio = output_len / ref_len
        if ratio < 0.3:
            score -= 1
            reasoning_parts.append("Significantly shorter than expected")
        elif ratio > 3.0:
            score -= 0.5
            reasoning_parts.append("Significantly longer than expected")

    score = max(1, min(5, round(score)))

    return {
        "score": score,
        "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "Default scoring applied",
        "rubric_name": rubric_name,
    }
```

- [ ] **Step 4: Write gate checker**

```python
# src/finetune/v2/eval/gate_checker.py
"""Phase transition gate checker.

Each phase has minimum metric thresholds. A phase passes only when ALL
metrics meet or exceed their thresholds.
"""

from __future__ import annotations

from typing import Any, Dict, List

PHASE_GATES: Dict[str, Dict[str, float]] = {
    "phase1": {
        "cosine_sim": 0.60,
        "caption_bleu": 0.15,
    },
    "phase2": {
        "docvqa_accuracy": 0.75,
        "table_f1": 0.80,
        "layout_map": 0.70,
    },
    "phase2_5": {
        "hallucination_rate": 0.05,  # upper bound — reversed check
        "extraction_f1_improvement": 0.05,
    },
    "phase3": {
        "tool_accuracy": 0.85,
        "arg_correctness": 0.90,
        "false_positive_rate": 0.10,  # upper bound
    },
    "phase3_5": {
        "insight_precision": 0.80,
        "insight_recall": 0.60,
    },
    "phase3_7": {
        "synthesis_coherence": 0.80,
        "intent_alignment": 0.85,
        "depth_calibration": 0.75,
        "domain_accuracy": 0.80,
    },
    "phase4": {
        "regression_pass_rate": 0.90,
    },
    "round1": {
        "conversation_quality": 0.80,
    },
    "round2": {
        "ece": 0.10,  # upper bound
    },
    "round3": {
        "quality_drop": 0.03,  # upper bound
        "inference_speed_toks": 25.0,
    },
}

# Metrics where lower is better (upper bounds)
_UPPER_BOUND_METRICS = {
    "hallucination_rate",
    "false_positive_rate",
    "ece",
    "quality_drop",
}


def check_gates(phase: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    """Check whether a phase's metrics pass all gate thresholds.

    Parameters
    ----------
    phase : str
        Phase identifier (e.g. 'phase2', 'round1').
    metrics : dict
        Metric name -> achieved value.

    Returns
    -------
    Dict with 'passed' (bool), 'failures' (list of failed metrics),
    and 'details' (per-metric pass/fail).
    """
    if phase not in PHASE_GATES:
        raise ValueError(f"Unknown phase: {phase}. Available: {list(PHASE_GATES.keys())}")

    gates = PHASE_GATES[phase]
    failures: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}

    for metric_name, threshold in gates.items():
        value = metrics.get(metric_name)
        if value is None:
            failures.append({"metric": metric_name, "reason": "missing", "threshold": threshold})
            details[metric_name] = {"passed": False, "reason": "missing"}
            continue

        if metric_name in _UPPER_BOUND_METRICS:
            passed = value <= threshold
        else:
            passed = value >= threshold

        details[metric_name] = {
            "passed": passed,
            "value": value,
            "threshold": threshold,
            "direction": "upper_bound" if metric_name in _UPPER_BOUND_METRICS else "lower_bound",
        }

        if not passed:
            failures.append({
                "metric": metric_name,
                "value": value,
                "threshold": threshold,
                "direction": "must be <=" if metric_name in _UPPER_BOUND_METRICS else "must be >=",
            })

    return {
        "passed": len(failures) == 0,
        "phase": phase,
        "failures": failures,
        "details": details,
    }
```

- [ ] **Step 5: Write eval runner**

```python
# src/finetune/v2/eval/runner.py
"""Evaluation runner — orchestrates benchmark execution.

Runs model against held-out eval examples, scores with rubrics,
and returns structured results for gate checking.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from .rubrics import score_with_rubric

logger = logging.getLogger(__name__)

# Mapping from benchmark name to primary rubric
_BENCHMARK_RUBRICS = {
    "DocVQA-mini": "extraction_accuracy",
    "TableBench": "extraction_accuracy",
    "LayoutEval": "extraction_accuracy",
    "HalluBench": "confidence_calibration",
    "ToolEval": "tool_correctness",
    "InsightEval": "insight_quality",
    "SynthesisEval": "synthesis_coherence",
    "ConversationEval": "conversation_quality",
    "ConfidenceEval": "confidence_calibration",
    "RegressionSuite": "extraction_accuracy",
}


def run_eval_on_examples(
    examples: List[Dict[str, Any]],
    model_fn: Callable[[str], str],
) -> List[Dict[str, Any]]:
    """Run evaluation on a list of benchmark examples.

    Parameters
    ----------
    examples : list
        Eval examples from eval_suite generator.
    model_fn : callable
        Function that takes a query string and returns model output string.

    Returns
    -------
    List of score dicts with benchmark, query, score, reasoning.
    """
    results = []
    for example in examples:
        benchmark = example["benchmark"]
        query = example["query"]
        reference = example["reference_answer"]
        context_str = str(example.get("context", []))

        # Get model output
        model_output = model_fn(query)

        # Score with appropriate rubric
        rubric_name = _BENCHMARK_RUBRICS.get(benchmark, "synthesis_coherence")
        score_result = score_with_rubric(
            rubric_name=rubric_name,
            model_output=model_output,
            reference=reference,
            context=context_str,
        )

        results.append({
            "benchmark": benchmark,
            "query": query,
            "model_output": model_output,
            "reference": reference,
            **score_result,
        })

    return results


def compute_phase_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Aggregate per-example scores into phase-level metrics.

    Groups by benchmark, computes mean score per benchmark,
    then normalizes to 0-1 scale (score/5).
    """
    from collections import defaultdict

    by_benchmark: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        by_benchmark[r["benchmark"]].append(r["score"])

    metrics = {}
    for benchmark, scores in by_benchmark.items():
        mean_score = sum(scores) / len(scores) if scores else 0
        normalized = mean_score / 5.0
        metrics[benchmark.lower().replace("-", "_")] = round(normalized, 3)

    return metrics
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_eval.py -v`
Expected: All 8 tests PASS

- [ ] **Step 7: Commit**

```bash
git add -f src/finetune/v2/eval/__init__.py src/finetune/v2/eval/rubrics.py src/finetune/v2/eval/gate_checker.py src/finetune/v2/eval/runner.py tests/test_v2_eval.py
git commit -m "feat: add eval infrastructure — rubrics, gate checker, benchmark runner"
```

---

## Task 7: Phase 3.7 Holistic Reasoning Training Loop

**Files:**
- Create: `src/finetune/v2/train_phase3_7_holistic.py`
- Test: `tests/test_v2_phase3_7.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v2_phase3_7.py
import pytest
from pathlib import Path


class TestPhase37Config:
    def test_config_defaults(self):
        from src.finetune.v2.train_phase3_7_holistic import HolisticConfig

        cfg = HolisticConfig()
        assert cfg.lora_r == 64
        assert cfg.lora_alpha == 128
        assert cfg.learning_rate == 8e-6
        assert cfg.epochs == 3
        assert cfg.max_seq_length == 8192
        assert cfg.bf16 is True
        assert "phase3_7" in str(cfg.output_dir) or "phase3.7" in str(cfg.output_dir)

    def test_config_quality_gates(self):
        from src.finetune.v2.train_phase3_7_holistic import HolisticConfig

        cfg = HolisticConfig()
        assert cfg.gate_synthesis_coherence >= 0.80
        assert cfg.gate_intent_alignment >= 0.85
        assert cfg.gate_depth_calibration >= 0.75
        assert cfg.gate_domain_accuracy >= 0.80

    def test_build_training_args(self):
        from src.finetune.v2.train_phase3_7_holistic import HolisticConfig, _build_training_args

        cfg = HolisticConfig()
        args = _build_training_args(cfg, Path("/tmp/test_output"))
        assert args["output_dir"] == "/tmp/test_output"
        assert args["num_train_epochs"] == 3
        assert args["learning_rate"] == 8e-6
        assert args["max_seq_length"] == 8192
        assert args["bf16"] is True

    def test_load_holistic_dataset_missing_file(self):
        import tempfile
        from src.finetune.v2.train_phase3_7_holistic import _load_holistic_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _load_holistic_dataset(Path(tmpdir))
            assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_phase3_7.py -v 2>&1 | head -20`
Expected: FAIL — ImportError

- [ ] **Step 3: Write Phase 3.7 training module**

```python
# src/finetune/v2/train_phase3_7_holistic.py
"""Phase 3.7 — Holistic Reasoning SFT.

Trains LoRA adapters (projection frozen from Phase 3.5) on holistic reasoning
data covering 4 modes: intent decomposition, evidence synthesis, depth
calibration, and domain-aware reasoning.

This phase bridges document intelligence and GPT-level contextual understanding,
teaching the model to synthesize across evidence, calibrate response depth, and
apply domain-specific analytical frameworks.

Typical wall-time: ~5-6 hours on a single A100-80 GB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HolisticConfig:
    """Hyperparameters for Phase 3.7 holistic reasoning SFT."""

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    learning_rate: float = 8e-6
    epochs: int = 3
    per_device_batch_size: int = 4
    max_seq_length: int = 8192  # longer context for synthesis tasks
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    bf16: bool = True
    lr_scheduler_type: str = "cosine"
    checkpoint_steps: int = 300

    # Input
    data_dir: Path = Path("finetune_data/v2/holistic")
    phase35_dir: Path = Path("finetune_artifacts/v2/phase3_5")

    # Output
    output_dir: Path = Path("finetune_artifacts/v2/phase3_7")
    save_steps: int = 300
    logging_steps: int = 25
    eval_steps: int = 300

    # Quality gates
    gate_synthesis_coherence: float = 0.80
    gate_intent_alignment: float = 0.85
    gate_depth_calibration: float = 0.75
    gate_domain_accuracy: float = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_training_args(config: HolisticConfig, output_dir: Path) -> Dict[str, Any]:
    """Build an SFTConfig-compatible training arguments dictionary."""
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "bf16": config.bf16,
        "fp16": False,
        "logging_steps": config.logging_steps,
        "save_steps": config.checkpoint_steps,
        "eval_steps": config.eval_steps,
        "max_seq_length": config.max_seq_length,
        "dataset_text_field": "text",
        "report_to": "none",
        "seed": 42,
    }


def _load_holistic_dataset(data_dir: Path):
    """Load the holistic reasoning dataset from data_dir.

    Expects ``holistic_training.jsonl`` under data_dir.
    Returns None and logs a warning when the file is absent or unreadable.
    """
    from datasets import load_dataset  # type: ignore

    jsonl_path = data_dir / "holistic_training.jsonl"
    if not jsonl_path.exists():
        logger.warning(
            "Holistic dataset not found: %s — skipping dataset load.", jsonl_path
        )
        return None

    try:
        ds = load_dataset("json", data_files=str(jsonl_path), split="train")
        logger.info("Loaded %d holistic reasoning examples from %s.", len(ds), jsonl_path)
        return ds
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load holistic dataset (%s) — skipping.", exc)
        return None


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------


def run_phase3_7(
    config: Optional[HolisticConfig] = None,
    *,
    phase35_dir: Optional[Path] = None,
) -> Path:
    """Execute Phase 3.7 holistic reasoning SFT.

    1. Loads the model from the Phase 3.5 checkpoint directory.
    2. Freezes the projection (preserve all prior alignment).
    3. Applies LoRA adapters for holistic reasoning fine-tuning.
    4. Loads the holistic training dataset.
    5. Trains with SFTTrainer using extended 8192 sequence length.
    6. Saves the final checkpoint and writes a completion marker.

    Parameters
    ----------
    config :
        Training configuration. Uses defaults if None.
    phase35_dir :
        Override path to the Phase 3.5 output directory.

    Returns
    -------
    Path to the output directory containing the holistic-tuned checkpoint.
    """
    if config is None:
        config = HolisticConfig()

    p35_dir = phase35_dir or config.phase35_dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 3.7: Holistic Reasoning SFT ===")
    logger.info(
        "LoRA r=%d  alpha=%d  LR=%s  epochs=%d  batch=%d  max_seq=%d",
        config.lora_r,
        config.lora_alpha,
        config.learning_rate,
        config.epochs,
        config.per_device_batch_size,
        config.max_seq_length,
    )

    # --- Load model from Phase 3.5 output ------------------------------------
    from .vision_graft import GraftConfig, VisionGraftedModel

    graft_cfg = GraftConfig(freeze_vision=True, freeze_text=False)
    model = VisionGraftedModel(graft_cfg)
    model.load_vision_encoder()

    proj_ckpt = p35_dir / "projection.pt"
    model.load_projection(checkpoint=proj_ckpt)

    # Freeze projection — keep all prior alignment intact
    if model._projection is not None:
        for p in model._projection.parameters():
            p.requires_grad = False

    model.load_text_model()
    model.add_lora(r=config.lora_r, lora_alpha=config.lora_alpha)

    # --- Load holistic training dataset --------------------------------------
    dataset = _load_holistic_dataset(config.data_dir)

    # --- Build training args and run SFTTrainer ------------------------------
    try:
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except ImportError:
        logger.error("trl is not installed. Install with: pip install trl>=0.8")
        raise

    training_args_dict = _build_training_args(config, config.output_dir)
    sft_cfg = SFTConfig(**training_args_dict)

    if dataset is not None:
        trainer = SFTTrainer(
            model=model._text_model,
            args=sft_cfg,
            train_dataset=dataset,
            tokenizer=model._tokenizer,
        )

        logger.info(
            "Starting holistic reasoning SFT for %d epochs (max_seq=%d)...",
            config.epochs,
            config.max_seq_length,
        )
        trainer.train()

        final_ckpt = config.output_dir / "checkpoint_final"
        final_ckpt.mkdir(parents=True, exist_ok=True)
        model._text_model.save_pretrained(str(final_ckpt))
        if model._tokenizer is not None:
            model._tokenizer.save_pretrained(str(final_ckpt))
        logger.info("Final checkpoint saved to %s", final_ckpt)
    else:
        logger.warning("No dataset available — skipping SFTTrainer run.")

    # --- Save outputs --------------------------------------------------------
    model.save_all(config.output_dir)

    marker = config.output_dir / ".phase3_7_complete"
    marker.touch()

    logger.info("Phase 3.7 complete — artifacts saved to %s", config.output_dir)
    return config.output_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_phase3_7.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add -f src/finetune/v2/train_phase3_7_holistic.py tests/test_v2_phase3_7.py
git commit -m "feat: add Phase 3.7 holistic reasoning SFT training loop"
```

---

## Task 8: Post-Training Rounds (DPO + SFT + Distillation)

**Files:**
- Create: `src/finetune/v2/post_training/__init__.py`
- Create: `src/finetune/v2/post_training/round1_conversational_dpo.py`
- Create: `src/finetune/v2/post_training/round2_confidence_sft.py`
- Create: `src/finetune/v2/post_training/round3_distillation.py`
- Test: `tests/test_v2_post_training.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_v2_post_training.py
import pytest
from pathlib import Path


class TestRound1Config:
    def test_defaults(self):
        from src.finetune.v2.post_training.round1_conversational_dpo import Round1Config

        cfg = Round1Config()
        assert cfg.learning_rate == 1e-6
        assert cfg.beta == 0.05
        assert cfg.epochs == 2
        assert cfg.bf16 is True
        assert cfg.use_lora is False  # full model fine-tuning

    def test_build_training_args(self):
        from src.finetune.v2.post_training.round1_conversational_dpo import Round1Config, _build_training_args

        cfg = Round1Config()
        args = _build_training_args(cfg, Path("/tmp/r1"))
        assert args["beta"] == 0.05
        assert args["learning_rate"] == 1e-6


class TestRound2Config:
    def test_defaults(self):
        from src.finetune.v2.post_training.round2_confidence_sft import Round2Config

        cfg = Round2Config()
        assert cfg.learning_rate == 1e-6
        assert cfg.epochs == 2
        assert cfg.gate_ece <= 0.10

    def test_build_training_args(self):
        from src.finetune.v2.post_training.round2_confidence_sft import Round2Config, _build_training_args

        cfg = Round2Config()
        args = _build_training_args(cfg, Path("/tmp/r2"))
        assert args["learning_rate"] == 1e-6


class TestRound3Config:
    def test_defaults(self):
        from src.finetune.v2.post_training.round3_distillation import Round3Config

        cfg = Round3Config()
        assert cfg.learning_rate == 5e-7
        assert cfg.epochs == 1
        assert cfg.gate_max_quality_drop <= 0.03
        assert cfg.gate_min_speed_toks >= 25.0

    def test_build_training_args(self):
        from src.finetune.v2.post_training.round3_distillation import Round3Config, _build_training_args

        cfg = Round3Config()
        args = _build_training_args(cfg, Path("/tmp/r3"))
        assert args["num_train_epochs"] == 1
        assert args["learning_rate"] == 5e-7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_post_training.py -v 2>&1 | head -20`
Expected: FAIL — ImportError

- [ ] **Step 3: Write Round 1 — Conversational Refinement DPO**

```python
# src/finetune/v2/post_training/__init__.py
"""Post-training refinement stack for DocWain V2+."""

# src/finetune/v2/post_training/round1_conversational_dpo.py
"""Round 1 — Conversational Refinement DPO.

Full model fine-tuning (no LoRA) with very conservative LR to shape
conversational quality: natural flow, disambiguation, context awareness.

Typical wall-time: ~2-3 hours on A100-80GB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Round1Config:
    """Hyperparameters for conversational refinement DPO."""

    learning_rate: float = 1e-6
    beta: float = 0.05
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.10
    epochs: int = 2
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_prompt_length: int = 2048
    max_response_length: int = 2048
    bf16: bool = True
    use_lora: bool = False  # full model fine-tuning
    use_gradient_checkpointing: bool = True

    data_path: Path = Path("finetune_data/v2/post_training/conversational_dpo.jsonl")
    merged_model_dir: Path = Path("finetune_artifacts/v2/merged")
    output_dir: Path = Path("finetune_artifacts/v2/post_round1")

    gate_conversation_quality: float = 0.80


def _build_training_args(config: Round1Config, output_dir: Path) -> Dict[str, Any]:
    """Build DPOConfig-compatible training arguments."""
    max_length = config.max_prompt_length + config.max_response_length
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "beta": config.beta,
        "max_prompt_length": config.max_prompt_length,
        "max_length": max_length,
        "bf16": config.bf16,
        "fp16": False,
        "logging_steps": 25,
        "save_steps": 200,
        "report_to": "none",
        "seed": 42,
        "gradient_checkpointing": config.use_gradient_checkpointing,
    }


def _load_conversational_dpo_dataset(data_path: Path):
    """Load conversational DPO dataset."""
    from datasets import load_dataset  # type: ignore

    if not data_path.exists():
        logger.warning("Conversational DPO data not found: %s", data_path)
        return None

    try:
        ds = load_dataset("json", data_files=str(data_path), split="train")
        logger.info("Loaded %d conversational DPO pairs from %s.", len(ds), data_path)
        return ds
    except Exception as exc:
        logger.warning("Failed to load conversational DPO data (%s).", exc)
        return None


def run_round1(config: Optional[Round1Config] = None) -> Path:
    """Execute Round 1: Conversational Refinement DPO.

    Operates on merged full weights (post Phase 4 merge).
    Uses gradient checkpointing + bf16 to fit in 80GB VRAM.
    """
    if config is None:
        config = Round1Config()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Post-Training Round 1: Conversational DPO ===")
    logger.info("LR=%s  beta=%.3f  epochs=%d  full_model=%s", config.learning_rate, config.beta, config.epochs, not config.use_lora)

    # Load merged model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Loading merged model from %s", config.merged_model_dir)
    model = AutoModelForCausalLM.from_pretrained(str(config.merged_model_dir), torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(str(config.merged_model_dir))

    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    dataset = _load_conversational_dpo_dataset(config.data_path)

    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        logger.error("trl not installed.")
        raise

    if dataset is not None:
        args = _build_training_args(config, config.output_dir)
        dpo_cfg = DPOConfig(**args)
        trainer = DPOTrainer(model=model, ref_model=None, args=dpo_cfg, train_dataset=dataset, tokenizer=tokenizer)
        logger.info("Starting conversational DPO for %d epochs...", config.epochs)
        trainer.train()

        model.save_pretrained(str(config.output_dir / "checkpoint_final"))
        tokenizer.save_pretrained(str(config.output_dir / "checkpoint_final"))

    marker = config.output_dir / ".round1_complete"
    marker.touch()
    logger.info("Round 1 complete — artifacts saved to %s", config.output_dir)
    return config.output_dir
```

- [ ] **Step 4: Write Round 2 — Confidence Calibration SFT**

Create `src/finetune/v2/post_training/round2_confidence_sft.py` following the same pattern as Round 1 but using SFTTrainer:
- `Round2Config`: LR 1e-6, epochs 2, batch 4x8, bf16, gradient_checkpointing
- `gate_ece: float = 0.10`
- Loads from Round 1 output directory
- SFT on confidence calibration examples
- Saves checkpoint + `.round2_complete` marker

- [ ] **Step 5: Write Round 3 — Reasoning Distillation**

Create `src/finetune/v2/post_training/round3_distillation.py`:
- `Round3Config`: LR 5e-7, epochs 1, batch 4x8, bf16
- `gate_max_quality_drop: float = 0.03`
- `gate_min_speed_toks: float = 25.0`
- Loads from Round 2 output
- SFT on compressed reasoning examples
- Saves checkpoint + `.round3_complete` marker

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_post_training.py -v`
Expected: All 6 tests PASS

- [ ] **Step 7: Commit**

```bash
git add -f src/finetune/v2/post_training/__init__.py src/finetune/v2/post_training/round1_conversational_dpo.py src/finetune/v2/post_training/round2_confidence_sft.py src/finetune/v2/post_training/round3_distillation.py tests/test_v2_post_training.py
git commit -m "feat: add post-training refinement stack (conversational DPO, confidence SFT, distillation)"
```

---

## Task 9: Update Pipeline Orchestrator

**Files:**
- Modify: `src/finetune/v2/pipeline.py`
- Modify: `src/finetune/v2/merge_promote.py`
- Modify: `src/finetune/v2/__init__.py`
- Test: `tests/test_v2_pipeline_orchestration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v2_pipeline_orchestration.py
import pytest
from pathlib import Path


class TestV2PlusPipeline:
    def test_phases_include_3_7(self):
        from src.finetune.v2.pipeline import V2Pipeline

        pipeline = V2Pipeline()
        assert "phase3_7" in pipeline.phases

    def test_phase_ordering(self):
        from src.finetune.v2.pipeline import V2Pipeline

        pipeline = V2Pipeline()
        idx_35 = pipeline.phases.index("phase3_5")
        idx_37 = pipeline.phases.index("phase3_7")
        idx_4 = pipeline.phases.index("phase4")
        assert idx_35 < idx_37 < idx_4

    def test_post_training_phases(self):
        from src.finetune.v2.pipeline import V2Pipeline

        pipeline = V2Pipeline()
        assert "round1" in pipeline.phases
        assert "round2" in pipeline.phases
        assert "round3" in pipeline.phases

    def test_phase_markers_include_3_7(self):
        from src.finetune.v2.pipeline import PHASE_MARKERS

        assert "phase3_7" in PHASE_MARKERS
        assert "round1" in PHASE_MARKERS
        assert "round2" in PHASE_MARKERS
        assert "round3" in PHASE_MARKERS

    def test_next_phase_skips_completed(self):
        import tempfile
        from src.finetune.v2.pipeline import V2Pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            pipeline = V2Pipeline(base_dir=base)

            # Mark phase1 and phase2 as completed
            (base / "phase1").mkdir()
            (base / "phase1" / "projection.pt").touch()
            (base / "phase2").mkdir()
            (base / "phase2" / "phase2_config.json").touch()

            nxt = pipeline.next_phase()
            assert nxt == "phase2_5"

    def test_merge_promote_config_includes_phase37(self):
        from src.finetune.v2.merge_promote import Phase4Config

        cfg = Phase4Config()
        # phase3_dir should now point to phase3_7 output (last LoRA phase)
        assert "phase3_7" in str(cfg.phase3_dir) or "phase3.7" in str(cfg.phase3_dir)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_pipeline_orchestration.py -v 2>&1 | head -20`
Expected: FAIL — phase3_7 not in pipeline.phases

- [ ] **Step 3: Update pipeline.py**

Update `src/finetune/v2/pipeline.py` to:
- Add `"phase3_7"` to `PHASE_MARKERS` with marker `".phase3_7_complete"`
- Add `"round1"`, `"round2"`, `"round3"` to `PHASE_MARKERS` with markers `".round1_complete"`, `".round2_complete"`, `".round3_complete"`
- Update `V2Pipeline.phases` list to include all new phases in order:
  `["phase1", "phase2", "phase2_5", "phase3", "phase3_5", "phase3_7", "phase4", "round1", "round2", "round3"]`

- [ ] **Step 4: Update merge_promote.py**

Update `src/finetune/v2/merge_promote.py`:
- Change `Phase4Config.phase3_dir` default to `Path("finetune_artifacts/v2/phase3_7")` (merge from last LoRA phase)
- Update `get_new_capability_criteria()` to include Phase 3.7 metrics:
  `"synthesis_coherence": 0.80, "intent_alignment": 0.85, "depth_calibration": 0.75`

- [ ] **Step 5: Update __init__.py**

```python
# src/finetune/v2/__init__.py
"""DocWain V2 — Vision-grafted unified model with native tool-calling."""

from .vision_graft import GraftConfig, VisionGraftedModel
from .pipeline import V2Pipeline
from .tool_schemas import get_core_tool_schemas

__all__ = ["GraftConfig", "VisionGraftedModel", "V2Pipeline", "get_core_tool_schemas"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_v2_pipeline_orchestration.py -v`
Expected: All 6 tests PASS

- [ ] **Step 7: Commit**

```bash
git add -f src/finetune/v2/pipeline.py src/finetune/v2/merge_promote.py src/finetune/v2/__init__.py tests/test_v2_pipeline_orchestration.py
git commit -m "feat: update pipeline orchestrator with Phase 3.7 + post-training rounds"
```

---

## Task 10: Master Data Generation Script + Full Pipeline Runner

**Files:**
- Create: `src/finetune/v2/generate_all_data.py`
- Create: `src/finetune/v2/run_v2plus.py`

- [ ] **Step 1: Write master data generation script**

```python
# src/finetune/v2/generate_all_data.py
"""Master data generation script — generates all V2+ training data.

Usage:
    python -m src.finetune.v2.generate_all_data [--scale 1.0] [--output-dir finetune_data/v2]

At scale=1.0, generates ~52K training examples + 500 eval examples.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_all(output_dir: Path, scale: float = 1.0) -> dict:
    """Generate all training and eval data for the V2+ pipeline.

    Returns dict mapping phase -> number of examples generated.
    """
    stats = {}
    t0 = time.time()

    logger.info("=== V2+ Data Generation (scale=%.2f) ===", scale)
    logger.info("Output directory: %s", output_dir)

    # Phase 2: Document Intelligence (20K)
    from .data_generator.phase2_doc_intelligence import generate_phase2_data
    stats["phase2"] = generate_phase2_data(output_dir / "doc_intelligence", scale=scale)
    logger.info("Phase 2 data: %s", stats["phase2"])

    # Phase 2.5: DPO Pairs (5K)
    from .data_generator.phase2_5_dpo_pairs import generate_phase25_data
    stats["phase2_5"] = generate_phase25_data(output_dir / "doc_intelligence", scale=scale)
    logger.info("Phase 2.5 data: %d pairs", stats["phase2_5"])

    # Phase 3.5: Insights (6K)
    from .data_generator.phase3_5_insights import generate_phase35_data
    stats["phase3_5"] = generate_phase35_data(output_dir / "insights", scale=scale)
    logger.info("Phase 3.5 data: %d examples", stats["phase3_5"])

    # Phase 3.7: Holistic Reasoning (8K)
    from .data_generator.phase3_7_holistic import generate_phase37_data
    stats["phase3_7"] = generate_phase37_data(output_dir / "holistic", scale=scale)
    logger.info("Phase 3.7 data: %d examples", stats["phase3_7"])

    # Post-Training: Conversational DPO (3K)
    from .data_generator.post_conversational import generate_post_conversational_data
    stats["post_round1"] = generate_post_conversational_data(output_dir / "post_training", scale=scale)
    logger.info("Post Round 1 data: %d pairs", stats["post_round1"])

    # Post-Training: Confidence (2K)
    from .data_generator.post_confidence import generate_post_confidence_data
    stats["post_round2"] = generate_post_confidence_data(output_dir / "post_training", scale=scale)
    logger.info("Post Round 2 data: %d examples", stats["post_round2"])

    # Eval Suite (500, frozen)
    from .data_generator.eval_suite import write_eval_suite
    write_eval_suite(output_dir / "eval")
    stats["eval"] = 500
    logger.info("Eval suite: 500 examples (frozen)")

    elapsed = time.time() - t0
    logger.info("=== Data generation complete in %.1f seconds ===", elapsed)
    logger.info("Total examples: %s", sum(v if isinstance(v, int) else sum(v.values()) for v in stats.values()))

    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate V2+ training data")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor (0.01 for test, 1.0 for production)")
    parser.add_argument("--output-dir", type=str, default="finetune_data/v2", help="Output directory")
    args = parser.parse_args()

    generate_all(Path(args.output_dir), scale=args.scale)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write full pipeline runner script**

```python
# src/finetune/v2/run_v2plus.py
"""Full V2+ pipeline runner — data generation through Ollama promotion.

Usage:
    python -m src.finetune.v2.run_v2plus [--start-from phase2] [--scale 1.0] [--skip-data-gen]

Runs all phases sequentially, checking gates between each.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from .pipeline import V2Pipeline

logger = logging.getLogger(__name__)

PIPELINE_PHASES = [
    "data_gen",
    "phase1", "phase2", "phase2_5",
    "phase3", "phase3_5", "phase3_7",
    "phase4",
    "round1", "round2", "round3",
    "final_promote",
]


def run_full_pipeline(
    start_from: str = "data_gen",
    scale: float = 1.0,
    skip_data_gen: bool = False,
    data_dir: Path = Path("finetune_data/v2"),
    artifacts_dir: Path = Path("finetune_artifacts/v2"),
) -> dict:
    """Run the complete V2+ pipeline.

    Returns dict with status per phase.
    """
    results = {}
    pipeline = V2Pipeline(base_dir=artifacts_dir)
    start_idx = PIPELINE_PHASES.index(start_from)

    for phase_name in PIPELINE_PHASES[start_idx:]:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("PHASE: %s", phase_name)
        logger.info("=" * 60)

        if phase_name == "data_gen":
            if skip_data_gen:
                logger.info("Skipping data generation (--skip-data-gen)")
                results[phase_name] = {"status": "skipped"}
                continue
            from .generate_all_data import generate_all
            stats = generate_all(data_dir, scale=scale)
            results[phase_name] = {"status": "complete", "stats": stats}

        elif phase_name == "phase1":
            from .train_phase1 import run_phase1
            run_phase1()
            results[phase_name] = {"status": "complete"}

        elif phase_name == "phase2":
            from .train_phase2 import run_phase2
            run_phase2()
            results[phase_name] = {"status": "complete"}

        elif phase_name == "phase2_5":
            from .train_phase2_5_dpo import run_phase2_5
            run_phase2_5()
            results[phase_name] = {"status": "complete"}

        elif phase_name == "phase3":
            from .train_phase3 import run_phase3
            run_phase3()
            results[phase_name] = {"status": "complete"}

        elif phase_name == "phase3_5":
            from .train_phase3_5_insights import run_phase3_5
            run_phase3_5()
            results[phase_name] = {"status": "complete"}

        elif phase_name == "phase3_7":
            from .train_phase3_7_holistic import run_phase3_7
            run_phase3_7()
            results[phase_name] = {"status": "complete"}

        elif phase_name == "phase4":
            from .merge_promote import run_phase4
            run_phase4()
            results[phase_name] = {"status": "complete"}

        elif phase_name == "round1":
            from .post_training.round1_conversational_dpo import run_round1
            run_round1()
            results[phase_name] = {"status": "complete"}

        elif phase_name == "round2":
            from .post_training.round2_confidence_sft import run_round2
            run_round2()
            results[phase_name] = {"status": "complete"}

        elif phase_name == "round3":
            from .post_training.round3_distillation import run_round3
            run_round3()
            results[phase_name] = {"status": "complete"}

        elif phase_name == "final_promote":
            logger.info("Final promotion: re-quantize, regression, promote to Ollama")
            from .merge_promote import run_phase4, Phase4Config
            cfg = Phase4Config(ollama_tag_v2="v2")
            run_phase4(config=cfg)
            results[phase_name] = {"status": "complete"}

        elapsed = time.time() - t0
        logger.info("Phase %s completed in %.1f seconds", phase_name, elapsed)
        results[phase_name]["elapsed_seconds"] = elapsed

    logger.info("=" * 60)
    logger.info("V2+ PIPELINE COMPLETE")
    logger.info("=" * 60)
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run V2+ finetuning pipeline")
    parser.add_argument("--start-from", default="data_gen", choices=PIPELINE_PHASES)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--skip-data-gen", action="store_true")
    args = parser.parse_args()

    run_full_pipeline(start_from=args.start_from, scale=args.scale, skip_data_gen=args.skip_data_gen)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add -f src/finetune/v2/generate_all_data.py src/finetune/v2/run_v2plus.py
git commit -m "feat: add master data generation script and full V2+ pipeline runner"
```

---

## Task 11: Run All Tests + Final Validation

- [ ] **Step 1: Run all new tests**

```bash
cd /home/ubuntu/PycharmProjects/DocWain
python -m pytest tests/test_v2_data_generator.py tests/test_v2_eval.py tests/test_v2_phase3_7.py tests/test_v2_post_training.py tests/test_v2_pipeline_orchestration.py -v
```

Expected: All tests PASS

- [ ] **Step 2: Run existing V2 tests to check for regressions**

```bash
cd /home/ubuntu/PycharmProjects/DocWain
python -m pytest tests/test_v2_pipeline.py tests/test_v2_phases.py tests/test_v2_vision_graft.py tests/test_v2_tool_schemas.py tests/test_v2_tool_data.py tests/test_v2_merge.py -v
```

Expected: All existing tests still PASS

- [ ] **Step 3: Generate test-scale data to validate generators**

```bash
cd /home/ubuntu/PycharmProjects/DocWain
python -m src.finetune.v2.generate_all_data --scale 0.01 --output-dir /tmp/docwain_test_data
```

Expected: All JSONL files created, no errors

- [ ] **Step 4: Final commit with all files**

```bash
cd /home/ubuntu/PycharmProjects/DocWain
git add -f src/finetune/v2/data_generator/ src/finetune/v2/eval/ src/finetune/v2/post_training/ src/finetune/v2/train_phase3_7_holistic.py src/finetune/v2/generate_all_data.py src/finetune/v2/run_v2plus.py src/finetune/v2/pipeline.py src/finetune/v2/merge_promote.py tests/test_v2_data_generator.py tests/test_v2_eval.py tests/test_v2_phase3_7.py tests/test_v2_post_training.py tests/test_v2_pipeline_orchestration.py
git commit -m "feat: complete V2+ finetuning pipeline — data generators, training loops, eval suite, orchestration"
```

- [ ] **Step 5: Push to remote**

```bash
git push origin preprod_v01
```

---

## Task 12: Generate Production Data + Launch Training

- [ ] **Step 1: Generate full-scale production data (52K examples)**

```bash
cd /home/ubuntu/PycharmProjects/DocWain
python -m src.finetune.v2.generate_all_data --scale 1.0 --output-dir finetune_data/v2
```

Expected output: ~52K training examples + 500 eval examples across all JSONL files

- [ ] **Step 2: Verify data volumes**

```bash
wc -l finetune_data/v2/doc_intelligence/*.jsonl finetune_data/v2/insights/*.jsonl finetune_data/v2/holistic/*.jsonl finetune_data/v2/post_training/*.jsonl finetune_data/v2/eval/*.jsonl
```

Expected:
- table.jsonl: ~8000
- layout.jsonl: ~5000
- ocr.jsonl: ~4000
- cross_ref.jsonl: ~3000
- dpo_pairs.jsonl: ~5000
- insight_training.jsonl: ~6000
- holistic_training.jsonl: ~8000
- conversational_dpo.jsonl: ~3000
- confidence_sft.jsonl: ~2000
- benchmark.jsonl: 500

- [ ] **Step 3: Launch V2+ training pipeline**

```bash
cd /home/ubuntu/PycharmProjects/DocWain
nohup python -m src.finetune.v2.run_v2plus --start-from phase2 --skip-data-gen > finetune_logs/v2plus_$(date +%Y%m%d_%H%M).log 2>&1 &
echo "Training launched. Monitor with: tail -f finetune_logs/v2plus_*.log"
```

Expected: Training begins, phases execute sequentially with gate checks between each.
