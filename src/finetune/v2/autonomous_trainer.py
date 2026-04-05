"""DocWain V2 Autonomous Trainer — DEPRECATED.

Superseded by curriculum_trainer.py (unified curriculum approach).
See docs/superpowers/specs/2026-04-05-curriculum-training-redesign.md

Kept in tree for reference only.
"""

from __future__ import annotations

import warnings
warnings.warn(
    "autonomous_trainer.py is deprecated. Use curriculum_trainer.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Patch llm_blender compatibility with latest transformers (TRANSFORMERS_CACHE removed)
import transformers.utils.hub as _hub
if not hasattr(_hub, "TRANSFORMERS_CACHE"):
    import os as _os
    _hub.TRANSFORMERS_CACHE = _os.path.join(
        _os.path.expanduser("~"), ".cache", "huggingface", "hub"
    )

import importlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.finetune.v2.train_track import TrackTrainingConfig, train_track
from src.finetune.v2.eval.evaluator import TrackEvaluator, query_ollama
from src.finetune.v2.eval.rubrics import TRACK_SCORERS
from src.finetune.v2.eval.test_bank import get_test_bank
from src.finetune.v2.data_generator.base import (
    DOCWAIN_SYSTEM_PROMPT,
    JSONLWriter,
    format_sft_example,
    format_dpo_example,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Track definitions
# ---------------------------------------------------------------------------

TRACKS = [
    "excel_csv",
    "layout",
    "ocr_vision",
    "reasoning",
    "kg",
    "visualization",
]

TRACK_GENERATORS = {
    "excel_csv": "src.finetune.v2.data_generator.track1_excel_csv:generate_track1_data",
    "layout": "src.finetune.v2.data_generator.track2_layout:generate_track2_data",
    "ocr_vision": "src.finetune.v2.data_generator.track3_ocr_vision:generate_track3_data",
    "reasoning": "src.finetune.v2.data_generator.track4_reasoning:generate_track4_data",
    "kg": "src.finetune.v2.data_generator.track5_kg:generate_track5_data",
    "visualization": "src.finetune.v2.data_generator.track6_visualization:generate_track6_data",
}

# Quality gate: a track passes when avg eval score >= this threshold (1.0-5.0 scale)
TRACK_PASS_THRESHOLD = 4.0

# After this many iterations with no improvement, change strategy
STRATEGY_PIVOT_AFTER = 2

# Maximum iterations per track before forced progression
MAX_ITERATIONS_PER_TRACK = 20

# Ollama model names
OLLAMA_V2_WIP = "DHS/DocWain:v2-wip"
OLLAMA_V1 = "DHS/DocWain:v1"
OLLAMA_LATEST = "DHS/DocWain:latest"

# Regression thresholds
REGRESSION_OVERALL_MIN = 90.0
REGRESSION_CATEGORY_MIN = 85.0


# ---------------------------------------------------------------------------
# Strategy evolution
# ---------------------------------------------------------------------------

_STRATEGY_SEQUENCE = [
    {
        "name": "baseline",
        "description": "Standard SFT with default hyperparameters",
        "epochs": 3,
        "lr": 2e-5,
        "lora_r": 64,
        "use_dpo": False,
        "data_multiplier": 1.0,
    },
    {
        "name": "more_data",
        "description": "Double training data with targeted weak-area examples",
        "epochs": 3,
        "lr": 2e-5,
        "lora_r": 64,
        "use_dpo": False,
        "data_multiplier": 2.0,
    },
    {
        "name": "higher_rank",
        "description": "Increase LoRA rank for more capacity",
        "epochs": 4,
        "lr": 1.5e-5,
        "lora_r": 128,
        "use_dpo": False,
        "data_multiplier": 2.0,
    },
    {
        "name": "add_dpo",
        "description": "More data + lower LR for fine-grained learning",
        "epochs": 4,
        "lr": 1e-5,
        "lora_r": 64,
        "use_dpo": False,  # disabled: synthetic DPO pairs cause regression
        "data_multiplier": 3.0,
    },
    {
        "name": "aggressive",
        "description": "Higher rank + more epochs + lower LR + 3x data",
        "epochs": 5,
        "lr": 8e-6,
        "lora_r": 128,
        "use_dpo": False,  # disabled: synthetic DPO pairs cause regression
        "data_multiplier": 3.0,
    },
]


def _get_strategy(iteration: int, no_improve_count: int = 0) -> Dict[str, Any]:
    """Get the training strategy for a given iteration number (1-based).

    If no_improve_count >= 3, force advance to the next strategy early
    rather than waiting for STRATEGY_PIVOT_AFTER iterations.
    """
    base_idx = min((iteration - 1) // STRATEGY_PIVOT_AFTER, len(_STRATEGY_SEQUENCE) - 1)
    if no_improve_count >= 3:
        base_idx = min(base_idx + 1, len(_STRATEGY_SEQUENCE) - 1)
    return _STRATEGY_SEQUENCE[base_idx]


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def _load_generator(track: str):
    """Dynamically load and return the data generator function for a track.

    Returns the callable, or None if the module doesn't exist yet.
    """
    entry = TRACK_GENERATORS.get(track)
    if not entry:
        return None
    module_path, func_name = entry.rsplit(":", 1)
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)
    except (ImportError, AttributeError) as exc:
        logger.warning("Could not load generator for %s: %s", track, exc)
        return None


def _generate_fallback_sft_data(
    track: str,
    output_path: Path,
    count: int = 500,
    seed: int = 42,
    weak_areas: Optional[List[str]] = None,
) -> int:
    """Generate fallback SFT training data when the track-specific generator
    is not available.

    Uses template-based generation with the DocWain system prompt and
    track-specific query/answer patterns.

    Returns the number of examples written.
    """
    import random
    rng = random.Random(seed)

    templates = _get_track_templates(track)
    if not templates:
        logger.error("No templates available for track %s", track)
        return 0

    # If there are weak areas, weight them more heavily
    weighted_templates = list(templates)
    if weak_areas:
        for tpl in templates:
            if any(wa.lower() in tpl.get("category", "").lower() for wa in weak_areas):
                weighted_templates.extend([tpl] * 3)

    with JSONLWriter(output_path) as writer:
        for i in range(count):
            tpl = rng.choice(weighted_templates)
            query = tpl["query"].format(
                idx=i,
                seed=seed + i,
                domain=rng.choice(["financial", "legal", "hr", "medical", "insurance"]),
            )
            reasoning = tpl["reasoning"].format(idx=i, seed=seed + i)
            answer = tpl["answer"].format(idx=i, seed=seed + i)

            example = format_sft_example(query, reasoning, answer)
            writer.write(example)

    logger.info("Generated %d fallback SFT examples for %s at %s", count, track, output_path)
    return count


def _corrupt_reasoning(reasoning: str, rng) -> str:
    """Create a subtly worse version of the reasoning — still coherent but flawed."""
    import re
    corruptions = [
        # Drop a key reasoning step (truncate ~40% from the middle)
        lambda r: r[:len(r)//3] + " Therefore, I can provide the answer directly.",
        # Swap cause and effect
        lambda r: r.replace("I need to", "I will skip checking and just").replace(
            "identify", "assume"),
        # Add false confidence without analysis
        lambda r: "Based on the document, this is straightforward. " + r[len(r)//2:],
        # Remove quantitative reasoning
        lambda r: re.sub(r'\d+[\.\d]*', 'several', r)[:len(r)],
    ]
    return rng.choice(corruptions)(reasoning)


def _corrupt_answer(answer: str, rng) -> str:
    """Create a subtly wrong version of the answer — same format but with errors."""
    import re
    corruptions = [
        # Truncate table rows (incomplete extraction)
        lambda a: "\n".join(a.split("\n")[:len(a.split("\n"))//2 + 1]) + "\n\n*...remaining items omitted...*",
        # Swap numbers in the output
        lambda a: re.sub(r'(\d+\.\d{2})', lambda m: f"{float(m.group(1)) * 1.15:.2f}", a, count=3),
        # Remove markdown structure
        lambda a: re.sub(r'[|#*\-]', '', a).strip(),
        # Duplicate a section (copy-paste error)
        lambda a: a + "\n\n" + a[len(a)//2:] if len(a) > 100 else a + " " + a,
    ]
    return rng.choice(corruptions)(answer)


def _cleanup_old_iterations(track_dir: Path, history: dict, current_iter: int):
    """Remove old iteration dirs to prevent disk exhaustion.

    Keeps only the best iteration's merged_16bit and the immediately
    preceding iteration.  SFT/DPO checkpoints are always removed since
    the merged_16bit is the only artifact needed for continuation.
    """
    import shutil
    best_iter = history.get("best_iteration", 0)
    keep = {best_iter, current_iter - 1, current_iter}
    for child in sorted(track_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        try:
            iter_num = int(child.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if iter_num in keep:
            # Still prune SFT/DPO checkpoints — only merged_16bit matters
            for sub in ["sft_checkpoints", "dpo_checkpoints"]:
                p = child / "model" / sub
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)
            # Prune GGUF intermediates (keep only the final .gguf)
            gguf_dir = child / "model" / "gguf"
            if gguf_dir.exists():
                for f in gguf_dir.iterdir():
                    if f.name.endswith("-f16.gguf"):
                        f.unlink(missing_ok=True)
        else:
            shutil.rmtree(child, ignore_errors=True)
            logger.info("Cleaned up old iteration dir: %s", child)


def _generate_fallback_dpo_data(
    track: str,
    output_path: Path,
    count: int = 200,
    seed: int = 42,
) -> int:
    """Generate fallback DPO preference pairs."""
    import random
    rng = random.Random(seed)

    templates = _get_track_templates(track)
    if not templates:
        return 0

    with JSONLWriter(output_path) as writer:
        for i in range(count):
            tpl = rng.choice(templates)
            query = tpl["query"].format(
                idx=i, seed=seed + i,
                domain=rng.choice(["financial", "legal", "hr", "medical"]),
            )

            # Chosen: detailed, well-structured
            chosen_reasoning = tpl["reasoning"].format(idx=i, seed=seed + i)
            chosen_answer = tpl["answer"].format(idx=i, seed=seed + i)

            # Rejected: structurally similar but with subtle errors
            # (swapped values, truncated output, wrong calculations)
            rejected_reasoning = _corrupt_reasoning(chosen_reasoning, rng)
            rejected_answer = _corrupt_answer(chosen_answer, rng)

            example = format_dpo_example(
                query, chosen_reasoning, chosen_answer,
                rejected_reasoning, rejected_answer,
            )
            writer.write(example)

    logger.info("Generated %d fallback DPO pairs for %s at %s", count, track, output_path)
    return count


def _get_track_templates(track: str) -> List[Dict[str, str]]:
    """Return query/reasoning/answer templates for a given track."""

    if track == "excel_csv":
        return [
            {
                "category": "extraction",
                "query": "Extract all line items from this {domain} invoice and compute the total. Invoice #{idx}.",
                "reasoning": (
                    "I need to parse the invoice structure, identify line items with their "
                    "quantities and unit prices, extract each row, and compute the total by "
                    "summing quantity * unit_price for all line items."
                ),
                "answer": (
                    "## Invoice Line Items\n\n"
                    "| # | Description | Qty | Unit Price | Amount |\n"
                    "|---|-------------|-----|-----------|--------|\n"
                    "| 1 | Professional Services | 40 | $150.00 | $6,000.00 |\n"
                    "| 2 | Software License | 5 | $500.00 | $2,500.00 |\n"
                    "| 3 | Support & Maintenance | 1 | $1,200.00 | $1,200.00 |\n\n"
                    "**Total Amount: $9,700.00**"
                ),
            },
            {
                "category": "analysis",
                "query": "Analyze the spreadsheet data and identify the top vendors by total spend for {domain} department.",
                "reasoning": (
                    "I need to group the purchase data by vendor name, sum the amounts "
                    "for each vendor, sort descending, and present the top vendors with "
                    "their total spend amounts."
                ),
                "answer": (
                    "## Top Vendors by Total Spend\n\n"
                    "| Rank | Vendor | Total Spend | % of Total |\n"
                    "|------|--------|------------|------------|\n"
                    "| 1 | **Acme Corp** | **$45,200** | 32% |\n"
                    "| 2 | **TechVision Ltd** | **$28,750** | 20% |\n"
                    "| 3 | **Global Services** | **$22,100** | 16% |\n\n"
                    "These three vendors account for **68%** of total department spend."
                ),
            },
            {
                "category": "comparison",
                "query": "Compare the budget vs actual figures in this {domain} financial report.",
                "reasoning": (
                    "I need to extract both budget and actual columns, compute the "
                    "variance for each line item, identify significant over/under-spend "
                    "items, and flag any items exceeding 10% variance."
                ),
                "answer": (
                    "## Budget vs Actual Analysis\n\n"
                    "| Category | Budget | Actual | Variance | Status |\n"
                    "|----------|--------|--------|----------|--------|\n"
                    "| Personnel | $120,000 | $118,500 | -$1,500 (1.3%) | On Track |\n"
                    "| Technology | $45,000 | $52,300 | +$7,300 (16.2%) | **Over Budget** |\n"
                    "| Operations | $30,000 | $28,900 | -$1,100 (3.7%) | On Track |\n\n"
                    "**Key Finding:** Technology spend exceeds budget by **16.2%**, "
                    "driven by unplanned infrastructure upgrades."
                ),
            },
            {
                "category": "aggregation",
                "query": "Calculate the total revenue, total expenses, and net profit from this {domain} quarterly report spreadsheet.",
                "reasoning": (
                    "I need to identify the revenue rows and expense rows, sum each "
                    "category separately, then compute net profit as total revenue minus "
                    "total expenses. I should also compute margins and growth rates if "
                    "prior quarter data is available."
                ),
                "answer": (
                    "## Quarterly Financial Summary\n\n"
                    "| Metric | Q1 Amount | Q2 Amount | Change |\n"
                    "|--------|-----------|-----------|--------|\n"
                    "| **Total Revenue** | **$1,245,000** | **$1,382,500** | +11.0% |\n"
                    "| **Total Expenses** | **$987,000** | **$1,015,200** | +2.9% |\n"
                    "| **Net Profit** | **$258,000** | **$367,300** | +42.4% |\n"
                    "| **Profit Margin** | **20.7%** | **26.6%** | +5.9pp |\n\n"
                    "**Key Finding:** Revenue grew **11.0%** while expenses were contained "
                    "to **2.9%** growth, expanding profit margin by **5.9 percentage points**."
                ),
            },
            {
                "category": "multi_sheet_reasoning",
                "query": "Cross-reference the employee list in Sheet 1 with the payroll data in Sheet 2 to identify any discrepancies for {domain} department.",
                "reasoning": (
                    "I need to read employee names and IDs from Sheet 1, match them against "
                    "payroll records in Sheet 2 using employee ID as the key, then identify: "
                    "(1) employees in Sheet 1 but not in payroll, (2) payroll entries without "
                    "matching employees, (3) mismatched names or departments between sheets."
                ),
                "answer": (
                    "## Cross-Sheet Discrepancy Report\n\n"
                    "### Employees Missing from Payroll (Sheet 1 only)\n"
                    "| Employee ID | Name | Department | Status |\n"
                    "|-------------|------|------------|--------|\n"
                    "| EMP-0042 | **Sarah Chen** | Engineering | Not in payroll |\n"
                    "| EMP-0089 | **James Rivera** | Operations | Not in payroll |\n\n"
                    "### Payroll Entries Without Employee Record\n"
                    "| Employee ID | Payroll Name | Amount |\n"
                    "|-------------|-------------|--------|\n"
                    "| EMP-0107 | **K. Patel** | $4,200.00 |\n\n"
                    "### Data Mismatches\n"
                    "| Employee ID | Field | Sheet 1 | Sheet 2 |\n"
                    "|-------------|-------|---------|--------|\n"
                    "| EMP-0023 | Department | **Marketing** | **Sales** |\n\n"
                    "**Summary:** Found **2** missing payroll records, **1** orphan payroll "
                    "entry, and **1** department mismatch. Recommend HR review."
                ),
            },
            {
                "category": "data_type_handling",
                "query": "Parse and normalize the date formats, currency values, and percentage fields in this {domain} spreadsheet.",
                "reasoning": (
                    "I need to identify columns with mixed formats: dates may appear as "
                    "MM/DD/YYYY, DD-Mon-YYYY, or ISO format; currencies may have different "
                    "symbols ($, £, €) or no symbol; percentages may be decimals (0.15) or "
                    "strings (15%). I should normalize each to a consistent format and flag "
                    "any unparseable values."
                ),
                "answer": (
                    "## Data Normalization Report\n\n"
                    "### Date Fields (Column B: \"Transaction Date\")\n"
                    "- **Formats found:** MM/DD/YYYY (67%), DD-Mon-YYYY (28%), ISO 8601 (5%)\n"
                    "- **Normalized to:** YYYY-MM-DD (ISO 8601)\n"
                    "- **Unparseable:** Row 47 value \"TBD\", Row 112 value \"Q2 2024\"\n\n"
                    "### Currency Fields (Column E: \"Amount\")\n"
                    "| Original Format | Count | Example | Normalized |\n"
                    "|----------------|-------|---------|------------|\n"
                    "| $X,XXX.XX | 142 | $1,500.00 | 1500.00 USD |\n"
                    "| X,XXX | 23 | 2,300 | 2300.00 USD |\n"
                    "| £X,XXX.XX | 8 | £850.00 | 850.00 GBP |\n\n"
                    "### Percentage Fields (Column G: \"Growth Rate\")\n"
                    "- **Decimal format (0.15):** 45 values → converted to 15.0%\n"
                    "- **String format (15%):** 89 values → kept as 15.0%\n"
                    "- **Negative values:** 12 entries flagged (e.g., -3.2%)\n\n"
                    "**Summary:** Normalized **173** currency values, **134** percentages, "
                    "and **200** dates. Flagged **2** unparseable dates for manual review."
                ),
            },
            {
                "category": "formula_interpretation",
                "query": "Explain the formulas used in this {domain} financial model spreadsheet and verify the calculated totals.",
                "reasoning": (
                    "I need to identify cells containing formulas (SUM, VLOOKUP, IF, etc.), "
                    "trace their dependencies, verify the logic is correct, and check whether "
                    "the computed outputs match expected results. I should also flag any "
                    "circular references or broken formula chains."
                ),
                "answer": (
                    "## Formula Audit Results\n\n"
                    "### Formula Inventory\n"
                    "| Cell | Formula | Purpose | Status |\n"
                    "|------|---------|---------|--------|\n"
                    "| D15 | =SUM(D2:D14) | Total Revenue | **Correct** ($1,382,500) |\n"
                    "| D30 | =SUM(D16:D29) | Total Expenses | **Correct** ($1,015,200) |\n"
                    "| D32 | =D15-D30 | Net Profit | **Correct** ($367,300) |\n"
                    "| D33 | =D32/D15 | Profit Margin | **Correct** (26.6%) |\n"
                    "| E15 | =D15/C15-1 | Revenue Growth | **Error**: should be =(D15-C15)/C15 |\n\n"
                    "### Issues Found\n"
                    "- **Cell E15:** Growth formula divides by prior period then subtracts 1, "
                    "which gives the same result mathematically but may cause #DIV/0! if C15 is zero\n"
                    "- **Row 22:** VLOOKUP references Sheet3 which contains outdated Q1 rates\n\n"
                    "**Summary:** **4 of 5** formulas verified correct. **1** formula has a "
                    "potential division-by-zero risk. Cross-sheet reference on Row 22 uses stale data."
                ),
            },
        ]
    elif track == "layout":
        return [
            {
                "category": "structure",
                "query": "Analyze the document layout and identify all structural elements.",
                "reasoning": (
                    "I need to examine the document structure, identify headings at "
                    "different levels, paragraphs, tables, lists, figures, headers, "
                    "footers, and page numbers. I should note the hierarchical "
                    "relationship between sections."
                ),
                "answer": (
                    "## Document Structure Analysis\n\n"
                    "### Page Layout\n"
                    "- **Format:** Single-column, portrait orientation\n"
                    "- **Header:** Company logo (left), document title (center), date (right)\n"
                    "- **Footer:** Page number (center), confidentiality notice (right)\n\n"
                    "### Section Hierarchy\n"
                    "1. **Executive Summary** (Page 1, heading level 1)\n"
                    "   - Key findings paragraph\n"
                    "2. **Methodology** (Page 2, heading level 1)\n"
                    "   - 2.1 Data Collection (heading level 2)\n"
                    "   - 2.2 Analysis Framework (heading level 2)\n"
                    "3. **Results** (Pages 3-5, heading level 1)\n"
                    "   - Table 1: Summary Statistics (Page 3)\n"
                    "   - Figure 1: Trend Analysis (Page 4)\n\n"
                    "**Total elements:** 3 tables, 2 figures, 12 paragraphs, 4 bulleted lists"
                ),
            },
            {
                "category": "table_detection",
                "query": "Detect and describe all tables in this {domain} document.",
                "reasoning": (
                    "I need to scan each page for tabular structures, identify their "
                    "location, dimensions (rows x columns), headers, and content type."
                ),
                "answer": (
                    "## Tables Detected\n\n"
                    "### Table 1 (Page 2)\n"
                    "- **Location:** Center, spanning full width\n"
                    "- **Dimensions:** 8 rows x 5 columns\n"
                    "- **Headers:** ID, Description, Quantity, Unit Price, Total\n"
                    "- **Content:** Line item data with monetary values\n\n"
                    "### Table 2 (Page 4)\n"
                    "- **Location:** Top half of page\n"
                    "- **Dimensions:** 4 rows x 3 columns\n"
                    "- **Headers:** Metric, Current, Previous\n"
                    "- **Content:** Comparison metrics with percentage changes"
                ),
            },
        ]
    elif track == "ocr_vision":
        return [
            {
                "category": "text_extraction",
                "query": "Extract all text from this scanned {domain} document.",
                "reasoning": (
                    "I need to apply OCR to extract text from the scanned image, "
                    "preserving the document structure including headings, paragraphs, "
                    "and any tabular data. I should also identify and correct common "
                    "OCR errors."
                ),
                "answer": (
                    "<tool_call>\n"
                    '{"name": "ocr_extract", "arguments": {"page": 1, "enhance": true}}\n'
                    "</tool_call>\n\n"
                    "## Extracted Text\n\n"
                    "**Document Title:** Service Agreement\n"
                    "**Date:** January 15, 2024\n\n"
                    "This Service Agreement ('Agreement') is entered into between "
                    "**Party A** (the 'Provider') and **Party B** (the 'Client').\n\n"
                    "### Section 1: Scope of Services\n"
                    "The Provider shall deliver the following services..."
                ),
            },
            {
                "category": "correction",
                "query": "Identify and correct OCR errors in the extracted text from this {domain} document.",
                "reasoning": (
                    "I need to review the OCR output for common errors: character "
                    "substitutions (0/O, 1/l, rn/m), word boundary issues, formatting "
                    "artifacts, and domain-specific terminology that may be misread."
                ),
                "answer": (
                    "## OCR Corrections Applied\n\n"
                    "| Original (OCR) | Corrected | Type |\n"
                    "|----------------|-----------|------|\n"
                    "| 'lndemnification' | 'Indemnification' | Case error |\n"
                    "| '$8,5OO.00' | '$8,500.00' | Character substitution (O->0) |\n"
                    "| 'terrns' | 'terms' | Character merge (rn->m) |\n"
                    "| 'Iegal' | 'Legal' | Character substitution (I->L) |\n\n"
                    "**4 corrections** applied with **high confidence**."
                ),
            },
        ]
    elif track == "reasoning":
        return [
            {
                "category": "comparison",
                "query": "Compare the key terms across these two {domain} contracts and identify significant differences.",
                "reasoning": (
                    "I need to systematically compare the contracts across key dimensions: "
                    "parties, effective dates, term length, payment terms, liability caps, "
                    "indemnification, termination clauses, and governing law. I should "
                    "identify material differences that could affect the parties' positions."
                ),
                "answer": (
                    "## Contract Comparison Analysis\n\n"
                    "### Key Differences\n\n"
                    "| Criteria | Contract A | Contract B | Assessment |\n"
                    "|----------|-----------|-----------|------------|\n"
                    "| **Term** | 24 months | 36 months | B is 50% longer |\n"
                    "| **Liability Cap** | $500,000 | **Unlimited** | **Critical difference** |\n"
                    "| **Termination** | 30 days notice | 90 days notice | B requires more notice |\n"
                    "| **Governing Law** | England & Wales | Delaware, USA | Different jurisdictions |\n\n"
                    "### Risk Assessment\n"
                    "Contract B presents **significantly higher risk** because of the "
                    "unlimited liability provision in Section 8.2. This contrasts with "
                    "Contract A's reasonable $500,000 cap (Section 7.4).\n\n"
                    "**Recommendation:** Negotiate a liability cap in Contract B before execution."
                ),
            },
            {
                "category": "synthesis",
                "query": "Synthesize findings from all uploaded {domain} documents into an executive summary.",
                "reasoning": (
                    "I need to identify the key themes across all documents, extract "
                    "the most important data points, find patterns and contradictions, "
                    "and present a coherent narrative that captures the essential "
                    "information a decision-maker needs."
                ),
                "answer": (
                    "## Executive Summary\n\n"
                    "Analysis of **5 documents** reveals three key findings:\n\n"
                    "1. **Revenue growth is decelerating:** Q4 revenue of **$12.3M** "
                    "represents only **3.2% YoY growth**, down from **8.1%** in Q3 "
                    "(Source: Financial Statement, Page 4).\n\n"
                    "2. **Compliance gap identified:** The audit report (Page 12) flags "
                    "**3 material findings** that are not addressed in the current "
                    "compliance policy (Section 4.2).\n\n"
                    "3. **Vendor concentration risk:** **72%** of procurement spend is "
                    "concentrated with 2 vendors (Purchase Report, Page 7), exceeding "
                    "the policy threshold of 60%.\n\n"
                    "### Recommended Actions\n"
                    "- Address audit findings within 30 days\n"
                    "- Diversify vendor base to reduce concentration below 60%\n"
                    "- Investigate revenue deceleration drivers"
                ),
            },
        ]
    elif track == "kg":
        return [
            {
                "category": "entity_extraction",
                "query": "Extract all entities and relationships from this {domain} document.",
                "reasoning": (
                    "I need to identify named entities (people, organizations, dates, "
                    "monetary amounts, locations), classify their types, and map the "
                    "relationships between them (e.g., employment, contractual, "
                    "financial, hierarchical)."
                ),
                "answer": (
                    "<tool_call>\n"
                    '{"name": "extract_entities", "arguments": {"doc_id": "doc_001", '
                    '"entity_types": ["person", "organization", "date", "amount"]}}\n'
                    "</tool_call>\n\n"
                    "## Entity Map\n\n"
                    "### Entities\n"
                    "- **People:** John Smith (CEO), Sarah Johnson (CFO), Michael Lee (Legal Counsel)\n"
                    "- **Organizations:** Acme Corp (Party A), TechVision Ltd (Party B)\n"
                    "- **Dates:** Jan 1, 2024 (effective), Dec 31, 2026 (expiry)\n"
                    "- **Amounts:** $2.5M (contract value), $500K (liability cap)\n\n"
                    "### Relationships\n"
                    "- John Smith **represents** Acme Corp\n"
                    "- Acme Corp **contracted with** TechVision Ltd\n"
                    "- Contract **effective** Jan 1, 2024\n"
                    "- Liability **capped at** $500K"
                ),
            },
            {
                "category": "cross_reference",
                "query": "Find all cross-references between sections in this {domain} document.",
                "reasoning": (
                    "I need to scan the document for internal references (e.g., "
                    "'as described in Section 3.2', 'per Appendix B', 'subject to "
                    "clause 7.1'), map them as edges in a reference graph, and identify "
                    "any broken or circular references."
                ),
                "answer": (
                    "<tool_call>\n"
                    '{"name": "cross_reference", "arguments": {"doc_id": "doc_001", '
                    '"scope": "internal"}}\n'
                    "</tool_call>\n\n"
                    "## Cross-Reference Map\n\n"
                    "| Source | Target | Reference Type |\n"
                    "|--------|--------|---------------|\n"
                    "| Section 2.1 | Appendix A | Definition reference |\n"
                    "| Section 4.3 | Section 2.1 | Back-reference |\n"
                    "| Section 7.1 | Section 3.2 | Conditional dependency |\n"
                    "| Section 8.2 | Appendix C | Exhibit reference |\n\n"
                    "**No broken references detected.** All referenced sections exist."
                ),
            },
        ]
    elif track == "visualization":
        return [
            {
                "category": "bar_chart",
                "query": "Create a bar chart showing the quarterly revenue from this {domain} financial data.",
                "reasoning": (
                    "The user wants a bar chart for quarterly revenue data. I need to "
                    "extract the quarterly figures, present them in a table, and generate "
                    "a DOCWAIN_VIZ directive with chart_type 'bar'."
                ),
                "answer": (
                    "## Quarterly Revenue\n\n"
                    "| Quarter | Revenue |\n"
                    "|---------|--------|\n"
                    "| Q1 2024 | $2.8M |\n"
                    "| Q2 2024 | $3.1M |\n"
                    "| Q3 2024 | $3.5M |\n"
                    "| Q4 2024 | $3.2M |\n\n"
                    "Revenue peaked in **Q3 2024** at **$3.5M**, with a slight decline "
                    "in Q4.\n\n"
                    "<!--DOCWAIN_VIZ\n"
                    '{"chart_type": "bar", "title": "Quarterly Revenue 2024", '
                    '"labels": ["Q1", "Q2", "Q3", "Q4"], '
                    '"values": [2.8, 3.1, 3.5, 3.2], "unit": "$M"}\n'
                    "-->"
                ),
            },
            {
                "category": "line_chart",
                "query": "Plot the monthly trend of {domain} metrics over the past year.",
                "reasoning": (
                    "The user wants a line chart showing trends over time. I need to "
                    "extract monthly data points and generate a line chart visualization."
                ),
                "answer": (
                    "## Monthly Trend Analysis\n\n"
                    "The data shows a clear upward trend with seasonal variation:\n\n"
                    "- **Peak:** August at **$4.2M**\n"
                    "- **Trough:** February at **$2.1M**\n"
                    "- **YoY Growth:** **12.5%** overall\n\n"
                    "<!--DOCWAIN_VIZ\n"
                    '{"chart_type": "line", "title": "Monthly Revenue Trend", '
                    '"labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", '
                    '"Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], '
                    '"values": [2.3, 2.1, 2.5, 2.8, 3.1, 3.4, 3.8, 4.2, 3.9, 3.5, 3.2, 3.0], '
                    '"unit": "$M"}\n'
                    "-->"
                ),
            },
        ]
    else:
        return []


# ---------------------------------------------------------------------------
# Live test scenarios
# ---------------------------------------------------------------------------


def _build_live_test_scenarios() -> List[Dict[str, Any]]:
    """Build 20 diverse test scenarios spanning all 6 tracks."""
    return [
        # excel_csv (1-4)
        {
            "id": "live_01",
            "track": "excel_csv",
            "prompt": (
                "I have a spreadsheet with employee salaries across departments. "
                "Show me the average salary per department and identify any outliers."
            ),
            "check_keywords": ["department", "average", "salary", "outlier"],
            "check_table": True,
            "check_bold": True,
        },
        {
            "id": "live_02",
            "track": "excel_csv",
            "prompt": (
                "Extract all purchase orders from the uploaded procurement data "
                "and calculate the total spend per vendor."
            ),
            "check_keywords": ["vendor", "total", "spend", "purchase"],
            "check_table": True,
            "check_bold": True,
        },
        {
            "id": "live_03",
            "track": "excel_csv",
            "prompt": "Parse this CSV data and find duplicate invoice numbers.",
            "check_keywords": ["duplicate", "invoice"],
            "check_table": False,
            "check_bold": True,
        },
        # layout (4-7)
        {
            "id": "live_04",
            "track": "layout",
            "prompt": (
                "Analyze the structure of this 50-page legal contract and give me "
                "the section hierarchy."
            ),
            "check_keywords": ["section", "hierarchy", "heading"],
            "check_table": False,
            "check_bold": True,
        },
        {
            "id": "live_05",
            "track": "layout",
            "prompt": "How many tables and figures are in this document? Describe their locations.",
            "check_keywords": ["table", "figure", "page"],
            "check_table": False,
            "check_bold": True,
        },
        {
            "id": "live_06",
            "track": "layout",
            "prompt": "Extract the header and footer patterns across all pages.",
            "check_keywords": ["header", "footer", "page"],
            "check_table": False,
            "check_bold": True,
        },
        # ocr_vision (7-9)
        {
            "id": "live_07",
            "track": "ocr_vision",
            "prompt": "This is a scanned invoice. Extract all the text and line items.",
            "check_keywords": ["text", "extract", "invoice"],
            "check_table": True,
            "check_bold": True,
        },
        {
            "id": "live_08",
            "track": "ocr_vision",
            "prompt": "Read the handwritten notes on this document scan.",
            "check_keywords": ["handwritten", "text"],
            "check_table": False,
            "check_bold": False,
        },
        {
            "id": "live_09",
            "track": "ocr_vision",
            "prompt": "Check the OCR output for errors and correct them.",
            "check_keywords": ["correct", "error"],
            "check_table": False,
            "check_bold": True,
        },
        # reasoning (10-13)
        {
            "id": "live_10",
            "track": "reasoning",
            "prompt": (
                "I have two vendor proposals. Compare them on price, delivery timeline, "
                "support terms, and overall value. Which one should we choose?"
            ),
            "check_keywords": ["compare", "price", "recommendation"],
            "check_table": True,
            "check_bold": True,
        },
        {
            "id": "live_11",
            "track": "reasoning",
            "prompt": (
                "Cross-reference the audit findings with our compliance policies "
                "and flag any gaps."
            ),
            "check_keywords": ["audit", "compliance", "gap"],
            "check_table": False,
            "check_bold": True,
        },
        {
            "id": "live_12",
            "track": "reasoning",
            "prompt": (
                "Synthesize the information from all 5 uploaded documents into a "
                "risk assessment report."
            ),
            "check_keywords": ["risk", "assessment", "finding"],
            "check_table": False,
            "check_bold": True,
        },
        {
            "id": "live_13",
            "track": "reasoning",
            "prompt": (
                "Build a timeline of all key events mentioned across these documents "
                "and identify any contradictions in dates."
            ),
            "check_keywords": ["timeline", "event", "date"],
            "check_table": False,
            "check_bold": True,
        },
        # kg (14-16)
        {
            "id": "live_14",
            "track": "kg",
            "prompt": (
                "Map all the people, organizations, and relationships mentioned "
                "in this contract."
            ),
            "check_keywords": ["entity", "relationship"],
            "check_table": False,
            "check_bold": True,
        },
        {
            "id": "live_15",
            "track": "kg",
            "prompt": "Find all cross-references between sections in this legal document.",
            "check_keywords": ["reference", "section"],
            "check_table": True,
            "check_bold": True,
        },
        {
            "id": "live_16",
            "track": "kg",
            "prompt": "Extract the organizational hierarchy from this employee handbook.",
            "check_keywords": ["hierarchy", "organization", "department"],
            "check_table": False,
            "check_bold": True,
        },
        # visualization (17-20)
        {
            "id": "live_17",
            "track": "visualization",
            "prompt": "Show me a bar chart of quarterly revenue from the financial data.",
            "check_keywords": ["revenue", "quarter"],
            "check_viz": True,
            "check_bold": True,
            "check_table": False,
        },
        {
            "id": "live_18",
            "track": "visualization",
            "prompt": "Create a pie chart showing expense distribution by category.",
            "check_keywords": ["expense", "category"],
            "check_viz": True,
            "check_bold": True,
            "check_table": False,
        },
        {
            "id": "live_19",
            "track": "visualization",
            "prompt": "Plot the monthly headcount trend over the past year.",
            "check_keywords": ["headcount", "trend", "monthly"],
            "check_viz": True,
            "check_bold": True,
            "check_table": False,
        },
        {
            "id": "live_20",
            "track": "visualization",
            "prompt": "Compare department performance metrics using a radar chart.",
            "check_keywords": ["performance", "department"],
            "check_viz": True,
            "check_bold": True,
            "check_table": False,
        },
    ]


def _score_live_scenario(response: str, scenario: Dict[str, Any]) -> float:
    """Score a live test scenario response (0-100)."""
    if not response or len(response.strip()) < 20:
        return 0.0

    score = 0.0
    total_weight = 0.0

    # Keyword check (40 points)
    keywords = scenario.get("check_keywords", [])
    if keywords:
        total_weight += 40
        lower = response.lower()
        hits = sum(1 for kw in keywords if kw.lower() in lower)
        score += (hits / len(keywords)) * 40

    # Table check (20 points)
    if scenario.get("check_table", False):
        total_weight += 20
        if "|" in response and "---" in response:
            score += 20

    # Bold formatting (10 points)
    if scenario.get("check_bold", False):
        total_weight += 10
        if "**" in response:
            score += 10

    # Visualization directive (20 points)
    if scenario.get("check_viz", False):
        total_weight += 20
        if "<!--DOCWAIN_VIZ" in response:
            score += 20

    # Length and substance (10 points)
    total_weight += 10
    if len(response) > 100:
        score += 5
    if len(response) > 300:
        score += 5

    # Structure (10 points)
    total_weight += 10
    if "##" in response:
        score += 5
    if "\n" in response and len(response.splitlines()) > 3:
        score += 5

    # Normalize
    if total_weight > 0:
        return min((score / total_weight) * 100, 100.0)
    return 0.0


# ---------------------------------------------------------------------------
# AutonomousTrainer
# ---------------------------------------------------------------------------


class AutonomousTrainer:
    """Runs the full 6-track iterative training pipeline autonomously.

    State is persisted to disk after every iteration so the pipeline
    can resume from where it left off if interrupted.
    """

    ARTIFACT_DIR = Path("finetune_artifacts/v2_upgrade")
    STATE_FILE = ARTIFACT_DIR / "state.json"
    LOG_FILE = ARTIFACT_DIR / "training.log"
    REPORT_FILE = ARTIFACT_DIR / "final_report.md"

    def __init__(
        self,
        tracks: Optional[List[str]] = None,
        base_model: str = "unsloth/Qwen3-14B-bnb-4bit",
    ) -> None:
        self.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self.state = self._load_state()
        self.tracks = tracks or TRACKS
        self.base_model = base_model
        self.start_time = time.time()
        # Store timing in state
        if "start_time" not in self.state:
            self.state["start_time"] = self.start_time
            self._save_state()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        """Configure file + console logging."""
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(str(self.LOG_FILE), mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        ))
        root.addHandler(fh)

        # Console handler (if not already present)
        has_console = any(
            isinstance(h, logging.StreamHandler) and h.stream == sys.stderr
            for h in root.handlers
        )
        if not has_console:
            ch = logging.StreamHandler(sys.stderr)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s"
            ))
            root.addHandler(ch)

    def log(self, msg: str, *args) -> None:
        """Log a message to both file and console."""
        logger.info(msg, *args)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Persist state to JSON for resume capability."""
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _load_state(self) -> Dict[str, Any]:
        """Load state from JSON if exists, otherwise return empty dict."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
                logger.info("Resumed from state: %s", self.STATE_FILE)
                return state
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Failed to load state, starting fresh: %s", exc)
        return {
            "completed_tracks": [],
            "last_checkpoint": None,
            "track_history": {},
            "cross_eval": None,
            "regression": None,
            "live_test": None,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main entry point -- runs all tracks sequentially."""
        self.log("=== DocWain V2 Autonomous Training Pipeline Started ===")
        self.log("Tracks: %s", self.tracks)
        self.log("Base model: %s", self.base_model)

        checkpoint = self.state.get("last_checkpoint")

        for track in self.tracks:
            if track in self.state.get("completed_tracks", []):
                self.log("Track %s already completed, skipping", track)
                continue

            self.log("=== Starting Track: %s ===", track)
            try:
                checkpoint = self.run_track(track, base_checkpoint=checkpoint)
                self.state.setdefault("completed_tracks", []).append(track)
                self.state["last_checkpoint"] = checkpoint
                self._save_state()
                self.log("Track %s completed. Checkpoint: %s", track, checkpoint)
            except Exception as exc:
                self.log("Track %s FAILED: %s", track, exc)
                logger.error("Traceback:\n%s", traceback.format_exc())
                # Record failure but continue to next track
                self.state.setdefault("failed_tracks", {})[track] = {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._save_state()

        # Cross-track evaluation
        self.log("=== Cross-Track Evaluation ===")
        try:
            cross_results = self.run_cross_track_eval()
            self.state["cross_eval"] = cross_results
            self._save_state()
        except Exception as exc:
            self.log("Cross-track eval failed: %s", exc)

        # Regression vs V1
        self.log("=== Regression vs V1 ===")
        try:
            regression_passed = self.run_regression_vs_v1()
            if regression_passed:
                self.log("Regression PASSED -- promoting to latest")
                self.promote_to_latest()
            else:
                self.log("Regression FAILED -- V2 stays as v2-wip only")
        except Exception as exc:
            self.log("Regression test failed: %s", exc)

        # Live testing
        self.log("=== Live Testing & Issue Fix Loop ===")
        try:
            self.run_live_test_loop()
        except Exception as exc:
            self.log("Live test loop failed: %s", exc)

        # Generate final report
        self.generate_report()

        elapsed = time.time() - self.state.get("start_time", self.start_time)
        self.log(
            "=== Pipeline Complete (%.1f hours) ===",
            elapsed / 3600,
        )

    # ------------------------------------------------------------------
    # Per-track iterative training
    # ------------------------------------------------------------------

    def run_track(self, track: str, base_checkpoint: Optional[str] = None) -> str:
        """Iterative loop for a single track -- no iteration cap until
        MAX_ITERATIONS_PER_TRACK.

        Loop:
        1. Generate training data (seed + targeted for weak areas)
        2. Train LoRA (SFT + optional DPO)
        3. Update Ollama model
        4. Evaluate all test bank examples for this track
        5. If gate passed: return checkpoint
        6. If not: evolve strategy and continue

        Returns path to the merged checkpoint.
        """
        track_dir = self.ARTIFACT_DIR / track
        track_dir.mkdir(parents=True, exist_ok=True)

        history = self.state.get("track_history", {}).get(track, {
            "iterations": [],
            "best_score": 0.0,
            "best_iteration": 0,
            "weak_areas": [],
        })
        start_iteration = len(history.get("iterations", [])) + 1
        weak_areas = history.get("weak_areas", [])
        best_score = history.get("best_score", 0.0)
        # Recover best checkpoint from history when resuming
        best_checkpoint = base_checkpoint
        if history.get("best_iteration", 0) > 0:
            for it in history.get("iterations", []):
                if it.get("iteration") == history["best_iteration"] and it.get("merged_dir"):
                    best_checkpoint = it["merged_dir"]
                    break
        # Recompute no_improve_count from history so restarts don't lose
        # stagnation state (needed for timely strategy escalation to DPO)
        no_improve_count = 0
        if history.get("iterations"):
            running_best = 0.0
            for it in history["iterations"]:
                score = it.get("avg_score", 0.0)
                if score > running_best:
                    if score - running_best >= 0.1:
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    running_best = score
                else:
                    no_improve_count += 1
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 3

        for iteration in range(start_iteration, MAX_ITERATIONS_PER_TRACK + 1):
            iter_start = time.time()
            strategy = _get_strategy(iteration, no_improve_count)
            self.log(
                "Track %s, iteration %d, strategy: %s",
                track, iteration, strategy["name"],
            )

            # 0. Disk housekeeping — keep only the best and latest checkpoints
            _cleanup_old_iterations(track_dir, history, iteration)

            # 1. Generate data
            data_dir = track_dir / f"iter_{iteration}" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            sft_path = data_dir / "sft_train.jsonl"
            dpo_path = data_dir / "dpo_train.jsonl"

            base_count = 500
            sft_count = int(base_count * strategy["data_multiplier"])

            generator = _load_generator(track)
            if generator is not None:
                try:
                    gen_result = generator(
                        output_dir=str(data_dir),
                        seed=42 + iteration * 1000,
                    )
                    sft_path = Path(gen_result.get("sft_path", str(sft_path)))
                    if "dpo_path" in gen_result and gen_result["dpo_path"]:
                        dpo_path = Path(gen_result["dpo_path"])
                    sft_count = gen_result.get("sft_count", sft_count)
                    self.log("Generated %d examples via track generator", sft_count)
                except Exception as exc:
                    self.log(
                        "Track generator failed, using fallback: %s", exc,
                    )
                    sft_count = _generate_fallback_sft_data(
                        track, sft_path, count=sft_count,
                        seed=42 + iteration * 1000,
                        weak_areas=weak_areas,
                    )
            else:
                sft_count = _generate_fallback_sft_data(
                    track, sft_path, count=sft_count,
                    seed=42 + iteration * 1000,
                    weak_areas=weak_areas,
                )

            # Generate DPO data if strategy calls for it
            dpo_actual_path: Optional[str] = None
            if strategy["use_dpo"]:
                dpo_count = _generate_fallback_dpo_data(
                    track, dpo_path, count=int(sft_count * 0.4),
                    seed=42 + iteration * 1000,
                )
                if dpo_count > 0:
                    dpo_actual_path = str(dpo_path)

            # 2. Train
            iter_output = track_dir / f"iter_{iteration}" / "model"
            config = TrackTrainingConfig(
                track_name=track,
                base_model=self.base_model,
                base_checkpoint=best_checkpoint,
                data_path=str(sft_path),
                dpo_path=dpo_actual_path,
                output_dir=str(iter_output),
                lora_r=strategy["lora_r"],
                lora_alpha=strategy["lora_r"] * 2,
                learning_rate=strategy["lr"],
                epochs=strategy["epochs"],
                batch_size=4,
                gradient_accumulation_steps=8,
                max_seq_length=4096,
                warmup_ratio=0.10,
                dpo_epochs=2,
                dpo_lr=5e-6,
                dpo_beta=0.1,
                ollama_model_name="DHS/DocWain",
                ollama_tag="v2-wip",
            )

            try:
                merged_dir = train_track(config)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                self.log("Training failed at iteration %d: %s", iteration, exc)
                logger.error("Traceback:\n%s", traceback.format_exc())
                history.setdefault("iterations", []).append({
                    "iteration": iteration,
                    "strategy": strategy["name"],
                    "error": str(exc),
                    "score": 0.0,
                    "duration": time.time() - iter_start,
                })
                self.state.setdefault("track_history", {})[track] = history
                self._save_state()
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    self.log(
                        "Track %s: %d consecutive failures, halting track to avoid "
                        "wasting iterations. Last error: %s",
                        track, consecutive_errors, exc,
                    )
                    break
                continue

            # 3. Evaluate
            self.log("Evaluating track %s after iteration %d", track, iteration)
            try:
                evaluator = TrackEvaluator(model_name=OLLAMA_V2_WIP)
                eval_result = evaluator.evaluate_track(track)
                avg_score = eval_result.get("avg_score", eval_result.get("overall_avg", 0.0))
                pass_rate = eval_result.get("pass_rate", 1.0 if eval_result.get("passed") else 0.0)
            except Exception as exc:
                self.log("Evaluation failed: %s", exc)
                logger.error("Eval traceback:\n%s", traceback.format_exc())
                avg_score = 0.0
                pass_rate = 0.0
                eval_result = {"overall_avg": 0, "passed": False, "per_example": [], "weak_categories": []}

            duration = time.time() - iter_start

            # Record iteration
            iter_record = {
                "iteration": iteration,
                "strategy": strategy["name"],
                "sft_count": sft_count,
                "dpo_used": strategy["use_dpo"],
                "avg_score": avg_score,
                "pass_rate": pass_rate,
                "duration": round(duration, 1),
                "merged_dir": merged_dir,
            }
            history.setdefault("iterations", []).append(iter_record)

            self.log(
                "Track %s iter %d: score=%.1f, pass_rate=%.0f%%, strategy=%s, time=%.1fs",
                track, iteration, avg_score, pass_rate * 100, strategy["name"], duration,
            )

            # Update best — require meaningful improvement (>=0.1) to reset
            # stagnation counter, otherwise SFT-only loops never escalate to DPO
            MINIMUM_IMPROVEMENT = 0.1
            if avg_score > best_score:
                improvement = avg_score - best_score
                best_score = avg_score
                best_checkpoint = merged_dir
                history["best_score"] = best_score
                history["best_iteration"] = iteration
                if improvement >= MINIMUM_IMPROVEMENT:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            # Identify weak areas from eval results
            new_weak = []
            for ex in eval_result.get("per_example", []):
                ex_scores = ex.get("scores", {})
                if ex_scores:
                    ex_avg = sum(ex_scores.values()) / len(ex_scores)
                    if ex_avg < 3.0:
                        cat = ex.get("category", "")
                        if cat and cat not in new_weak:
                            new_weak.append(cat)
            # Also include weak categories identified by the evaluator
            for wc in eval_result.get("weak_categories", []):
                if wc not in new_weak:
                    new_weak.append(wc)
            if new_weak:
                weak_areas = new_weak
                history["weak_areas"] = weak_areas

            self.state.setdefault("track_history", {})[track] = history
            self._save_state()

            # 5. Check gate
            if avg_score >= TRACK_PASS_THRESHOLD:
                self.log(
                    "Track %s PASSED (%.1f >= %.1f) at iteration %d",
                    track, avg_score, TRACK_PASS_THRESHOLD, iteration,
                )
                break

            # 6. Log strategy evolution if applicable
            next_strategy = _get_strategy(iteration + 1, no_improve_count)
            if next_strategy["name"] != strategy["name"]:
                self.log(
                    "Strategy evolution: %s -> %s (no improvement for %d iters)",
                    strategy["name"], next_strategy["name"], no_improve_count,
                )

        return best_checkpoint or base_checkpoint or ""

    # ------------------------------------------------------------------
    # Cross-track evaluation
    # ------------------------------------------------------------------

    def run_cross_track_eval(self) -> Dict[str, Any]:
        """Evaluate the model across all tracks after all individual tracks pass."""
        evaluator = TrackEvaluator(model_name=OLLAMA_V2_WIP)
        results = evaluator.evaluate_all_tracks()

        self.log("Cross-track results:")
        for track_name, track_result in results.get("per_track", {}).items():
            self.log(
                "  %s: avg=%.1f, passed=%s",
                track_name,
                track_result.get("overall_avg", 0.0),
                track_result.get("passed", False),
            )
        self.log(
            "  Overall: avg=%.1f, all_passed=%s",
            results.get("overall_avg", 0.0),
            results.get("all_passed", False),
        )

        return results

    # ------------------------------------------------------------------
    # Regression vs V1
    # ------------------------------------------------------------------

    def run_regression_vs_v1(self) -> bool:
        """Query both V1 and V2 with same prompts, compare scores.

        Must pass >= 90% of V1 capabilities with no category below 85%.
        """
        regression_prompts = [
            {
                "id": "reg_01",
                "category": "persona",
                "prompt": "Who are you? Tell me about yourself.",
                "keywords": ["DocWain", "DHS", "document", "intelligence"],
            },
            {
                "id": "reg_02",
                "category": "persona",
                "prompt": "What can you do?",
                "keywords": ["document", "extract", "analyze", "compare", "summarize"],
            },
            {
                "id": "reg_03",
                "category": "formatting",
                "prompt": "Extract the key details from this contract summary: Parties are Acme Corp and Beta LLC, value is $500K, term is 24 months starting Jan 2024.",
                "keywords": ["Acme", "Beta", "$500", "24 months", "2024"],
            },
            {
                "id": "reg_04",
                "category": "formatting",
                "prompt": "Compare two proposals: Proposal A costs $100K with 6-month delivery; Proposal B costs $80K with 9-month delivery.",
                "keywords": ["$100", "$80", "6-month", "9-month"],
            },
            {
                "id": "reg_05",
                "category": "rag_accuracy",
                "prompt": "What is DHS IT Solutions and where is it headquartered?",
                "keywords": ["DHS", "Newcastle", "UK", "2016"],
            },
            {
                "id": "reg_06",
                "category": "rag_accuracy",
                "prompt": "Explain how DocWain processes a user query, step by step.",
                "keywords": ["intent", "retrieval", "rerank", "grounding", "generation"],
            },
            {
                "id": "reg_07",
                "category": "coherence",
                "prompt": "I uploaded 3 financial documents. Summarize the overall financial health.",
                "keywords": ["revenue", "financial", "summary", "document"],
            },
            {
                "id": "reg_08",
                "category": "coherence",
                "prompt": "Explain the difference between PII screening and compliance screening.",
                "keywords": ["PII", "compliance", "screening", "personal"],
            },
            {
                "id": "reg_09",
                "category": "citation",
                "prompt": "Based on the uploaded documents, what are the payment terms?",
                "keywords": ["payment", "terms", "document"],
            },
            {
                "id": "reg_10",
                "category": "citation",
                "prompt": "What does Section 5 of the agreement say about termination?",
                "keywords": ["Section", "termination", "agreement"],
            },
        ]

        v1_scores: Dict[str, List[float]] = {}
        v2_scores: Dict[str, List[float]] = {}

        for item in regression_prompts:
            cat = item["category"]
            prompt = item["prompt"]
            keywords = item["keywords"]

            # Query V1
            v1_response = query_ollama(prompt, model=OLLAMA_V1, timeout=120)
            v1_kw_score = 0.0
            if v1_response:
                lower = v1_response.lower()
                hits = sum(1 for kw in keywords if kw.lower() in lower)
                v1_kw_score = (hits / len(keywords)) * 100 if keywords else 100
            v1_scores.setdefault(cat, []).append(v1_kw_score)

            # Query V2
            v2_response = query_ollama(prompt, model=OLLAMA_V2_WIP, timeout=120)
            v2_kw_score = 0.0
            if v2_response:
                lower = v2_response.lower()
                hits = sum(1 for kw in keywords if kw.lower() in lower)
                v2_kw_score = (hits / len(keywords)) * 100 if keywords else 100
            v2_scores.setdefault(cat, []).append(v2_kw_score)

            self.log(
                "Regression %s [%s]: V1=%.0f V2=%.0f",
                item["id"], cat, v1_kw_score, v2_kw_score,
            )

        # Compute per-category and overall
        category_results: Dict[str, Dict[str, float]] = {}
        all_v1: List[float] = []
        all_v2: List[float] = []

        for cat in set(list(v1_scores.keys()) + list(v2_scores.keys())):
            v1_cat = v1_scores.get(cat, [0])
            v2_cat = v2_scores.get(cat, [0])
            v1_avg = sum(v1_cat) / len(v1_cat) if v1_cat else 0
            v2_avg = sum(v2_cat) / len(v2_cat) if v2_cat else 0
            pct_of_v1 = (v2_avg / v1_avg * 100) if v1_avg > 0 else 100
            category_results[cat] = {
                "v1_avg": v1_avg,
                "v2_avg": v2_avg,
                "pct_of_v1": pct_of_v1,
            }
            all_v1.extend(v1_cat)
            all_v2.extend(v2_cat)

        overall_v1 = sum(all_v1) / len(all_v1) if all_v1 else 0
        overall_v2 = sum(all_v2) / len(all_v2) if all_v2 else 0
        overall_pct = (overall_v2 / overall_v1 * 100) if overall_v1 > 0 else 100

        self.log(
            "Regression overall: V1=%.1f V2=%.1f (%.1f%% of V1)",
            overall_v1, overall_v2, overall_pct,
        )

        # Check thresholds
        passed = overall_pct >= REGRESSION_OVERALL_MIN
        for cat, result in category_results.items():
            if result["pct_of_v1"] < REGRESSION_CATEGORY_MIN:
                self.log(
                    "Regression FAIL: category %s at %.1f%% (min %.1f%%)",
                    cat, result["pct_of_v1"], REGRESSION_CATEGORY_MIN,
                )
                passed = False

        regression_data = {
            "overall_v1": overall_v1,
            "overall_v2": overall_v2,
            "overall_pct": overall_pct,
            "passed": passed,
            "categories": category_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.state["regression"] = regression_data
        self._save_state()

        return passed

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_to_latest(self) -> None:
        """Copy DHS/DocWain:v2-wip to DHS/DocWain:latest."""
        self.log("Promoting %s to %s", OLLAMA_V2_WIP, OLLAMA_LATEST)
        try:
            # Backup current latest as v1-backup
            subprocess.run(
                ["ollama", "cp", OLLAMA_LATEST, OLLAMA_V1],
                capture_output=True, text=True, timeout=300,
            )
            self.log("Backed up current latest as %s", OLLAMA_V1)
        except Exception as exc:
            self.log("V1 backup failed (may not exist): %s", exc)

        try:
            result = subprocess.run(
                ["ollama", "cp", OLLAMA_V2_WIP, OLLAMA_LATEST],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                self.log("Promotion successful: %s is now latest", OLLAMA_V2_WIP)
                self.state["promoted"] = True
            else:
                self.log("Promotion failed: %s", result.stderr)
                self.state["promoted"] = False
        except Exception as exc:
            self.log("Promotion failed: %s", exc)
            self.state["promoted"] = False

        self._save_state()

    # ------------------------------------------------------------------
    # Live test loop
    # ------------------------------------------------------------------

    def run_live_test_loop(self) -> None:
        """After all tracks pass, test the model with realistic user scenarios.

        Generates 20 realistic user prompts spanning all capabilities.
        Queries the model, programmatically evaluates responses.
        Identifies issues, generates targeted fix data, retrains.
        Repeats until all test scenarios pass or max retries reached.
        """
        max_live_iterations = 5
        scenarios = _build_live_test_scenarios()

        for live_iter in range(1, max_live_iterations + 1):
            self.log("Live test iteration %d/%d", live_iter, max_live_iterations)

            results: List[Dict[str, Any]] = []
            failing: List[Dict[str, Any]] = []

            for scenario in scenarios:
                response = query_ollama(
                    scenario["prompt"],
                    model=OLLAMA_V2_WIP,
                    timeout=120,
                )
                score = _score_live_scenario(response, scenario)
                results.append({
                    "id": scenario["id"],
                    "track": scenario["track"],
                    "score": score,
                    "response_length": len(response),
                })
                if score < 70.0:
                    failing.append(scenario)
                self.log(
                    "  %s [%s]: score=%.1f",
                    scenario["id"], scenario["track"], score,
                )

            avg_live = (
                sum(r["score"] for r in results) / len(results)
                if results else 0
            )
            pass_count = sum(1 for r in results if r["score"] >= 70.0)
            self.log(
                "Live test avg=%.1f, passed=%d/%d",
                avg_live, pass_count, len(results),
            )

            # Store results
            live_data = {
                "iteration": live_iter,
                "avg_score": avg_live,
                "pass_count": pass_count,
                "total": len(results),
                "per_scenario": results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.state.setdefault("live_test_history", []).append(live_data)
            self._save_state()

            if not failing:
                self.log("All live test scenarios passed!")
                break

            # Generate targeted fix data for failing scenarios
            self.log(
                "%d scenarios failing, generating targeted fix data",
                len(failing),
            )

            fix_data_dir = self.ARTIFACT_DIR / "live_fix" / f"iter_{live_iter}"
            fix_data_dir.mkdir(parents=True, exist_ok=True)

            # Group failures by track
            track_failures: Dict[str, List[Dict]] = {}
            for f in failing:
                track_failures.setdefault(f["track"], []).append(f)

            for fail_track, fail_scenarios in track_failures.items():
                fix_sft_path = fix_data_dir / f"{fail_track}_fix.jsonl"
                count = _generate_fallback_sft_data(
                    fail_track, fix_sft_path,
                    count=len(fail_scenarios) * 50,  # 50 examples per failing scenario
                    seed=42 + live_iter * 10000,
                )

                if count == 0:
                    continue

                # Quick retrain with targeted data
                fix_output = fix_data_dir / f"{fail_track}_model"
                last_checkpoint = self.state.get("last_checkpoint")
                config = TrackTrainingConfig(
                    track_name=fail_track,
                    base_model=self.base_model,
                    base_checkpoint=last_checkpoint,
                    data_path=str(fix_sft_path),
                    output_dir=str(fix_output),
                    lora_r=64,
                    lora_alpha=128,
                    learning_rate=1e-5,  # Lower LR for fix pass
                    epochs=2,
                    batch_size=4,
                    gradient_accumulation_steps=8,
                    ollama_model_name="DHS/DocWain",
                    ollama_tag="v2-wip",
                )

                try:
                    new_checkpoint = train_track(config)
                    self.state["last_checkpoint"] = new_checkpoint
                    self._save_state()
                    self.log(
                        "Fix training for %s complete: %s",
                        fail_track, new_checkpoint,
                    )
                except Exception as exc:
                    self.log("Fix training for %s failed: %s", fail_track, exc)

        # Record final live test state
        self.state["live_test"] = {
            "completed": True,
            "iterations": live_iter,
            "final_avg": avg_live,
            "final_pass_count": pass_count,
            "final_total": len(results),
        }
        self._save_state()

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> None:
        """Generate comprehensive markdown report of everything done."""
        elapsed = time.time() - self.state.get("start_time", self.start_time)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines: List[str] = []
        lines.append("# DocWain V2 Autonomous Training Report")
        lines.append("")
        lines.append(f"**Generated:** {now}")
        lines.append(f"**Total training time:** {elapsed / 3600:.1f} hours")
        lines.append(f"**Base model:** {self.base_model}")
        lines.append(f"**Tracks:** {', '.join(self.tracks)}")
        lines.append("")

        # Summary
        completed = self.state.get("completed_tracks", [])
        failed = self.state.get("failed_tracks", {})
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Completed tracks:** {len(completed)}/{len(self.tracks)}")
        lines.append(f"- **Failed tracks:** {len(failed)}")
        if completed:
            lines.append(f"- **Completed:** {', '.join(completed)}")
        if failed:
            lines.append(f"- **Failed:** {', '.join(failed.keys())}")
        lines.append("")

        # Per-track details
        lines.append("## Per-Track Results")
        lines.append("")

        track_history = self.state.get("track_history", {})
        for track_name in self.tracks:
            th = track_history.get(track_name, {})
            iterations = th.get("iterations", [])
            best_score = th.get("best_score", 0.0)
            best_iter = th.get("best_iteration", 0)

            lines.append(f"### {track_name}")
            lines.append("")
            lines.append(f"- **Iterations:** {len(iterations)}")
            lines.append(f"- **Best score:** {best_score:.1f}")
            lines.append(f"- **Best iteration:** {best_iter}")
            lines.append(f"- **Status:** {'PASSED' if track_name in completed else 'INCOMPLETE'}")
            lines.append("")

            if iterations:
                lines.append("| Iter | Strategy | Score | Pass Rate | SFT Count | DPO | Duration |")
                lines.append("|------|----------|-------|-----------|-----------|-----|----------|")
                for it in iterations:
                    if "error" in it:
                        lines.append(
                            f"| {it.get('iteration', '?')} | {it.get('strategy', '?')} "
                            f"| ERROR | - | - | - | {it.get('duration', 0):.0f}s |"
                        )
                    else:
                        lines.append(
                            f"| {it.get('iteration', '?')} "
                            f"| {it.get('strategy', '?')} "
                            f"| {it.get('avg_score', 0):.1f} "
                            f"| {it.get('pass_rate', 0) * 100:.0f}% "
                            f"| {it.get('sft_count', 0)} "
                            f"| {'Yes' if it.get('dpo_used') else 'No'} "
                            f"| {it.get('duration', 0):.0f}s |"
                        )
                lines.append("")

            # Strategy changes
            strategies_used = list(dict.fromkeys(
                it.get("strategy", "") for it in iterations if "strategy" in it
            ))
            if len(strategies_used) > 1:
                lines.append(f"**Strategy evolution:** {' -> '.join(strategies_used)}")
                lines.append("")

        # Cross-track evaluation
        cross_eval = self.state.get("cross_eval")
        if cross_eval:
            lines.append("## Cross-Track Evaluation")
            lines.append("")
            lines.append(f"- **Overall average:** {cross_eval.get('overall_avg', 0):.1f}")
            lines.append(
                f"- **Overall pass rate:** "
                f"{cross_eval.get('overall_pass_rate', 0) * 100:.0f}%"
            )
            lines.append("")

            per_track = cross_eval.get("per_track", {})
            if per_track:
                lines.append("| Track | Avg Score | Pass Rate |")
                lines.append("|-------|-----------|-----------|")
                for tn, tr in per_track.items():
                    lines.append(
                        f"| {tn} | {tr.get('avg_score', 0):.1f} "
                        f"| {tr.get('pass_rate', 0) * 100:.0f}% |"
                    )
                lines.append("")

        # Regression results
        regression = self.state.get("regression")
        if regression:
            lines.append("## Regression vs V1")
            lines.append("")
            lines.append(f"- **V1 average:** {regression.get('overall_v1', 0):.1f}")
            lines.append(f"- **V2 average:** {regression.get('overall_v2', 0):.1f}")
            lines.append(
                f"- **V2 as % of V1:** {regression.get('overall_pct', 0):.1f}%"
            )
            lines.append(
                f"- **Passed:** {'Yes' if regression.get('passed') else 'No'}"
            )
            lines.append("")

            categories = regression.get("categories", {})
            if categories:
                lines.append("| Category | V1 Avg | V2 Avg | % of V1 |")
                lines.append("|----------|--------|--------|---------|")
                for cat, cdata in categories.items():
                    lines.append(
                        f"| {cat} | {cdata.get('v1_avg', 0):.1f} "
                        f"| {cdata.get('v2_avg', 0):.1f} "
                        f"| {cdata.get('pct_of_v1', 0):.1f}% |"
                    )
                lines.append("")

        # Live test results
        live_test = self.state.get("live_test")
        if live_test:
            lines.append("## Live Testing")
            lines.append("")
            lines.append(f"- **Iterations:** {live_test.get('iterations', 0)}")
            lines.append(f"- **Final average:** {live_test.get('final_avg', 0):.1f}")
            lines.append(
                f"- **Final pass rate:** "
                f"{live_test.get('final_pass_count', 0)}/"
                f"{live_test.get('final_total', 0)}"
            )
            lines.append("")

        live_history = self.state.get("live_test_history", [])
        if live_history:
            lines.append("### Live Test Iteration History")
            lines.append("")
            lines.append("| Iteration | Avg Score | Passed | Total |")
            lines.append("|-----------|-----------|--------|-------|")
            for lh in live_history:
                lines.append(
                    f"| {lh.get('iteration', '?')} "
                    f"| {lh.get('avg_score', 0):.1f} "
                    f"| {lh.get('pass_count', 0)} "
                    f"| {lh.get('total', 0)} |"
                )
            lines.append("")

        # Promotion status
        lines.append("## Promotion Status")
        lines.append("")
        promoted = self.state.get("promoted", False)
        lines.append(
            f"- **V2 promoted to latest:** {'Yes' if promoted else 'No'}"
        )
        lines.append(f"- **Last checkpoint:** {self.state.get('last_checkpoint', 'N/A')}")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append(
            "*This report was generated automatically by the DocWain V2 "
            "Autonomous Training Pipeline.*"
        )

        report_content = "\n".join(lines) + "\n"
        self.REPORT_FILE.write_text(report_content, encoding="utf-8")
        self.log("Report written to %s", self.REPORT_FILE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the autonomous trainer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DocWain V2 Autonomous Training Pipeline",
    )
    parser.add_argument(
        "--tracks", nargs="*", default=None,
        help="Specific tracks to train (default: all 6)",
    )
    parser.add_argument(
        "--base-model", default="unsloth/Qwen3-14B-bnb-4bit",
        help="Base model for fine-tuning",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved state (default behavior if state exists)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    trainer = AutonomousTrainer(
        tracks=args.tracks,
        base_model=args.base_model,
    )
    trainer.run()


if __name__ == "__main__":
    main()
