"""Tests for the knowledge distillation data generators."""

import pytest

from src.finetune.distillation.generators import (
    generate_all_categories,
    generate_analytical_examples,
    generate_boundary_examples,
    generate_content_examples,
    generate_crossdoc_examples,
    generate_dpo_pairs,
    generate_extraction_examples,
    generate_reasoning_examples,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESUME = (
    "John Smith is a Senior Software Engineer with 10 years of experience. "
    "He worked at Acme Corporation from 01/15/2016 to 03/20/2022. "
    "His email is john.smith@acme.com. He earned $150,000 annually. "
    "Skills include Python, Machine Learning, and Cloud Architecture. "
    "Education: Bachelor of Science from Stanford University, 2012."
)

SAMPLE_INVOICE = (
    "Invoice #INV-2024-0042 from Global Supplies Ltd to Bright Future Inc. "
    "Date: March 15, 2024. Line items: Office Chairs $2,400.00, "
    "Standing Desks $3,600.00, Monitor Arms $800.00. "
    "Subtotal: $6,800.00. Tax: $612.00. Total: $7,412.00. "
    "Payment terms: Net 30. Due date: 04/14/2024. "
    "Contact: billing@globalsupplies.com."
)

SHORT_TEXT = "Too short."

REQUIRED_SFT_META = {"area", "difficulty", "category", "source"}
QWEN_MARKERS = ("<|im_start|>", "<|im_end|>", "<think>")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_sft_example(example: dict) -> None:
    """Assert an SFT example has correct format and metadata."""
    assert "text" in example, "SFT example missing 'text' key"
    for marker in QWEN_MARKERS:
        assert marker in example["text"], f"Missing Qwen3 marker: {marker}"
    for key in REQUIRED_SFT_META:
        assert key in example, f"Missing metadata key: {key}"
    assert example["source"] == "claude_distillation"
    assert example["difficulty"] in ("easy", "medium", "hard")


def _check_dpo_pair(pair: dict) -> None:
    """Assert a DPO pair has correct keys."""
    for key in ("prompt", "chosen", "rejected"):
        assert key in pair, f"DPO pair missing '{key}' key"
    assert "source" in pair
    assert "category" in pair


# ---------------------------------------------------------------------------
# Tests: Extraction
# ---------------------------------------------------------------------------


class TestExtractionExamples:
    def test_produces_examples(self):
        examples = generate_extraction_examples(SAMPLE_RESUME, "resume", "test_resume.pdf")
        assert len(examples) >= 1
        for ex in examples:
            _check_sft_example(ex)

    def test_empty_input(self):
        assert generate_extraction_examples("", "resume", "x") == []

    def test_short_input(self):
        assert generate_extraction_examples(SHORT_TEXT, "generic", "x") == []

    def test_invoice_type(self):
        examples = generate_extraction_examples(SAMPLE_INVOICE, "invoice", "inv.pdf")
        assert len(examples) >= 1
        for ex in examples:
            assert ex["category"] == "extraction"


# ---------------------------------------------------------------------------
# Tests: Analytical
# ---------------------------------------------------------------------------


class TestAnalyticalExamples:
    def test_produces_examples(self):
        examples = generate_analytical_examples(SAMPLE_INVOICE, "invoice", "inv.pdf")
        assert len(examples) >= 1
        for ex in examples:
            _check_sft_example(ex)
            assert ex["category"] == "analytical"

    def test_empty_input(self):
        assert generate_analytical_examples("", "contract", "x") == []


# ---------------------------------------------------------------------------
# Tests: Cross-document
# ---------------------------------------------------------------------------


class TestCrossdocExamples:
    def test_produces_examples(self):
        examples = generate_crossdoc_examples(
            [SAMPLE_RESUME, SAMPLE_INVOICE],
            ["resume", "invoice"],
            ["resume.pdf", "invoice.pdf"],
        )
        assert len(examples) >= 1
        for ex in examples:
            _check_sft_example(ex)
            assert ex["category"] == "crossdoc"

    def test_single_doc_returns_empty(self):
        assert generate_crossdoc_examples(
            [SAMPLE_RESUME], ["resume"], ["r.pdf"]
        ) == []

    def test_short_doc_returns_empty(self):
        assert generate_crossdoc_examples(
            [SAMPLE_RESUME, SHORT_TEXT],
            ["resume", "generic"],
            ["r.pdf", "s.pdf"],
        ) == []


# ---------------------------------------------------------------------------
# Tests: Content generation
# ---------------------------------------------------------------------------


class TestContentExamples:
    def test_produces_examples(self):
        examples = generate_content_examples(SAMPLE_INVOICE, "invoice", "inv.pdf")
        assert len(examples) >= 1
        for ex in examples:
            _check_sft_example(ex)
            assert ex["category"] == "content"

    def test_empty_input(self):
        assert generate_content_examples("", "generic", "x") == []


# ---------------------------------------------------------------------------
# Tests: Boundary
# ---------------------------------------------------------------------------


class TestBoundaryExamples:
    def test_produces_examples(self):
        examples = generate_boundary_examples(SAMPLE_RESUME, "resume", "r.pdf")
        assert len(examples) >= 1
        for ex in examples:
            _check_sft_example(ex)
            assert ex["category"] == "boundary"

    def test_empty_input(self):
        assert generate_boundary_examples("", "resume", "x") == []


# ---------------------------------------------------------------------------
# Tests: Reasoning
# ---------------------------------------------------------------------------


class TestReasoningExamples:
    def test_produces_examples(self):
        examples = generate_reasoning_examples(SAMPLE_RESUME, "resume", "r.pdf")
        assert len(examples) >= 1
        for ex in examples:
            _check_sft_example(ex)
            assert ex["category"] == "reasoning"

    def test_empty_input(self):
        assert generate_reasoning_examples("", "generic", "x") == []


# ---------------------------------------------------------------------------
# Tests: DPO pairs
# ---------------------------------------------------------------------------


class TestDpoPairs:
    def test_produces_pairs(self):
        pairs = generate_dpo_pairs(SAMPLE_INVOICE, "invoice")
        assert len(pairs) >= 1
        for p in pairs:
            _check_dpo_pair(p)

    def test_dpo_qwen_markers(self):
        pairs = generate_dpo_pairs(SAMPLE_RESUME, "resume")
        for p in pairs:
            assert "<|im_start|>" in p["prompt"]
            assert "<think>" in p["chosen"]
            assert "<think>" in p["rejected"]

    def test_empty_input(self):
        assert generate_dpo_pairs("", "generic") == []


# ---------------------------------------------------------------------------
# Tests: generate_all_categories
# ---------------------------------------------------------------------------


class TestAllCategories:
    def test_produces_multiple_categories(self):
        examples = generate_all_categories(SAMPLE_INVOICE, "invoice", "inv.pdf")
        assert len(examples) >= 3
        categories = {ex["category"] for ex in examples}
        assert len(categories) >= 3, f"Expected 3+ categories, got {categories}"

    def test_all_have_correct_format(self):
        examples = generate_all_categories(SAMPLE_RESUME, "resume", "r.pdf")
        for ex in examples:
            _check_sft_example(ex)

    def test_empty_input(self):
        assert generate_all_categories("", "generic", "x") == []
