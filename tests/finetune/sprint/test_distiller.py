import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_distiller_categories():
    from src.finetune.sprint.distiller import DISTILL_CATEGORIES

    expected = [
        "completeness_extraction", "intent_context", "anti_hallucination",
        "ocr_vision", "excel_csv", "deep_reasoning", "cross_document",
    ]
    for cat in expected:
        assert cat in DISTILL_CATEGORIES


def test_format_sft_uses_existing_format():
    from src.finetune.sprint.distiller import format_sft

    example = format_sft(
        query="Extract all info from this invoice",
        reasoning="I will scan the invoice for key fields...",
        answer="Invoice #123, Total: $500",
        category="completeness_extraction",
        difficulty="medium",
    )
    assert "text" in example
    assert "<|im_start|>" in example["text"]
    assert "<think>" in example["text"]
    assert "category" in example
    assert example["category"] == "completeness_extraction"


def test_format_dpo_structure():
    from src.finetune.sprint.distiller import format_dpo

    example = format_dpo(
        query="What is the total?",
        chosen_reasoning="Looking at the invoice...",
        chosen_answer="The total is $500.",
        rejected_reasoning="I don't see clear data...",
        rejected_answer="0 items found.",
        category="anti_hallucination",
    )
    assert "prompt" in example
    assert "chosen" in example
    assert "rejected" in example
    assert "0 items found" in example["rejected"]


def test_generate_sft_batch_returns_correct_count():
    from src.finetune.sprint.distiller import generate_sft_batch

    with patch("src.finetune.sprint.distiller._call_claude", return_value={
        "question": "What are the key details?",
        "reasoning": "Step 1: Read the document...",
        "answer": "The invoice total is $500.",
    }):
        examples = generate_sft_batch(
            category="completeness_extraction",
            count=5,
            seed=42,
        )

    assert len(examples) == 5
    for ex in examples:
        assert "text" in ex
        assert "category" in ex


def test_save_examples_jsonl():
    from src.finetune.sprint.distiller import save_examples

    examples = [
        {"text": "example 1", "category": "test"},
        {"text": "example 2", "category": "test"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        save_examples(examples, path)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["text"] == "example 1"
