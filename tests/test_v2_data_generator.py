"""Tests for V2+ data generator base infrastructure."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.finetune.v2.data_generator import (
    JSONLWriter,
    format_dpo_example,
    format_eval_example,
    format_sft_example,
)
from src.finetune.v2.data_generator.base import (
    DOCWAIN_SYSTEM_PROMPT,
    DOMAINS,
    DOC_TYPES,
)


class TestBaseGenerator(unittest.TestCase):
    """Tests for the data generator base utilities."""

    # ---- JSONLWriter tests ----

    def test_jsonl_writer_creates_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sub", "out.jsonl")
            writer = JSONLWriter(path)
            writer.write({"key": "value"})
            writer.close()

            self.assertTrue(os.path.isfile(path))
            with open(path) as f:
                data = json.loads(f.readline())
            self.assertEqual(data, {"key": "value"})

    def test_jsonl_writer_multiple_records(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bulk.jsonl")
            writer = JSONLWriter(path)
            for i in range(100):
                writer.write({"index": i})
            writer.close()

            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 100)
            self.assertEqual(json.loads(lines[0])["index"], 0)
            self.assertEqual(json.loads(lines[99])["index"], 99)
            self.assertEqual(writer.count, 100)

    def test_jsonl_writer_context_manager(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ctx.jsonl")
            with JSONLWriter(path) as w:
                w.write({"a": 1})
                w.write({"b": 2})

            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)

    # ---- format_sft_example tests ----

    def test_format_sft_example(self):
        result = format_sft_example(
            query="What is the contract value?",
            reasoning="I need to look at clause 3.",
            answer="The contract value is $1M.",
        )

        self.assertIn("text", result)
        text = result["text"]
        self.assertIn("<|im_start|>system", text)
        self.assertIn(DOCWAIN_SYSTEM_PROMPT, text)
        self.assertIn("<|im_start|>user", text)
        self.assertIn("What is the contract value?", text)
        self.assertIn("<|im_start|>assistant", text)
        self.assertIn("<think>", text)
        self.assertIn("I need to look at clause 3.", text)
        self.assertIn("</think>", text)
        self.assertIn("The contract value is $1M.", text)
        self.assertIn("<|im_end|>", text)

    def test_format_sft_example_preserves_think_block(self):
        result = format_sft_example(
            query="Q",
            reasoning="Step 1\nStep 2",
            answer="A",
        )
        text = result["text"]
        # The think block must contain the multi-line reasoning intact
        self.assertIn("<think>\nStep 1\nStep 2\n</think>", text)

    def test_format_sft_example_custom_system_prompt(self):
        custom = "You are a test assistant."
        result = format_sft_example(
            query="Q", reasoning="R", answer="A", system_prompt=custom
        )
        text = result["text"]
        self.assertIn(custom, text)
        self.assertNotIn(DOCWAIN_SYSTEM_PROMPT, text)

    # ---- format_dpo_example tests ----

    def test_format_dpo_example(self):
        result = format_dpo_example(
            query="Summarise the invoice.",
            chosen_reasoning="Good reasoning",
            chosen_answer="Good answer",
            rejected_reasoning="Bad reasoning",
            rejected_answer="Bad answer",
        )

        self.assertIn("prompt", result)
        self.assertIn("chosen", result)
        self.assertIn("rejected", result)
        self.assertIn("<think>", result["chosen"])
        self.assertIn("Good reasoning", result["chosen"])
        self.assertIn("Good answer", result["chosen"])
        self.assertIn("<think>", result["rejected"])
        self.assertIn("Bad reasoning", result["rejected"])
        self.assertIn("Bad answer", result["rejected"])

    # ---- format_eval_example tests ----

    def test_format_eval_example(self):
        result = format_eval_example(
            benchmark="doc_qa",
            query="What is clause 5?",
            context="Clause 5 states ...",
            reference_answer="It states X.",
            rubric={"accuracy": 1.0},
        )

        for key in ("benchmark", "query", "context", "reference_answer", "rubric"):
            self.assertIn(key, result)
        self.assertEqual(result["benchmark"], "doc_qa")
        self.assertIsNone(result.get("expected_tools"))

    def test_format_eval_example_with_tools(self):
        result = format_eval_example(
            benchmark="tool_use",
            query="Extract table",
            context="...",
            reference_answer="...",
            rubric={},
            expected_tools=["table_extract"],
        )
        self.assertEqual(result["expected_tools"], ["table_extract"])

    # ---- Constants tests ----

    def test_domains_list(self):
        self.assertIsInstance(DOMAINS, list)
        self.assertIn("legal", DOMAINS)
        self.assertIn("financial", DOMAINS)
        self.assertEqual(len(DOMAINS), 7)

    def test_doc_types_list(self):
        self.assertIsInstance(DOC_TYPES, list)
        self.assertIn("contract", DOC_TYPES)
        self.assertIn("invoice", DOC_TYPES)
        self.assertEqual(len(DOC_TYPES), 18)


class TestPhase2Generator(unittest.TestCase):
    """Tests for Phase 2 document intelligence data generator."""

    def test_generate_table_examples(self):
        from src.finetune.v2.data_generator.phase2_doc_intelligence import (
            generate_table_examples,
        )

        examples = generate_table_examples(count=10)
        self.assertEqual(len(examples), 10)
        for ex in examples:
            self.assertIn("text", ex)
            self.assertIn("<think>", ex["text"])
            self.assertIn("</think>", ex["text"])
            self.assertIn("<|im_start|>", ex["text"])

    def test_generate_layout_examples(self):
        from src.finetune.v2.data_generator.phase2_doc_intelligence import (
            generate_layout_examples,
        )

        examples = generate_layout_examples(count=10)
        self.assertEqual(len(examples), 10)
        for ex in examples:
            self.assertIn("text", ex)
            self.assertIn("<think>", ex["text"])
            self.assertIn("</think>", ex["text"])

    def test_generate_ocr_examples(self):
        from src.finetune.v2.data_generator.phase2_doc_intelligence import (
            generate_ocr_examples,
        )

        examples = generate_ocr_examples(count=10)
        self.assertEqual(len(examples), 10)
        for ex in examples:
            self.assertIn("text", ex)
            self.assertIn("<think>", ex["text"])
            self.assertIn("</think>", ex["text"])

    def test_generate_cross_ref_examples(self):
        from src.finetune.v2.data_generator.phase2_doc_intelligence import (
            generate_cross_ref_examples,
        )

        examples = generate_cross_ref_examples(count=10)
        self.assertEqual(len(examples), 10)
        for ex in examples:
            self.assertIn("text", ex)
            self.assertIn("<think>", ex["text"])
            self.assertIn("</think>", ex["text"])

    def test_generate_all_phase2_data(self):
        from src.finetune.v2.data_generator.phase2_doc_intelligence import (
            generate_phase2_data,
        )

        with tempfile.TemporaryDirectory() as td:
            stats = generate_phase2_data(td, scale=0.01)

            self.assertIn("table_understanding", stats)
            self.assertIn("layout_comprehension", stats)
            self.assertIn("ocr_correction", stats)
            self.assertIn("cross_document_reasoning", stats)

            for name, count in stats.items():
                self.assertGreater(count, 0)
                fpath = os.path.join(td, f"phase2_{name}.jsonl")
                self.assertTrue(os.path.isfile(fpath), f"Missing file: {fpath}")
                with open(fpath) as f:
                    lines = f.readlines()
                self.assertGreater(len(lines), 0)

    def test_table_tiers_distribution(self):
        from src.finetune.v2.data_generator.phase2_doc_intelligence import (
            generate_table_examples,
        )

        examples = generate_table_examples(count=100)
        self.assertEqual(len(examples), 100)

        # Verify there is variation in the generated examples
        texts = [ex["text"] for ex in examples]
        unique_texts = set(texts)
        # With 100 examples and random variation, we should get significant uniqueness
        self.assertGreater(len(unique_texts), 50, "Expected diverse examples")


class TestPhase25DPOGenerator(unittest.TestCase):
    """Tests for Phase 2.5 DPO preference pair generator."""

    def test_generate_dpo_pairs(self):
        from src.finetune.v2.data_generator.phase2_5_dpo_pairs import (
            generate_dpo_pairs,
        )

        pairs = generate_dpo_pairs(count=10)
        self.assertEqual(len(pairs), 10)
        for pair in pairs:
            self.assertIn("prompt", pair)
            self.assertIn("chosen", pair)
            self.assertIn("rejected", pair)
            self.assertIn("<think>", pair["chosen"])
            self.assertIn("<think>", pair["rejected"])

    def test_corruption_types_present(self):
        from src.finetune.v2.data_generator.phase2_5_dpo_pairs import (
            generate_dpo_pairs,
            CORRUPTION_TYPES,
        )

        pairs = generate_dpo_pairs(count=50)
        self.assertEqual(len(pairs), 50)
        # With 10 templates and 50 examples, we get good variety
        # Verify all 5 corruption types are defined
        self.assertEqual(len(CORRUPTION_TYPES), 5)
        self.assertIn("reasoning_corruption", CORRUPTION_TYPES)
        self.assertIn("hallucination_injection", CORRUPTION_TYPES)
        self.assertIn("over_confidence", CORRUPTION_TYPES)
        self.assertIn("omission_corruption", CORRUPTION_TYPES)
        self.assertIn("structure_corruption", CORRUPTION_TYPES)

    def test_generate_phase25_data(self):
        from src.finetune.v2.data_generator.phase2_5_dpo_pairs import (
            generate_phase25_data,
        )

        with tempfile.TemporaryDirectory() as td:
            count = generate_phase25_data(td, scale=0.01)
            self.assertGreater(count, 0)
            fpath = os.path.join(td, "phase25_dpo_pairs.jsonl")
            self.assertTrue(os.path.isfile(fpath))
            with open(fpath) as f:
                lines = f.readlines()
            self.assertGreater(len(lines), 0)


class TestPhase35InsightGenerator(unittest.TestCase):
    """Tests for Phase 3.5 insight data generator."""

    def test_generate_insight_examples(self):
        from src.finetune.v2.data_generator.phase3_5_insights import (
            generate_insight_examples,
        )

        examples = generate_insight_examples(count=20)
        self.assertEqual(len(examples), 20)
        for ex in examples:
            self.assertIn("text", ex)
            self.assertIn("<think>", ex["text"])
            self.assertIn("</think>", ex["text"])
            self.assertIn("<insight", ex["text"])

    def test_all_7_categories_covered(self):
        from src.finetune.v2.data_generator.phase3_5_insights import (
            INSIGHT_CATEGORIES,
        )

        self.assertEqual(len(INSIGHT_CATEGORIES), 7)
        self.assertIn("holistic_synthesis", INSIGHT_CATEGORIES)
        self.assertIn("risk_assessment", INSIGHT_CATEGORIES)
        self.assertIn("pattern_recognition", INSIGHT_CATEGORIES)
        self.assertIn("anomaly_detection", INSIGHT_CATEGORIES)
        self.assertIn("trend_analysis", INSIGHT_CATEGORIES)
        self.assertIn("comparative_analysis", INSIGHT_CATEGORIES)
        self.assertIn("gap_analysis", INSIGHT_CATEGORIES)

    def test_generate_phase35_data(self):
        from src.finetune.v2.data_generator.phase3_5_insights import (
            generate_phase35_data,
        )

        with tempfile.TemporaryDirectory() as td:
            count = generate_phase35_data(td, scale=0.01)
            self.assertGreater(count, 0)
            fpath = os.path.join(td, "phase35_insights.jsonl")
            self.assertTrue(os.path.isfile(fpath))


class TestPhase37HolisticGenerator(unittest.TestCase):
    """Tests for Phase 3.7 holistic reasoning data generator."""

    def test_generate_holistic_examples(self):
        from src.finetune.v2.data_generator.phase3_7_holistic import (
            generate_holistic_examples,
        )

        examples = generate_holistic_examples(count=20)
        self.assertEqual(len(examples), 20)
        for ex in examples:
            self.assertIn("text", ex)
            self.assertIn("<think>", ex["text"])
            self.assertIn("</think>", ex["text"])

    def test_all_4_modes_covered(self):
        from src.finetune.v2.data_generator.phase3_7_holistic import (
            REASONING_MODES,
        )

        self.assertEqual(len(REASONING_MODES), 4)
        self.assertIn("intent_decomposition", REASONING_MODES)
        self.assertIn("evidence_synthesis", REASONING_MODES)
        self.assertIn("depth_calibration", REASONING_MODES)
        self.assertIn("domain_reasoning", REASONING_MODES)

    def test_generate_phase37_data(self):
        from src.finetune.v2.data_generator.phase3_7_holistic import (
            generate_phase37_data,
        )

        with tempfile.TemporaryDirectory() as td:
            count = generate_phase37_data(td, scale=0.01)
            self.assertGreater(count, 0)
            fpath = os.path.join(td, "phase37_holistic.jsonl")
            self.assertTrue(os.path.isfile(fpath))


class TestPostTrainingGenerators(unittest.TestCase):
    """Tests for post-training data generators and eval suite."""

    def test_conversational_dpo(self):
        from src.finetune.v2.data_generator.post_conversational import (
            generate_conversational_dpo,
        )

        pairs = generate_conversational_dpo(count=10)
        self.assertEqual(len(pairs), 10)
        for pair in pairs:
            self.assertIn("prompt", pair)
            self.assertIn("chosen", pair)
            self.assertIn("rejected", pair)

    def test_confidence_calibration(self):
        from src.finetune.v2.data_generator.post_confidence import (
            generate_confidence_examples,
        )

        examples = generate_confidence_examples(count=10)
        self.assertEqual(len(examples), 10)
        for ex in examples:
            self.assertIn("text", ex)
            text_lower = ex["text"].lower()
            self.assertTrue(
                "confidence" in text_lower,
                "Expected 'confidence' in example text",
            )

    def test_eval_suite(self):
        from src.finetune.v2.data_generator.eval_suite import (
            generate_eval_suite,
            BENCHMARKS,
        )

        suite = generate_eval_suite()
        self.assertEqual(len(suite), 500)

        benchmarks_found = {ex["benchmark"] for ex in suite}
        self.assertIn("TableBench", benchmarks_found)
        self.assertIn("HalluBench", benchmarks_found)
        self.assertIn("SynthesisEval", benchmarks_found)

        # Verify counts match
        for name, expected_count in BENCHMARKS.items():
            actual = sum(1 for ex in suite if ex["benchmark"] == name)
            self.assertEqual(actual, expected_count, f"{name} count mismatch")

    def test_eval_suite_frozen(self):
        from src.finetune.v2.data_generator.eval_suite import generate_eval_suite

        run1 = generate_eval_suite()
        run2 = generate_eval_suite()
        self.assertEqual(len(run1), len(run2))
        for a, b in zip(run1, run2):
            self.assertEqual(a, b, "Eval suite is not deterministic")


if __name__ == "__main__":
    unittest.main()
