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


if __name__ == "__main__":
    unittest.main()
