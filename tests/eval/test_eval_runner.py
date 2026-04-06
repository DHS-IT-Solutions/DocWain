"""Tests for the evaluation runner."""

import json
import os
import tempfile

from eval.eval_runner import run_single_eval, run_eval, load_test_bank


def test_run_single_eval_captures_facts():
    case = {
        "id": "test_001",
        "category": "simple_lookup",
        "query": "What is the revenue?",
        "expected_facts": ["revenue", "1.5M"],
        "negative_facts": ["projected"],
        "expected_doc_ids": [],
    }

    def mock_ask(query, profile_id=None, subscription_id=None):
        return {"answer": "The total revenue is 1.5M for the period.", "source_doc_ids": []}

    result = run_single_eval(case, mock_ask)
    assert result["status"] == "ok"
    assert result["fact_coverage"] == 1.0
    assert result["hallucination_count"] == 0


def test_run_single_eval_detects_hallucination():
    case = {
        "id": "test_002",
        "category": "simple_lookup",
        "query": "What is the revenue?",
        "expected_facts": ["revenue"],
        "negative_facts": ["projected"],
        "expected_doc_ids": [],
    }

    def mock_ask(query, profile_id=None, subscription_id=None):
        return {"answer": "The projected revenue is $2M.", "source_doc_ids": []}

    result = run_single_eval(case, mock_ask)
    assert result["hallucination_count"] == 1
    assert "projected" in result["hallucinated_facts"]


def test_run_single_eval_handles_errors():
    case = {
        "id": "test_003",
        "category": "simple_lookup",
        "query": "test",
        "expected_facts": [],
        "negative_facts": [],
        "expected_doc_ids": [],
    }

    def failing_ask(query, **kwargs):
        raise ConnectionError("LLM unavailable")

    result = run_single_eval(case, failing_ask)
    assert result["status"] == "error"
    assert "LLM unavailable" in result["error"]


def test_run_single_eval_partial_fact_coverage():
    case = {
        "id": "test_004",
        "category": "multi_doc",
        "query": "Compare 2023 and 2024",
        "expected_facts": ["salary", "2023", "2024", "change"],
        "negative_facts": [],
        "expected_doc_ids": [],
    }

    def mock_ask(query, profile_id=None, subscription_id=None):
        return {"answer": "The salary in 2023 was $50k.", "source_doc_ids": []}

    result = run_single_eval(case, mock_ask)
    assert result["status"] == "ok"
    assert result["fact_coverage"] == 0.5  # 2 of 4 facts found


def test_run_single_eval_retrieval_recall():
    case = {
        "id": "test_005",
        "category": "simple_lookup",
        "query": "test",
        "expected_facts": [],
        "negative_facts": [],
        "expected_doc_ids": ["doc_1", "doc_2", "doc_3"],
    }

    def mock_ask(query, profile_id=None, subscription_id=None):
        return {"answer": "Some answer.", "source_doc_ids": ["doc_1", "doc_3"]}

    result = run_single_eval(case, mock_ask)
    assert result["status"] == "ok"
    assert abs(result["retrieval_recall"] - 2 / 3) < 1e-9


def test_load_test_bank():
    bank = load_test_bank("eval/test_bank.json")
    assert len(bank) == 4
    assert bank[0]["id"] == "eval_001"


def test_run_eval_saves_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a minimal test bank
        bank_path = os.path.join(tmpdir, "bank.json")
        with open(bank_path, "w") as f:
            json.dump(
                [
                    {
                        "id": "e1",
                        "category": "simple_lookup",
                        "query": "test?",
                        "expected_facts": ["test"],
                        "negative_facts": [],
                        "expected_doc_ids": [],
                    }
                ],
                f,
            )

        out_dir = os.path.join(tmpdir, "results")

        def mock_ask(query, profile_id=None, subscription_id=None):
            return {"answer": "This is a test response.", "source_doc_ids": []}

        report = run_eval(bank_path, mock_ask, output_dir=out_dir)
        assert report["summary"]["total_cases"] == 1
        assert report["summary"]["passed"] == 1
        assert os.path.isfile(report["output_file"])
