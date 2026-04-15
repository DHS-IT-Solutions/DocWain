import pytest
from unittest.mock import MagicMock, patch


def test_judge_prompt_includes_five_dimensions():
    from src.finetune.sprint.judge import JUDGE_SYSTEM_PROMPT, DIMENSIONS

    assert len(DIMENSIONS) == 5
    for dim in DIMENSIONS:
        assert dim in JUDGE_SYSTEM_PROMPT


def test_parse_judge_response_valid():
    from src.finetune.sprint.judge import parse_judge_response

    raw = '{"accuracy": 4.5, "completeness": 3.8, "reasoning": 4.0, "honesty": 4.2, "format": 3.5}'
    scores = parse_judge_response(raw)
    assert scores["accuracy"] == 4.5
    assert scores["honesty"] == 4.2
    assert len(scores) == 5


def test_parse_judge_response_extracts_from_text():
    from src.finetune.sprint.judge import parse_judge_response

    raw = 'Here are my scores:\n```json\n{"accuracy": 4.0, "completeness": 3.5, "reasoning": 4.0, "honesty": 3.0, "format": 4.0}\n```'
    scores = parse_judge_response(raw)
    assert scores["accuracy"] == 4.0


def test_parse_judge_response_invalid_returns_none():
    from src.finetune.sprint.judge import parse_judge_response

    scores = parse_judge_response("This is not valid JSON at all")
    assert scores is None


def test_score_response_returns_all_dimensions():
    from src.finetune.sprint.judge import score_response

    mock_scores = {"accuracy": 4.0, "completeness": 3.5, "reasoning": 4.2, "honesty": 3.8, "format": 4.0}

    with patch("src.finetune.sprint.judge._call_judge", return_value=mock_scores):
        result = score_response(
            prompt="Extract info from this invoice",
            response="Invoice #123, Total: $500",
            reference={"expected_answer": "Invoice #123, Total: $500"},
        )

    assert result["accuracy"] == 4.0
    assert result["honesty"] == 3.8
    assert result["average"] == pytest.approx(3.9)


def test_evaluate_batch():
    from src.finetune.sprint.judge import evaluate_batch

    mock_scores = {"accuracy": 4.0, "completeness": 4.0, "reasoning": 4.0, "honesty": 4.0, "format": 4.0}

    examples = [
        {"prompt": "q1", "reference": {"expected_answer": "a1"}},
        {"prompt": "q2", "reference": {"expected_answer": "a2"}},
    ]
    responses = ["a1", "a2"]

    with patch("src.finetune.sprint.judge.score_response", return_value={**mock_scores, "average": 4.0}):
        results = evaluate_batch(examples, responses)

    assert len(results) == 2
    assert results[0]["average"] == 4.0


def test_check_regression_detects_drop():
    from src.finetune.sprint.judge import check_regression

    previous = {"accuracy": 4.2, "completeness": 4.0, "reasoning": 4.1, "honesty": 3.8, "format": 4.0}
    current = {"accuracy": 4.1, "completeness": 3.5, "reasoning": 4.0, "honesty": 3.7, "format": 4.0}

    regressions = check_regression(previous, current, threshold=0.3)
    assert "completeness" in regressions
    assert "accuracy" not in regressions
