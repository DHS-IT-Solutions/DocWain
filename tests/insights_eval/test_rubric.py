import pytest

from tests.insights_eval.rubric import (
    PrecisionRecallRubric,
    RubricResult,
    score_precision_recall,
)


def test_precision_recall_perfect_match():
    expected = [{"id": "a"}, {"id": "b"}]
    actual = [{"id": "a"}, {"id": "b"}]
    result = score_precision_recall(actual, expected, key="id")
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.passed(min_precision=0.7, min_recall=0.6) is True


def test_precision_recall_partial():
    expected = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    actual = [{"id": "a"}, {"id": "x"}]  # 1 hit, 1 miss
    result = score_precision_recall(actual, expected, key="id")
    assert result.precision == 0.5
    assert result.recall == pytest.approx(1 / 3, abs=1e-6)


def test_passed_threshold():
    r = RubricResult(precision=0.5, recall=0.4)
    assert r.passed(min_precision=0.7, min_recall=0.6) is False
    assert r.passed(min_precision=0.4, min_recall=0.3) is True
