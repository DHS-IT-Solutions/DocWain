"""Mechanical rubrics for capability eval gates.

Every gate scoring is pass/fail by script — no human judgment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass
class RubricResult:
    precision: float = 0.0
    recall: float = 0.0
    notes: str = ""

    def passed(self, *, min_precision: float, min_recall: float) -> bool:
        return self.precision >= min_precision and self.recall >= min_recall


@dataclass
class PrecisionRecallRubric:
    min_precision: float
    min_recall: float


def score_precision_recall(
    actual: Sequence[Mapping],
    expected: Sequence[Mapping],
    *,
    key: str,
) -> RubricResult:
    expected_keys = {e[key] for e in expected}
    actual_keys = {a[key] for a in actual}
    if not actual_keys:
        precision = 1.0 if not expected_keys else 0.0
    else:
        precision = len(actual_keys & expected_keys) / len(actual_keys)
    if not expected_keys:
        recall = 1.0
    else:
        recall = len(actual_keys & expected_keys) / len(expected_keys)
    return RubricResult(precision=precision, recall=recall)
