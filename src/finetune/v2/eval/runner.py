"""Benchmark runner for DocWain V2+ evaluation.

Scores a list of examples using a model function, then aggregates results
into phase-level metrics suitable for the gate checker.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from .rubrics import RUBRIC_NAMES, score_with_rubric

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark -> rubric mapping
# ---------------------------------------------------------------------------

_BENCHMARK_RUBRIC_MAP: Dict[str, str] = {
    "docvqa": "extraction_accuracy",
    "table_extraction": "extraction_accuracy",
    "tool_calling": "tool_correctness",
    "insight_generation": "insight_quality",
    "synthesis": "synthesis_coherence",
    "intent": "intent_alignment",
    "depth": "depth_calibration",
    "conversation": "conversation_quality",
    "confidence": "confidence_calibration",
}


def run_eval_on_examples(
    examples: List[Dict[str, Any]],
    model_fn: Callable[[str], str],
) -> List[Dict[str, Any]]:
    """Run evaluation on a list of examples using *model_fn*.

    Each example dict must have at minimum:
    - ``prompt`` (str): Input to pass to the model.
    - ``reference`` (str): Expected/reference output.
    - ``benchmark`` (str): Benchmark name used to select the scoring rubric.

    Optional:
    - ``context`` (str): Additional context for scoring.

    Returns a list of result dicts, one per example, each containing the
    original example fields plus ``score``, ``reasoning``, and ``rubric_name``.
    """
    results: List[Dict[str, Any]] = []

    for example in examples:
        prompt = example.get("prompt", "")
        reference = example.get("reference", "")
        benchmark = example.get("benchmark", "synthesis")
        context = example.get("context", "")

        # Generate model output
        try:
            model_output = model_fn(prompt)
        except Exception as exc:
            logger.warning("Model function failed on prompt: %s", exc)
            model_output = ""

        # Select rubric
        rubric_name = _BENCHMARK_RUBRIC_MAP.get(benchmark, "synthesis_coherence")

        # Score
        score_result = score_with_rubric(
            rubric_name=rubric_name,
            model_output=model_output,
            reference=reference,
            context=context,
        )

        result = {**example, **score_result, "model_output": model_output}
        results.append(result)

    return results


def compute_phase_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate eval results into phase-level metrics normalised to 0-1.

    Groups results by benchmark, computes mean score per benchmark, and
    normalises the 1-5 rubric scale to 0.0-1.0 via ``(mean - 1) / 4``.

    Returns a dict mapping benchmark names to their normalised score.
    """
    from collections import defaultdict

    benchmark_scores: Dict[str, List[float]] = defaultdict(list)

    for result in results:
        benchmark = result.get("benchmark", "unknown")
        score = result.get("score")
        if score is not None:
            benchmark_scores[benchmark].append(float(score))

    metrics: Dict[str, float] = {}
    for benchmark, scores in benchmark_scores.items():
        if scores:
            mean_score = sum(scores) / len(scores)
            # Normalise 1-5 scale to 0-1
            normalised = (mean_score - 1.0) / 4.0
            metrics[benchmark] = round(max(0.0, min(1.0, normalised)), 4)

    return metrics
