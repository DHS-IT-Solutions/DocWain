"""Phase transition gate checker for DocWain V2+ training pipeline.

Every phase and post-training round defines minimum quality thresholds that
must be satisfied before the pipeline advances.  For most metrics a *lower
bound* applies (the metric must be **at least** the threshold).  A small set
of metrics — hallucination rate, false-positive rate, ECE, and quality drop —
use an *upper bound* (the metric must be **at most** the threshold).
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------

PHASE_GATES: Dict[str, Dict[str, float]] = {
    "phase1": {"cosine_sim": 0.60, "caption_bleu": 0.15},
    "phase2": {"docvqa_accuracy": 0.75, "table_f1": 0.80, "layout_map": 0.70},
    "phase2_5": {"hallucination_rate": 0.05, "extraction_f1_improvement": 0.05},
    "phase3": {
        "tool_accuracy": 0.85,
        "arg_correctness": 0.90,
        "false_positive_rate": 0.10,
    },
    "phase3_5": {"insight_precision": 0.80, "insight_recall": 0.60},
    "phase3_7": {
        "synthesis_coherence": 0.80,
        "intent_alignment": 0.85,
        "depth_calibration": 0.75,
        "domain_accuracy": 0.80,
    },
    "phase4": {"regression_pass_rate": 0.90},
    "round1": {"conversation_quality": 0.80},
    "round2": {"ece": 0.10},
    "round3": {"quality_drop": 0.03, "inference_speed_toks": 25.0},
}

# Metrics where lower is better (upper-bound check).
_UPPER_BOUND_METRICS = {"hallucination_rate", "false_positive_rate", "ece", "quality_drop"}


def check_gates(phase: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    """Check whether *metrics* satisfy the gates for *phase*.

    Parameters
    ----------
    phase:
        Phase identifier (e.g. ``"phase2"``, ``"round1"``).
    metrics:
        Metric name -> value mapping produced by the eval runner.

    Returns
    -------
    dict with keys:
        - ``passed`` (bool): ``True`` when all gates are satisfied.
        - ``phase`` (str): The phase that was checked.
        - ``failures`` (list[str]): Metric names that failed their gate.
        - ``details`` (dict): Per-metric pass/fail and values.
    """
    if phase not in PHASE_GATES:
        raise ValueError(
            f"Unknown phase {phase!r}. Available: {', '.join(sorted(PHASE_GATES))}"
        )

    gates = PHASE_GATES[phase]
    failures: List[str] = []
    details: Dict[str, Any] = {}

    for metric_name, threshold in gates.items():
        value = metrics.get(metric_name)
        if value is None:
            failures.append(metric_name)
            details[metric_name] = {
                "passed": False,
                "threshold": threshold,
                "value": None,
                "reason": "metric missing",
            }
            continue

        if metric_name in _UPPER_BOUND_METRICS:
            passed = value <= threshold
        else:
            passed = value >= threshold

        if not passed:
            failures.append(metric_name)

        details[metric_name] = {
            "passed": passed,
            "threshold": threshold,
            "value": value,
        }

    return {
        "passed": len(failures) == 0,
        "phase": phase,
        "failures": failures,
        "details": details,
    }
