"""Scoring rubrics for Claude Code as judge.

Each rubric defines a 1-5 scale for evaluating model outputs across eight
quality dimensions used throughout the DocWain V2+ training pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Rubric definitions
# ---------------------------------------------------------------------------

_RUBRICS: Dict[str, str] = {
    "synthesis_coherence": (
        "Evaluate how well the model synthesises information from multiple sources "
        "into a coherent, logically structured response.\n"
        "5: Seamless synthesis — all sources integrated into a unified narrative with "
        "clear logical flow and no contradictions.\n"
        "4: Strong synthesis — most sources integrated well with minor gaps in "
        "logical connection.\n"
        "3: Adequate synthesis — sources referenced but not fully integrated; some "
        "disjoint sections.\n"
        "2: Weak synthesis — sources largely treated in isolation with superficial "
        "connections.\n"
        "1: No synthesis — response ignores available sources or presents "
        "contradictory information without resolution."
    ),
    "intent_alignment": (
        "Evaluate how accurately the model identifies and responds to the user's "
        "underlying intent, including implicit requirements.\n"
        "5: Perfect alignment — addresses explicit and implicit intent with "
        "appropriate scope and depth.\n"
        "4: Strong alignment — captures primary intent with minor misses on "
        "secondary requirements.\n"
        "3: Partial alignment — addresses the surface-level question but misses "
        "important implicit needs.\n"
        "2: Weak alignment — partially off-topic or addresses only a fragment of "
        "the request.\n"
        "1: Misaligned — response does not address the user's question or intent."
    ),
    "depth_calibration": (
        "Evaluate whether the response provides an appropriate level of detail — "
        "neither too shallow nor unnecessarily verbose.\n"
        "5: Perfectly calibrated — depth matches the complexity of the query with "
        "every detail serving a purpose.\n"
        "4: Well calibrated — mostly appropriate depth with minor over- or under-"
        "explanation.\n"
        "3: Moderately calibrated — some sections too shallow or too detailed for "
        "the query.\n"
        "2: Poorly calibrated — significantly too brief or excessively verbose.\n"
        "1: Uncalibrated — response length and depth bear no relation to the "
        "query's requirements."
    ),
    "conversation_quality": (
        "Evaluate the overall quality of the model's conversational behaviour "
        "including tone, helpfulness, and naturalness.\n"
        "5: Excellent — natural, helpful, well-structured, and contextually "
        "appropriate throughout.\n"
        "4: Good — generally helpful and natural with minor awkwardness.\n"
        "3: Acceptable — functional but somewhat mechanical or formulaic.\n"
        "2: Poor — unnatural, unhelpful, or tonally inappropriate in places.\n"
        "1: Very poor — robotic, confusing, or actively unhelpful."
    ),
    "confidence_calibration": (
        "Evaluate whether the model's expressed confidence matches the actual "
        "reliability of its claims.\n"
        "5: Perfectly calibrated — confident assertions are correct, uncertainty "
        "is expressed where appropriate.\n"
        "4: Well calibrated — minor miscalibration but generally reliable "
        "confidence signals.\n"
        "3: Moderately calibrated — some overconfident claims or unnecessary "
        "hedging.\n"
        "2: Poorly calibrated — frequently overconfident on wrong answers or "
        "underconfident on correct ones.\n"
        "1: Uncalibrated — confidence bears no relation to accuracy."
    ),
    "extraction_accuracy": (
        "Evaluate the accuracy of information extraction from documents, including "
        "tables, figures, and text.\n"
        "5: Flawless extraction — all relevant information captured with perfect "
        "fidelity.\n"
        "4: Accurate extraction — minor omissions or imprecisions that do not "
        "affect conclusions.\n"
        "3: Adequate extraction — most key information captured but with notable "
        "gaps or errors.\n"
        "2: Inaccurate extraction — significant errors or omissions in extracted "
        "information.\n"
        "1: Failed extraction — critical information missed or substantially "
        "incorrect."
    ),
    "tool_correctness": (
        "Evaluate whether the model selects the correct tools and provides "
        "accurate arguments for function calls.\n"
        "5: Perfect tool use — correct tool selected with all arguments accurate "
        "and well-formed.\n"
        "4: Good tool use — correct tool with minor argument imprecisions.\n"
        "3: Adequate tool use — right tool but some arguments incorrect or "
        "missing.\n"
        "2: Poor tool use — wrong tool selected or major argument errors.\n"
        "1: Failed tool use — no tool call when needed, or completely incorrect "
        "tool and arguments."
    ),
    "insight_quality": (
        "Evaluate the quality and actionability of insights generated from "
        "document analysis.\n"
        "5: Exceptional insights — novel, actionable, well-supported by evidence, "
        "and clearly articulated.\n"
        "4: Good insights — relevant and useful with adequate supporting "
        "evidence.\n"
        "3: Adequate insights — reasonable observations but lacking novelty or "
        "clear evidence.\n"
        "2: Weak insights — superficial or obvious observations with little "
        "analytical value.\n"
        "1: No useful insights — irrelevant, unsupported, or trivially obvious "
        "statements."
    ),
}

RUBRIC_NAMES: List[str] = sorted(_RUBRICS.keys())


def get_rubric(name: str) -> str:
    """Return the full rubric text for *name*.

    Raises :class:`ValueError` if the rubric name is not recognised.
    """
    if name not in _RUBRICS:
        raise ValueError(
            f"Unknown rubric {name!r}. Available: {', '.join(RUBRIC_NAMES)}"
        )
    return _RUBRICS[name]


def score_with_rubric(
    rubric_name: str,
    model_output: str,
    reference: str,
    context: str = "",
) -> Dict[str, Any]:
    """Score *model_output* against *reference* using heuristic rubric scoring.

    Heuristics applied:
    - Base score of 3.
    - +1 if output contains ``<think>`` blocks (reasoning traces).
    - +0.5 if output contains citations (``[source``, ``[ref``, ``[doc``).
    - +0.5 if output contains confidence statements (``confidence``, ``certain``).
    - -1 if output is less than 20% of reference length (too short).
    - -1 if output is more than 300% of reference length (too long).
    - Final score clamped to [1, 5].

    Returns
    -------
    dict with keys ``score`` (int 1-5), ``reasoning`` (str), ``rubric_name`` (str).
    """
    _ = get_rubric(rubric_name)  # validate rubric exists

    score = 3.0
    reasons: list[str] = []

    # Reasoning traces
    if "<think>" in model_output:
        score += 1.0
        reasons.append("+1 for <think> reasoning block")

    # Citations
    output_lower = model_output.lower()
    if any(tag in output_lower for tag in ("[source", "[ref", "[doc")):
        score += 0.5
        reasons.append("+0.5 for citations")

    # Confidence statements
    if any(word in output_lower for word in ("confidence", "certain")):
        score += 0.5
        reasons.append("+0.5 for confidence statements")

    # Length calibration relative to reference
    ref_len = max(len(reference), 1)
    out_len = len(model_output)
    ratio = out_len / ref_len

    if ratio < 0.20:
        score -= 1.0
        reasons.append("-1 for output too short relative to reference")
    elif ratio > 3.0:
        score -= 1.0
        reasons.append("-1 for output too long relative to reference")

    # Clamp
    score = int(max(1, min(5, round(score))))

    reasoning = "; ".join(reasons) if reasons else "Base score — no heuristic adjustments"

    return {
        "score": score,
        "reasoning": reasoning,
        "rubric_name": rubric_name,
    }
