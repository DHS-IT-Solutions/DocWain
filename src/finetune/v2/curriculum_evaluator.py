"""Curriculum evaluator -- LoRA inference, subagent judging, and gate checks.

Loads a trained LoRA adapter via Unsloth, runs inference against the frozen
test bank, dispatches subagent judging briefs, parses scored outputs, and
evaluates pass/fail against basics and production gate thresholds.

Usage::

    from src.finetune.v2.curriculum_evaluator import (
        run_lora_inference,
        JudgingBrief,
        parse_judge_scores,
        aggregate_scores,
        check_gates,
        build_failure_analysis,
    )

    responses = run_lora_inference(base_model, adapter_path, prompts)
    brief = JudgingBrief(examples=scored_examples, batch_index=0)
    judge_prompt = brief.to_prompt()
    scores = parse_judge_scores(raw_judge_output)
    agg = aggregate_scores(scores)
    gate = check_gates(agg)
    analysis = build_failure_analysis(scores)
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.finetune.v2.data_generator.base import DOCWAIN_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGE_DIMENSIONS: List[str] = [
    "factual_correctness",
    "reasoning_quality",
    "completeness",
    "grounding",
]

BASICS_AVG_THRESHOLD: float = 3.5
BASICS_MIN_DIMENSION: float = 3.0

PRODUCTION_AVG_THRESHOLD: float = 4.0
PRODUCTION_MIN_DIMENSION: float = 3.5

_DIMENSION_DESCRIPTIONS: Dict[str, str] = {
    "factual_correctness": (
        "Are the facts, numbers, and claims in the response accurate and "
        "consistent with the reference answer and source material?"
    ),
    "reasoning_quality": (
        "Is the step-by-step reasoning logical, coherent, and well-structured? "
        "Does the model justify its conclusions?"
    ),
    "completeness": (
        "Does the response address all parts of the question without "
        "significant omissions?"
    ),
    "grounding": (
        "Are claims anchored to specific evidence from the document/context? "
        "Does the model cite sources rather than assert unsupported conclusions?"
    ),
}

_SCORING_GUIDELINES: str = (
    "5.0 = Excellent — near-perfect on this dimension\n"
    "4.0 = Good — minor issues that do not meaningfully harm quality\n"
    "3.0 = Acceptable — noticeable issues but core answer is correct\n"
    "2.0 = Poor — significant errors that reduce usefulness\n"
    "1.0 = Failing — incorrect, missing, or completely off-target"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Outcome of evaluating a model checkpoint against pass/fail thresholds."""

    basics_passed: bool
    production_passed: bool
    overall_avg: float
    min_dimension: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgingBrief:
    """Bundle of examples sent to a subagent judge for scoring."""

    examples: List[Dict[str, Any]]
    batch_index: int

    def to_prompt(self) -> str:
        """Build a structured judging prompt for a subagent LLM."""
        lines: List[str] = []

        lines.append("# Judging Task")
        lines.append(
            "You are an expert evaluator for an enterprise document intelligence model. "
            "Score each example below on four dimensions using a 1.0–5.0 scale."
        )
        lines.append("")

        # Dimensions
        lines.append("## Scoring Dimensions")
        for dim in JUDGE_DIMENSIONS:
            lines.append(f"### {dim}")
            lines.append(_DIMENSION_DESCRIPTIONS[dim])
        lines.append("")

        # Scale guidelines
        lines.append("## Scoring Guidelines (1.0–5.0 scale)")
        lines.append(_SCORING_GUIDELINES)
        lines.append("")

        # Examples
        lines.append("## Examples to Score")
        for idx, ex in enumerate(self.examples):
            lines.append(f"### Example {idx}")
            lines.append(f"**Track:** {ex.get('track', 'unknown')}")
            lines.append(f"**Category:** {ex.get('category', 'unknown')}")
            lines.append(f"**Difficulty:** {ex.get('difficulty', 'unknown')}")
            lines.append(f"**Prompt:**\n{ex.get('prompt', '')}")
            lines.append(f"**Model Response:**\n{ex.get('response', '')}")
            ref = ex.get("reference", {})
            ref_str = json.dumps(ref) if isinstance(ref, dict) else str(ref)
            lines.append(f"**Reference Answer:**\n{ref_str}")
            lines.append("")

        # Output format
        lines.append("## Output Format")
        lines.append(
            "Return a JSON array — one object per example — with this schema:\n"
            "```json\n"
            '[\n  {"example_index": 0, "scores": {"factual_correctness": <float>, '
            '"reasoning_quality": <float>, "completeness": <float>, "grounding": <float>}},\n'
            "  ...\n"
            "]\n"
            "```\n"
            "Output ONLY the JSON array. Do not include explanatory prose outside the array."
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LoRA inference
# ---------------------------------------------------------------------------


def run_lora_inference(
    base_model: str,
    adapter_path: str,
    prompts: List[str],
    max_new_tokens: int = 2048,
) -> List[str]:
    """Load a LoRA adapter via Unsloth and run batch inference.

    Loads the base model + adapter, generates responses for each prompt using
    the DocWain system prompt, then frees GPU memory before returning.

    Args:
        base_model: HuggingFace model ID or local path for the base model.
        adapter_path: Path to the saved LoRA adapter directory.
        prompts: List of user prompts to run inference on.
        max_new_tokens: Maximum tokens to generate per prompt.

    Returns:
        List of generated response strings, one per input prompt.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore
        import torch
    except ImportError as exc:
        raise ImportError(
            "unsloth and torch are required for run_lora_inference. "
            "Install them with: pip install unsloth torch"
        ) from exc

    logger.info("Loading model from %s (base: %s)", adapter_path, base_model)
    # Load directly from the checkpoint path — if it's a merged FP16 checkpoint,
    # the LoRA weights are already merged in. If it's a LoRA adapter directory,
    # Unsloth will load the base + adapter automatically.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    responses: List[str] = []

    for prompt in prompts:
        messages = [
            {"role": "system", "content": DOCWAIN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)

    # Free GPU memory
    del model
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    logger.info("Inference complete — %d responses generated", len(responses))
    return responses


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


def parse_judge_scores(raw_text: str) -> List[Dict[str, Any]]:
    """Parse a JSON array of scores from subagent judge output.

    Handles markdown code fences (```json ... ```) gracefully.

    Args:
        raw_text: Raw text output from the subagent judge.

    Returns:
        List of score dicts, each with ``example_index`` and ``scores`` keys.
        Returns an empty list if parsing fails.
    """
    text = raw_text.strip()

    # Strip markdown fences if present
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        text = match.group(1).strip()

    # Find the JSON array even if there is surrounding text
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        text = array_match.group(0)

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            logger.warning("parse_judge_scores: expected list, got %s", type(parsed))
            return []
        return parsed
    except json.JSONDecodeError as exc:
        logger.error("parse_judge_scores: JSON decode error — %s", exc)
        return []


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------


def aggregate_scores(all_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group scores by track and compute per-dimension averages.

    Also adds top-level ``overall_avg`` and ``min_dimension`` keys that
    summarise across all tracks and dimensions.

    Args:
        all_scores: List of score dicts.  Each dict must have ``track`` and
            ``scores`` keys.  The ``scores`` value maps dimension names to
            float scores.

    Returns:
        Dict keyed by track name (plus ``overall_avg`` and ``min_dimension``).
        Each track value is a dict of dimension → average score.
    """
    # Accumulate raw values per track per dimension
    bucket: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for entry in all_scores:
        track = entry.get("track", "unknown")
        scores = entry.get("scores", {})
        for dim, val in scores.items():
            try:
                bucket[track][dim].append(float(val))
            except (TypeError, ValueError):
                logger.warning("Skipping non-numeric score for %s/%s: %r", track, dim, val)

    # Compute averages per track
    result: Dict[str, Any] = {}
    all_dim_avgs: List[float] = []

    for track, dims in bucket.items():
        track_avgs: Dict[str, float] = {}
        for dim, vals in dims.items():
            avg = sum(vals) / len(vals) if vals else 0.0
            track_avgs[dim] = round(avg, 4)
            all_dim_avgs.append(avg)
        result[track] = track_avgs

    overall_avg = round(sum(all_dim_avgs) / len(all_dim_avgs), 4) if all_dim_avgs else 0.0
    min_dimension = round(min(all_dim_avgs), 4) if all_dim_avgs else 0.0

    result["overall_avg"] = overall_avg
    result["min_dimension"] = min_dimension

    return result


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def check_gates(aggregated: Dict[str, Any]) -> GateResult:
    """Evaluate aggregated scores against basics and production gates.

    Args:
        aggregated: Output of :func:`aggregate_scores`.

    Returns:
        :class:`GateResult` with pass/fail flags and summary stats.
    """
    overall_avg: float = float(aggregated.get("overall_avg", 0.0))
    min_dimension: float = float(aggregated.get("min_dimension", 0.0))

    basics_passed = (
        overall_avg >= BASICS_AVG_THRESHOLD
        and min_dimension >= BASICS_MIN_DIMENSION
    )
    production_passed = (
        overall_avg >= PRODUCTION_AVG_THRESHOLD
        and min_dimension >= PRODUCTION_MIN_DIMENSION
    )

    # Collect per-track details (exclude meta keys)
    details = {k: v for k, v in aggregated.items() if k not in ("overall_avg", "min_dimension")}

    return GateResult(
        basics_passed=basics_passed,
        production_passed=production_passed,
        overall_avg=overall_avg,
        min_dimension=min_dimension,
        details=details,
    )


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------


def build_failure_analysis(
    all_scores: List[Dict[str, Any]],
    threshold: float = 3.5,
) -> Dict[str, Any]:
    """Identify weak track+dimension pairs and cluster failure patterns.

    Groups entries by ``(track, dimension)`` pairs that fall below *threshold*,
    then collects low-scoring examples for each weak area to expose patterns
    that should drive data augmentation.

    Args:
        all_scores: List of score dicts with ``track``, ``category``,
            ``difficulty``, ``prompt``, ``response``, and ``scores`` keys.
        threshold: Dimension average below which an area is considered weak.

    Returns:
        Dict with:
        - ``weak_areas``: list of dicts, each with ``track``, ``dimension``,
          ``avg_score``, ``count``, ``examples`` (list of low-scoring prompts /
          responses), ``categories``, ``difficulties``.
        - ``total_augmentation_count``: total number of low-scoring examples
          across all weak areas.
    """
    # Accumulate raw values and examples per (track, dimension)
    dim_vals: Dict[tuple, List[float]] = defaultdict(list)
    dim_examples: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    for entry in all_scores:
        track = entry.get("track", "unknown")
        scores = entry.get("scores", {})
        for dim, val in scores.items():
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            key = (track, dim)
            dim_vals[key].append(fval)
            if fval < threshold:
                dim_examples[key].append({
                    "prompt": entry.get("prompt", ""),
                    "response": entry.get("response", ""),
                    "category": entry.get("category", ""),
                    "difficulty": entry.get("difficulty", ""),
                    "score": fval,
                })

    weak_areas: List[Dict[str, Any]] = []
    total_augmentation_count = 0

    for (track, dim), vals in dim_vals.items():
        avg = sum(vals) / len(vals) if vals else 0.0
        if avg < threshold:
            low_examples = dim_examples[(track, dim)]
            categories = list({ex["category"] for ex in low_examples if ex["category"]})
            difficulties = list({ex["difficulty"] for ex in low_examples if ex["difficulty"]})
            weak_areas.append({
                "track": track,
                "dimension": dim,
                "avg_score": round(avg, 4),
                "count": len(low_examples),
                "examples": low_examples,
                "categories": categories,
                "difficulties": difficulties,
            })
            total_augmentation_count += len(low_examples)

    # Sort by avg_score ascending so the worst areas come first
    weak_areas.sort(key=lambda x: x["avg_score"])

    return {
        "weak_areas": weak_areas,
        "total_augmentation_count": total_augmentation_count,
    }
