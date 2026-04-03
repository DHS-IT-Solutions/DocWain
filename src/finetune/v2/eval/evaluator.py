"""Track evaluator — queries Ollama and scores responses against test bank.

The rubric scorers return dicts of dimension scores on a 1.0-5.0 scale.
This evaluator converts them to a single 0-100 composite score for
gate-checking and comparison.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any, Dict, List, Optional

from src.finetune.v2.eval.rubrics import TRACK_SCORERS
from src.finetune.v2.eval.test_bank import get_test_bank

logger = logging.getLogger(__name__)


def query_ollama(prompt: str, model: str = "DHS/DocWain", timeout: int = 120) -> str:
    """Send a prompt to Ollama and return the response text.

    Parameters
    ----------
    prompt:
        The user query to send.
    model:
        Ollama model name.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    Model response string, or empty string on failure.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.warning("Ollama returned code %d: %s", result.returncode, result.stderr)
        return ""
    except FileNotFoundError:
        logger.warning("ollama CLI not found")
        return ""
    except subprocess.TimeoutExpired:
        logger.warning("Ollama query timed out after %ds", timeout)
        return ""


def _rubric_dict_to_score(rubric_result) -> float:
    """Convert a rubric scorer result to a 0-100 composite score.

    The rubric scorers return either:
    - A dict of dimension scores on a 1.0-5.0 scale
    - A float (0-100) directly

    This normalises to 0-100 in all cases.
    """
    if isinstance(rubric_result, (int, float)):
        return float(rubric_result)
    if isinstance(rubric_result, dict):
        values = [v for v in rubric_result.values() if isinstance(v, (int, float))]
        if not values:
            return 0.0
        avg_1_5 = sum(values) / len(values)
        # Map 1.0-5.0 to 0-100
        return max(0.0, min(100.0, (avg_1_5 - 1.0) / 4.0 * 100.0))
    return 0.0


class TrackEvaluator:
    """Evaluates a model against the test bank for one or more tracks.

    Parameters
    ----------
    model_name:
        Ollama model name to query.
    """

    def __init__(self, model_name: str = "DHS/DocWain") -> None:
        self.model_name = model_name

    def evaluate_track(self, track: str) -> Dict[str, Any]:
        """Evaluate all test bank examples for a single track.

        Returns
        -------
        Dict with track, scores (list of 0-100 floats), avg_score,
        pass_rate, per_example details.
        """
        bank = get_test_bank(track)
        tests = bank[track]
        scorer = TRACK_SCORERS.get(track)
        if scorer is None:
            raise ValueError(f"No scorer found for track: {track!r}")

        per_example: List[Dict[str, Any]] = []
        scores: List[float] = []

        for test in tests:
            query = test["query"]
            reference = test["reference"]
            test_id = test["id"]

            logger.info("Evaluating %s: %s", test_id, query[:60])
            response = query_ollama(query, model=self.model_name)

            if not response:
                score = 0.0
                raw_rubric = {}
                logger.warning("Empty response for %s", test_id)
            else:
                raw_rubric = scorer(response, reference)
                score = _rubric_dict_to_score(raw_rubric)

            scores.append(score)
            per_example.append({
                "id": test_id,
                "query": query,
                "response_length": len(response),
                "score": score,
                "rubric_dimensions": raw_rubric if isinstance(raw_rubric, dict) else {},
            })
            logger.info("  %s score: %.1f", test_id, score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        pass_rate = sum(1 for s in scores if s >= 70.0) / len(scores) if scores else 0.0

        return {
            "track": track,
            "num_examples": len(tests),
            "avg_score": avg_score,
            "pass_rate": pass_rate,
            "scores": scores,
            "per_example": per_example,
        }

    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all tracks and return combined results.

        Returns
        -------
        Dict with per_track results, overall_avg, and overall_pass_rate.
        """
        per_track: Dict[str, Dict[str, Any]] = {}
        all_scores: List[float] = []

        for track in TRACK_SCORERS:
            result = self.evaluate_track(track)
            per_track[track] = result
            all_scores.extend(result["scores"])

        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        overall_pass = (
            sum(1 for s in all_scores if s >= 70.0) / len(all_scores)
            if all_scores
            else 0.0
        )

        return {
            "per_track": per_track,
            "overall_avg": overall_avg,
            "overall_pass_rate": overall_pass,
            "num_total": len(all_scores),
        }
