"""Track evaluator -- queries Ollama and scores responses against the test bank.

Queries the model via the Ollama HTTP API, scores responses using the
programmatic rubrics, and returns aggregate results with gate-check verdicts.

Usage::

    from src.finetune.v2.eval.evaluator import TrackEvaluator

    ev = TrackEvaluator(model_name="DHS/DocWain")
    result = ev.evaluate_track("excel_csv")
    all_results = ev.evaluate_all_tracks()
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.finetune.v2.data_generator.base import DOCWAIN_SYSTEM_PROMPT
from src.finetune.v2.eval.rubrics import TRACK_SCORERS
from src.finetune.v2.eval.test_bank import get_test_bank

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate thresholds  (1.0-5.0 scale, per-dimension average)
# ---------------------------------------------------------------------------

GATE_THRESHOLDS: Dict[str, float] = {
    "excel_csv": 4.0,
    "layout": 4.0,
    "ocr_vision": 4.0,
    "reasoning": 4.0,
    "kg": 3.8,
    "visualization": 4.0,
}

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 120  # seconds


# ---------------------------------------------------------------------------
# Standalone query helper (used by autonomous_trainer)
# ---------------------------------------------------------------------------


def query_ollama(
    prompt: str,
    *,
    model: str = "DHS/DocWain",
    system: Optional[str] = None,
    timeout: int = OLLAMA_TIMEOUT,
) -> str:
    """Query Ollama and return the response text.

    Convenience function for callers that don't need a full TrackEvaluator.
    Returns empty string on any error.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system or DOCWAIN_SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 4096},
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            response = body.get("response", "")
            # Qwen3 thinking models may put content in "thinking" field
            # and leave "response" empty when num_predict is exhausted
            if not response.strip() and body.get("thinking"):
                response = body["thinking"]
            return response
    except Exception as exc:
        logger.warning("query_ollama failed (model=%s): %s", model, exc)
        return ""


# ---------------------------------------------------------------------------
# Model query
# ---------------------------------------------------------------------------

class TrackEvaluator:
    """Evaluates a model against the frozen test bank for one or more tracks.

    Parameters
    ----------
    model_name:
        Ollama model name to query (e.g. ``DHS/DocWain``).
    ollama_url:
        Base URL for the Ollama generate endpoint.
    temperature:
        Sampling temperature for model queries.
    max_tokens:
        Maximum tokens to generate per response.
    """

    def __init__(
        self,
        model_name: str = "DHS/DocWain",
        ollama_url: str = OLLAMA_URL,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Model interaction
    # ------------------------------------------------------------------

    def query_model(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Query Ollama model and return response text.

        Sends a POST to ``/api/generate`` with ``stream: false`` and
        returns the full response string.  On any error returns an
        empty string so evaluation can continue.
        """
        sys_prompt = system_prompt or DOCWAIN_SYSTEM_PROMPT
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": sys_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                response = body.get("response", "")
                # Qwen3 thinking models may put content in "thinking" field
                if not response.strip() and body.get("thinking"):
                    response = body["thinking"]
                return response
        except (URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
            logger.warning("Ollama query failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Per-track evaluation
    # ------------------------------------------------------------------

    def _check_model_available(self) -> bool:
        """Verify the Ollama model exists before running full eval."""
        test_response = self.query_model("Hello")
        if not test_response:
            logger.error(
                "Model %s is not responding — check Ollama model exists",
                self.model_name,
            )
            return False
        return True

    def evaluate_track(self, track_name: str) -> dict:
        """Run all test bank examples for a track, score them, return aggregate.

        Returns
        -------
        dict with keys:
            track, dimensions (dim_name -> avg_score), overall_avg,
            passed (bool), per_example, weak_categories
        """
        scorer = TRACK_SCORERS.get(track_name)
        if scorer is None:
            raise ValueError(f"Unknown track {track_name!r}")

        if not self._check_model_available():
            raise RuntimeError(
                f"Model {self.model_name!r} not available in Ollama. "
                f"GGUF export or ollama create may have failed."
            )

        examples = get_test_bank(track=track_name)
        gate = GATE_THRESHOLDS.get(track_name, 4.0)

        # Accumulators
        per_example: List[Dict[str, Any]] = []
        dimension_sums: Dict[str, float] = {}
        dimension_counts: Dict[str, int] = {}
        category_scores: Dict[str, List[float]] = {}

        for idx, example in enumerate(examples):
            prompt = example["prompt"]
            reference = example["reference"]
            category = example["category"]

            logger.info(
                "[%s %d/%d] category=%s difficulty=%s",
                track_name, idx + 1, len(examples),
                category, example.get("difficulty", "?"),
            )

            t0 = time.monotonic()
            response = self.query_model(prompt)
            elapsed = time.monotonic() - t0

            if not response:
                # Score everything at 1.0 (minimum) on failure
                scores = {dim: 1.0 for dim in _default_dims(track_name)}
                logger.warning("  Empty response (%.1fs)", elapsed)
            else:
                scores = scorer(response, reference)
                logger.info(
                    "  scored in %.1fs: %s",
                    elapsed,
                    {k: f"{v:.1f}" for k, v in scores.items()},
                )

            # Accumulate dimensions
            for dim, val in scores.items():
                dimension_sums[dim] = dimension_sums.get(dim, 0.0) + val
                dimension_counts[dim] = dimension_counts.get(dim, 0) + 1

            # Accumulate per-category
            example_avg = sum(scores.values()) / max(len(scores), 1)
            category_scores.setdefault(category, []).append(example_avg)

            per_example.append({
                "prompt": prompt[:120] + ("..." if len(prompt) > 120 else ""),
                "response_len": len(response),
                "scores": scores,
                "category": category,
                "difficulty": example.get("difficulty", "medium"),
                "elapsed_s": round(elapsed, 1),
            })

        # Aggregate dimensions
        dimensions: Dict[str, float] = {}
        for dim in sorted(dimension_sums):
            dimensions[dim] = round(
                dimension_sums[dim] / max(dimension_counts[dim], 1), 2
            )

        overall_avg = round(
            sum(dimensions.values()) / max(len(dimensions), 1), 2
        )
        passed = overall_avg >= gate

        # Identify weak categories (avg below 3.5)
        weak_categories: List[str] = []
        for cat, cat_scores in sorted(category_scores.items()):
            cat_avg = sum(cat_scores) / len(cat_scores)
            if cat_avg < 3.5:
                weak_categories.append(cat)

        return {
            "track": track_name,
            "dimensions": dimensions,
            "overall_avg": overall_avg,
            "passed": passed,
            "gate_threshold": gate,
            "num_examples": len(examples),
            "per_example": per_example,
            "weak_categories": weak_categories,
        }

    # ------------------------------------------------------------------
    # All-tracks evaluation
    # ------------------------------------------------------------------

    def evaluate_all_tracks(self) -> dict:
        """Evaluate all 6 tracks and return combined results.

        Returns
        -------
        dict with keys:
            per_track (track_name -> track_result),
            overall_avg, all_passed (bool),
            tracks_passed, tracks_failed
        """
        per_track: Dict[str, dict] = {}
        track_avgs: List[float] = []
        tracks_passed: List[str] = []
        tracks_failed: List[str] = []

        for track_name in sorted(TRACK_SCORERS.keys()):
            logger.info("=" * 60)
            logger.info("Evaluating track: %s", track_name)
            logger.info("=" * 60)
            result = self.evaluate_track(track_name)
            per_track[track_name] = result
            track_avgs.append(result["overall_avg"])
            if result["passed"]:
                tracks_passed.append(track_name)
            else:
                tracks_failed.append(track_name)

        overall_avg = round(
            sum(track_avgs) / max(len(track_avgs), 1), 2
        )

        return {
            "per_track": per_track,
            "overall_avg": overall_avg,
            "all_passed": len(tracks_failed) == 0,
            "tracks_passed": tracks_passed,
            "tracks_failed": tracks_failed,
            "total_examples": sum(
                r["num_examples"] for r in per_track.values()
            ),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRACK_DIMS = {
    "excel_csv": ["tabular_qa_accuracy", "cross_sheet_reasoning",
                  "data_type_correctness", "aggregation_accuracy"],
    "layout": ["structure_accuracy", "relationship_extraction",
               "noise_robustness", "completeness_score"],
    "ocr_vision": ["printed_accuracy", "handwriting_accuracy",
                   "diagram_understanding", "image_table_reconstruction",
                   "overlay_handling"],
    "reasoning": ["reasoning_depth", "evidence_grounding",
                  "synthesis_coherence"],
    "kg": ["entity_usage", "relationship_reasoning", "citation_accuracy"],
    "visualization": ["trigger_judgment", "spec_correctness",
                      "data_accuracy", "type_selection"],
}


def _default_dims(track_name: str) -> List[str]:
    """Return the dimension names for a track (for default scoring on failure)."""
    return _TRACK_DIMS.get(track_name, ["score"])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Evaluate DocWain V2 model across all tracks"
    )
    parser.add_argument("--model", default="DHS/DocWain")
    parser.add_argument("--track", default=None, help="Evaluate a single track")
    parser.add_argument("--url", default=OLLAMA_URL)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    evaluator = TrackEvaluator(model_name=args.model, ollama_url=args.url)

    if args.track:
        result = evaluator.evaluate_track(args.track)
        _print_track_result(result)
        return 0 if result["passed"] else 1
    else:
        results = evaluator.evaluate_all_tracks()
        for track_name in sorted(results["per_track"]):
            _print_track_result(results["per_track"][track_name])
        print(f"\n{'=' * 60}")
        print(f"Overall Average: {results['overall_avg']:.2f}")
        print(f"All Passed:      {results['all_passed']}")
        if results["tracks_failed"]:
            print(f"Failed Tracks:   {', '.join(results['tracks_failed'])}")
        print(f"Total Examples:  {results['total_examples']}")
        print(f"{'=' * 60}")
        return 0 if results["all_passed"] else 1


def _print_track_result(result: dict) -> None:
    """Pretty-print a single track evaluation result."""
    status = "PASS" if result["passed"] else "FAIL"
    print(f"\n--- {result['track']} ({status}) ---")
    print(f"  Overall: {result['overall_avg']:.2f} / {result['gate_threshold']}")
    for dim, score in sorted(result["dimensions"].items()):
        print(f"  {dim}: {score:.2f}")
    if result["weak_categories"]:
        print(f"  Weak: {', '.join(result['weak_categories'])}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
