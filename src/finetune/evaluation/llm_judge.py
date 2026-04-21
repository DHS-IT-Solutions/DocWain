"""LLM-judge evaluation pipeline for DocWain V2.

Replaces heuristic (regex/keyword) scoring with an LLM-based judge that scores
model responses on four dimensions: factual_correctness, reasoning_quality,
completeness, grounding.

Fallback chain:
  1. vLLM at http://localhost:8100 (OpenAI-compatible /v1/chat/completions)
  2. Ollama at http://localhost:11434 (native /api/chat)
  3. Heuristic scoring (keyword overlap)

Usage::

    from src.finetune.evaluation.llm_judge import (
        score_response,
        evaluate_model,
        run_evaluation_suite,
    )

    result = score_response(prompt, response, reference)
    full   = run_evaluation_suite(output_dir="eval_results")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGE_DIMENSIONS = [
    "factual_correctness",
    "reasoning_quality",
    "completeness",
    "grounding",
]

DEFAULT_VLLM_URL = "http://localhost:8100/v1/chat/completions"
DEFAULT_VLLM_MODEL = "docwain"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "qwen3:14b"

QUERY_TIMEOUT = 60   # seconds per model query
JUDGE_TIMEOUT = 30   # seconds per judge call

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator scoring AI responses about documents. "
    "Return ONLY the requested JSON with no additional text."
)

JUDGE_USER_TEMPLATE = """\
You are an expert evaluator scoring AI responses about documents.

QUESTION: {prompt}
AI RESPONSE: {response}
REFERENCE ANSWER: {reference}

Score the response on these 4 dimensions (1.0-5.0):
- factual_correctness: Are facts accurate and consistent with reference?
- reasoning_quality: Is reasoning logical and well-justified?
- completeness: Does it address all parts of the question?
- grounding: Are claims anchored to specific evidence?

Return ONLY a JSON object:
{{"factual_correctness": X.X, "reasoning_quality": X.X, "completeness": X.X, "grounding": X.X}}"""


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: Dict[str, Any], timeout: int) -> Optional[Dict[str, Any]]:
    """POST *payload* as JSON to *url* and return the parsed response dict.

    Returns ``None`` on any network or parse error.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        logger.debug("HTTP error calling %s: %s", url, exc)
        return None
    except Exception as exc:
        logger.debug("Unexpected error calling %s: %s", url, exc)
        return None


def _vllm_chat(
    messages: List[Dict[str, str]],
    url: str,
    model: str,
    timeout: int,
) -> Optional[str]:
    """Call a vLLM (OpenAI-compatible) chat endpoint; return the assistant text."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 256,
    }
    result = _post_json(url, payload, timeout)
    if result is None:
        return None
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.debug("vLLM response parse error: %s — raw: %s", exc, result)
        return None


def _ollama_chat(
    messages: List[Dict[str, str]],
    url: str,
    model: str,
    timeout: int,
) -> Optional[str]:
    """Call an Ollama chat endpoint; return the assistant text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 256},
    }
    result = _post_json(url, payload, timeout)
    if result is None:
        return None
    try:
        return result["message"]["content"]
    except (KeyError, TypeError) as exc:
        logger.debug("Ollama response parse error: %s — raw: %s", exc, result)
        return None


def _query_model_vllm(
    prompt: str,
    url: str,
    model: str,
    system_prompt: str = "",
    timeout: int = QUERY_TIMEOUT,
) -> Optional[str]:
    """Query a vLLM endpoint and return the generated text, or None."""
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    result = _post_json(url, payload, timeout)
    if result is None:
        return None
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Judge score parsing
# ---------------------------------------------------------------------------


def _parse_scores_from_text(raw: str) -> Optional[Dict[str, float]]:
    """Extract a dimension→float dict from *raw* LLM judge output.

    Handles markdown fences and surrounding prose gracefully.
    """
    text = raw.strip()

    # Strip markdown code fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # Find the first JSON object
    obj_match = re.search(r"\{[^{}]*\}", text)
    if obj_match:
        text = obj_match.group(0)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("_parse_scores_from_text: JSON decode failed on: %r", raw[:200])
        return None

    if not isinstance(parsed, dict):
        return None

    scores: Dict[str, float] = {}
    for dim in JUDGE_DIMENSIONS:
        val = parsed.get(dim)
        if val is None:
            return None
        try:
            scores[dim] = max(1.0, min(5.0, float(val)))
        except (TypeError, ValueError):
            return None

    return scores


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------


def _heuristic_score(prompt: str, response: str, reference: Any) -> Dict[str, float]:
    """Simple keyword-overlap heuristic when all LLM judges are unavailable."""
    ref_str = json.dumps(reference) if isinstance(reference, dict) else str(reference)
    ref_tokens = set(re.findall(r"\w+", ref_str.lower()))
    resp_tokens = set(re.findall(r"\w+", response.lower()))

    if not ref_tokens:
        overlap = 0.5
    else:
        overlap = len(ref_tokens & resp_tokens) / len(ref_tokens)

    # Map overlap [0,1] → score [1,5]
    base = 1.0 + overlap * 4.0

    return {
        "factual_correctness": round(base, 2),
        "reasoning_quality": round(base * 0.9, 2),
        "completeness": round(base * 0.95, 2),
        "grounding": round(base * 0.85, 2),
    }


# ---------------------------------------------------------------------------
# Public API: score_response
# ---------------------------------------------------------------------------


def score_response(
    prompt: str,
    response: str,
    reference: Any,
    judge_url: str = DEFAULT_VLLM_URL,
    judge_model: str = DEFAULT_VLLM_MODEL,
) -> Dict[str, Any]:
    """Score a single (prompt, response, reference) triple using an LLM judge.

    Tries vLLM first, then Ollama, then heuristics.

    Args:
        prompt: The original question / user prompt.
        response: The model's generated response to evaluate.
        reference: Reference answer (string or dict).
        judge_url: vLLM chat completions endpoint URL.
        judge_model: vLLM model name / alias.

    Returns:
        Dict with key ``"scores"`` mapping each of the four dimension names
        to a float in [1.0, 5.0].  Also includes ``"judge_source"`` indicating
        which backend produced the scores (``"vllm"``, ``"ollama"``,
        or ``"heuristic"``).
    """
    ref_str = json.dumps(reference) if isinstance(reference, dict) else str(reference)

    user_msg = JUDGE_USER_TEMPLATE.format(
        prompt=prompt,
        response=response,
        reference=ref_str,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # --- Attempt 1: vLLM ---
    raw = _vllm_chat(messages, judge_url, judge_model, JUDGE_TIMEOUT)
    if raw is not None:
        scores = _parse_scores_from_text(raw)
        if scores is not None:
            return {"scores": scores, "judge_source": "vllm"}
        logger.debug("vLLM judge returned unparseable output; trying Ollama")

    # --- Attempt 2: Ollama ---
    ollama_messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw_ollama = _ollama_chat(
        ollama_messages, DEFAULT_OLLAMA_URL, DEFAULT_OLLAMA_MODEL, JUDGE_TIMEOUT
    )
    if raw_ollama is not None:
        scores = _parse_scores_from_text(raw_ollama)
        if scores is not None:
            return {"scores": scores, "judge_source": "ollama"}
        logger.debug("Ollama judge returned unparseable output; falling back to heuristic")

    # --- Attempt 3: Heuristic ---
    logger.warning("All LLM judges unavailable; using heuristic scoring")
    return {"scores": _heuristic_score(prompt, response, reference), "judge_source": "heuristic"}


# ---------------------------------------------------------------------------
# Public API: evaluate_model
# ---------------------------------------------------------------------------


def evaluate_model(
    model_url: str,
    model_name: str,
    test_bank: List[Dict[str, Any]],
    judge_url: str = DEFAULT_VLLM_URL,
    judge_model: str = DEFAULT_VLLM_MODEL,
    max_examples: int = 50,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Query *model_url* with test bank examples and score every response.

    Args:
        model_url: OpenAI-compatible chat completions URL for the model under test.
        model_name: Model name/alias to include in the request payload.
        test_bank: List of example dicts (track, category, prompt, reference, difficulty).
        judge_url: LLM judge endpoint (vLLM-compatible).
        judge_model: Judge model name.
        max_examples: Cap on how many examples to evaluate.
        output_dir: If given, save detailed results JSON here.

    Returns:
        Aggregated result dict compatible with
        :func:`src.finetune.v2.curriculum_evaluator.aggregate_scores`, plus
        extra keys ``per_example`` (list), ``judge_sources`` (counter dict),
        and ``metadata``.
    """
    examples = test_bank[:max_examples]
    total = len(examples)
    all_scored: List[Dict[str, Any]] = []
    judge_sources: Dict[str, int] = defaultdict(int)

    for idx, ex in enumerate(examples, start=1):
        print(f"Evaluating {idx}/{total}...", flush=True)

        prompt = ex.get("prompt", "")
        reference = ex.get("reference", "")
        track = ex.get("track", "unknown")
        category = ex.get("category", "unknown")
        difficulty = ex.get("difficulty", "medium")

        # Query model under test
        response = _query_model_vllm(prompt, model_url, model_name, timeout=QUERY_TIMEOUT)
        if response is None:
            logger.warning("Model query failed for example %d/%d; using empty response", idx, total)
            response = ""

        # Score response
        print(f"Scoring {idx}/{total}...", flush=True)
        scored = score_response(prompt, response, reference, judge_url, judge_model)
        judge_sources[scored.get("judge_source", "unknown")] += 1

        all_scored.append({
            "track": track,
            "category": category,
            "difficulty": difficulty,
            "prompt": prompt,
            "response": response,
            "reference": reference,
            "scores": scored["scores"],
            "judge_source": scored.get("judge_source", "unknown"),
        })

    # Aggregate using the same logic as curriculum_evaluator.aggregate_scores
    aggregated = _aggregate_scores(all_scored)
    aggregated["per_example"] = all_scored
    aggregated["judge_sources"] = dict(judge_sources)
    aggregated["metadata"] = {
        "model_url": model_url,
        "model_name": model_name,
        "judge_url": judge_url,
        "judge_model": judge_model,
        "total_examples": total,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
    }

    # Persist detailed results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"eval_{model_name}_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(aggregated, fh, indent=2)
        logger.info("Detailed results saved to %s", out_path)
        print(f"Results saved → {out_path}", flush=True)

    return aggregated


# ---------------------------------------------------------------------------
# Internal aggregation (mirrors curriculum_evaluator.aggregate_scores)
# ---------------------------------------------------------------------------


def _aggregate_scores(all_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-track dimension averages plus overall_avg and min_dimension."""
    bucket: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for entry in all_scores:
        track = entry.get("track", "unknown")
        scores = entry.get("scores", {})
        for dim, val in scores.items():
            try:
                bucket[track][dim].append(float(val))
            except (TypeError, ValueError):
                pass

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
# Public API: run_evaluation_suite
# ---------------------------------------------------------------------------


def run_evaluation_suite(
    output_dir: str = "eval_results",
    model_url: str = DEFAULT_VLLM_URL,
    model_name: str = DEFAULT_VLLM_MODEL,
    judge_url: Optional[str] = None,
    judge_model: str = DEFAULT_VLLM_MODEL,
    max_examples: int = 50,
) -> Dict[str, Any]:
    """End-to-end evaluation: load test bank → query model → score → aggregate → save.

    Args:
        output_dir: Directory where the detailed JSON report is written.
        model_url: Chat completions URL for the model under test.
        model_name: Model name/alias.
        judge_url: Judge endpoint; defaults to *model_url* when ``None``.
        judge_model: Judge model name/alias.
        max_examples: Maximum test bank examples to evaluate.

    Returns:
        Aggregated results dict (same format as :func:`evaluate_model`).
    """
    from src.finetune.v2.eval.test_bank import get_test_bank  # lazy import

    if judge_url is None:
        judge_url = model_url

    print("Loading test bank...", flush=True)
    bank = get_test_bank()
    print(f"Test bank loaded — {len(bank)} examples total (capping at {max_examples})", flush=True)

    results = evaluate_model(
        model_url=model_url,
        model_name=model_name,
        test_bank=bank,
        judge_url=judge_url,
        judge_model=judge_model,
        max_examples=max_examples,
        output_dir=output_dir,
    )

    # Console summary
    print("\n" + "=" * 60, flush=True)
    print("EVALUATION SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  Overall avg  : {results['overall_avg']:.3f}", flush=True)
    print(f"  Min dimension: {results['min_dimension']:.3f}", flush=True)
    print(f"  Examples     : {results['metadata']['total_examples']}", flush=True)
    print(f"  Judge sources: {results['judge_sources']}", flush=True)
    print("\nPer-track breakdown:", flush=True)
    for key, val in results.items():
        if key in ("overall_avg", "min_dimension", "per_example", "judge_sources", "metadata"):
            continue
        if isinstance(val, dict):
            dim_str = "  ".join(f"{d}={v:.2f}" for d, v in val.items())
            print(f"  {key:20s}  {dim_str}", flush=True)
    print("=" * 60, flush=True)

    return results
