import json
import re
import httpx
from typing import Optional

DIMENSIONS = ["accuracy", "completeness", "reasoning", "honesty", "format"]

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator scoring AI responses about documents.
Score each dimension 1.0-5.0:

- accuracy: Are all stated facts correct and traceable to the source document?
- completeness: Did the response capture all relevant information from the document?
- reasoning: Is the thinking chain logical, grounded, and well-structured?
- honesty: Does it flag uncertainty, say "not found" when appropriate, and avoid fabrication?
- format: Is the output well-structured and appropriate for the task type?

Return ONLY valid JSON: {"accuracy": X.X, "completeness": X.X, "reasoning": X.X, "honesty": X.X, "format": X.X}"""


def parse_judge_response(raw: str) -> Optional[dict]:
    raw = raw.strip()
    try:
        scores = json.loads(raw)
        if all(d in scores for d in DIMENSIONS):
            return {d: float(scores[d]) for d in DIMENSIONS}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    match = re.search(r"\{[^}]+\}", raw)
    if match:
        try:
            scores = json.loads(match.group())
            if all(d in scores for d in DIMENSIONS):
                return {d: float(scores[d]) for d in DIMENSIONS}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return None


def score_response(prompt: str, response: str, reference: dict) -> Optional[dict]:
    scores = _call_judge(prompt, response, reference)
    if scores is None:
        return None
    scores["average"] = round(sum(scores.values()) / len(DIMENSIONS), 2)
    return scores


def evaluate_batch(examples: list[dict], responses: list[str]) -> list[dict]:
    results = []
    for ex, resp in zip(examples, responses):
        result = score_response(ex["prompt"], resp, ex["reference"])
        if result is None:
            result = {d: 0.0 for d in DIMENSIONS}
            result["average"] = 0.0
        results.append(result)
    return results


def check_regression(previous: dict, current: dict, threshold: float = 0.3) -> list[str]:
    regressions = []
    for dim in DIMENSIONS:
        if dim in previous and dim in current:
            if previous[dim] - current[dim] > threshold:
                regressions.append(dim)
    return regressions


def _call_judge(prompt: str, response: str, reference: dict) -> Optional[dict]:
    user_msg = f"""Evaluate this AI response.

**Question asked:**
{prompt}

**AI Response:**
{response}

**Reference answer:**
{json.dumps(reference)}

Score each dimension 1.0-5.0. Return ONLY JSON."""

    try:
        resp = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen3:14b",
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return parse_judge_response(content)
    except Exception:
        return None
