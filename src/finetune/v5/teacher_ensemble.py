"""Five-teacher voting ensemble for V5 data generation.

Teachers
--------
1. DocWain-V3 (local vLLM :8100)             — in-domain schema + extraction
2. HF muthugsubramanian/DocWain-14B-v2       — regression floor
3. Ollama DocWain                            — consistency check
4. Nvidia Nemotron-3-Super-120B via Qubrid   — frontier tiebreaker
5. Claude (hand-written seed)                — narrative + identity (offline)

Claude's contribution is NOT a per-row API call (100K rows * 4 teachers
of network calls is already the bottleneck). Claude's contribution is
the prompt templates encoded in the Capability Charter and hand-written
seed exemplars committed under ``data/v5_seeds/``. At ensemble voting
time the runtime voices are four: V3, HF-V2, Ollama, Nemotron.

Voting rule
-----------
Each teacher is asked the same prompt. Responses are normalised to a
canonical shape (JSON-parsed where the capability demands it, whitespace-
collapsed otherwise). A row is accepted if:

    * ≥3 of 4 primary voices agree on the canonical response, OR
    * 2 voices agree AND Nemotron (called only then) sides with them.

Rows accepted with full agreement are high-confidence (ship).
Rows accepted via Nemotron tiebreak are medium-confidence (ship,
logged as such for later audit).
Rows with no majority are quarantined — never shipped to training.

This module is network-heavy and intentionally synchronous per row so
failures surface cleanly. The data_generator orchestrator parallelises
across rows with an asyncio worker pool.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical response shape
# ---------------------------------------------------------------------------

@dataclass
class TeacherResponse:
    teacher: str
    raw_text: str
    parsed: Any  # Canonicalised form (dict for structured, str for free-text)
    ok: bool
    latency_s: float
    error: Optional[str] = None


@dataclass
class EnsembleVote:
    accepted: bool
    confidence: str  # "high" | "medium" | "quarantine"
    consensus: Any  # The agreed-on canonical response, None on quarantine
    teacher_responses: List[TeacherResponse] = field(default_factory=list)
    agreement_count: int = 0
    primary_voice_count: int = 0
    nemotron_tiebreak: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks, common on Qwen3/Nemotron."""
    return _THINK_RE.sub("", text or "").strip()


def _canonicalise(raw_text: str, expect_json: bool) -> Any:
    """Normalise a teacher response for agreement comparison."""
    text = _strip_thinking(raw_text or "")
    text = text.strip()
    if not text:
        return None
    if expect_json:
        # Try to parse JSON out of the response; tolerate surrounding text
        json_candidate = text
        # Strip ```json fences
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            json_candidate = "\n".join(lines).strip()
        # Find the first { and last } if there's extra prose
        if "{" in json_candidate and "}" in json_candidate:
            start = json_candidate.find("{")
            end = json_candidate.rfind("}")
            json_candidate = json_candidate[start : end + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            return None
    # Free-text: lowercase + whitespace-normalize for agreement comparison
    return _WS_RE.sub(" ", text.lower()).strip()


def _fingerprint(value: Any) -> str:
    """Stable hash of a canonical response, for agreement bucketing."""
    try:
        serialized = json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        serialized = repr(value)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Teacher clients
# ---------------------------------------------------------------------------


def _call_vllm(prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> TeacherResponse:
    """Local DocWain-V3 via the running vLLM instance."""
    t0 = time.monotonic()
    try:
        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager()
        if not mgr.health_check():
            return TeacherResponse("v3_vllm", "", None, False, time.monotonic() - t0, "vllm_down")
        raw = mgr.query(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            require_vllm=True,
        )
        return TeacherResponse("v3_vllm", raw or "", None, bool(raw), time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001
        return TeacherResponse("v3_vllm", "", None, False, time.monotonic() - t0, str(exc)[:200])


def _call_hf(prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> TeacherResponse:
    """HF muthugsubramanian/DocWain-14B-v2 via local Transformers.

    Reuses the same weights we already have at ``models/DocWain-14B-v2/``.
    Loaded lazily and cached on the module so successive calls reuse the
    same pipeline instance — otherwise we'd pay model-load cost per row.
    """
    t0 = time.monotonic()
    global _HF_PIPE  # lazy cache
    try:
        if "_HF_PIPE" not in globals() or _HF_PIPE is None:
            # Defer import so vllm_manager callers don't pay this cost.
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
            model_path = os.getenv(
                "DOCWAIN_HF_TEACHER_PATH",
                "/home/ubuntu/PycharmProjects/DocWain/models/DocWain-14B-v2",
            )
            if not os.path.exists(model_path):
                return TeacherResponse(
                    "hf_v2", "", None, False, time.monotonic() - t0,
                    f"hf_teacher_path_missing:{model_path}",
                )
            tok = AutoTokenizer.from_pretrained(model_path)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
            )
            _HF_PIPE = pipeline("text-generation", model=mdl, tokenizer=tok)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        result = _HF_PIPE(
            messages, max_new_tokens=max_tokens, temperature=temperature,
            do_sample=temperature > 0,
        )
        # pipeline returns [{"generated_text": [messages...]}]
        gen = result[0]["generated_text"]
        raw = gen[-1]["content"] if isinstance(gen, list) else str(gen)
        return TeacherResponse("hf_v2", raw or "", None, bool(raw), time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001
        return TeacherResponse("hf_v2", "", None, False, time.monotonic() - t0, str(exc)[:200])


def _call_ollama(prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> TeacherResponse:
    """Ollama-hosted DocWain via local HTTP."""
    t0 = time.monotonic()
    try:
        from urllib import request as urllib_request
        from urllib.error import HTTPError, URLError
        model_name = os.getenv("DOCWAIN_OLLAMA_MODEL", "docwain")
        body = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib_request.Request(
            "http://localhost:11434/api/generate",
            data=body, headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        raw = data.get("response", "")
        return TeacherResponse("ollama_docwain", raw, None, bool(raw), time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001
        return TeacherResponse("ollama_docwain", "", None, False, time.monotonic() - t0, str(exc)[:200])


def _call_nemotron(prompt: str, system_prompt: str, max_tokens: int, temperature: float) -> TeacherResponse:
    """Nvidia Nemotron-3-Super-120B via Qubrid OpenAI-compatible endpoint.

    The smoke test showed Nemotron has an internal thinking pass that
    can exhaust a small token budget before emitting visible output. We
    always allocate at least 1500 tokens to give it room and strip any
    leading whitespace / think blocks from the response.
    """
    t0 = time.monotonic()
    # Clamp up — Nemotron burns tokens on internal reasoning
    effective_max_tokens = max(max_tokens, 1500)
    try:
        from openai import OpenAI
        api_key = os.getenv(
            "QUBRID_API_KEY",
            "k_afdba890b39c.kUoS2Y7G7i8I-6U-vpCu4OFT0nm0pP1Fimdh9OfLV_ii8UE1DxqQKA",
        )
        client = OpenAI(base_url="https://platform.qubrid.com/v1", api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        r = client.chat.completions.create(
            model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=temperature,
        )
        raw = r.choices[0].message.content or ""
        finish = r.choices[0].finish_reason
        if finish == "length" and not raw.strip():
            return TeacherResponse(
                "nemotron", raw, None, False, time.monotonic() - t0,
                "length_truncated_pre_output",
            )
        return TeacherResponse("nemotron", raw, None, bool(raw), time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001
        return TeacherResponse("nemotron", "", None, False, time.monotonic() - t0, str(exc)[:200])


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------


def vote(
    prompt: str,
    *,
    expect_json: bool = False,
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.1,
    call_nemotron_on_tie: bool = True,
    teacher_callers: Optional[List[Callable[..., TeacherResponse]]] = None,
) -> EnsembleVote:
    """Run the ensemble on a single prompt and return the voting outcome.

    ``expect_json`` drives canonicalisation — when True, responses must
    parse to the same JSON object to count as agreeing; when False, a
    whitespace-collapsed lower-cased string match is used.

    ``call_nemotron_on_tie`` keeps the Nemotron API call off the hot path
    for rows where the primary voices already agree. The pilot run uses
    True unconditionally to exercise the tiebreak branch.
    """
    primary = teacher_callers or [_call_vllm, _call_hf, _call_ollama]
    responses: List[TeacherResponse] = []
    for caller in primary:
        r = caller(prompt, system_prompt, max_tokens, temperature)
        r.parsed = _canonicalise(r.raw_text, expect_json)
        responses.append(r)

    # Bucket primary voices by canonical fingerprint
    buckets: Dict[str, List[TeacherResponse]] = {}
    for r in responses:
        if not r.ok or r.parsed is None:
            continue
        buckets.setdefault(_fingerprint(r.parsed), []).append(r)

    ordered = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    primary_voice_count = sum(len(v) for v in buckets.values())

    # High-confidence path: ≥3 primary voices agree (of 3 primaries → all 3,
    # since the caller list defaults to 3 entries). We still keep the rule
    # expressed as ≥3 so the ensemble can grow with an extra primary in
    # the future without changing voting semantics.
    if ordered and len(ordered[0][1]) >= 3:
        winning = ordered[0][1]
        return EnsembleVote(
            accepted=True,
            confidence="high",
            consensus=winning[0].parsed,
            teacher_responses=responses,
            agreement_count=len(winning),
            primary_voice_count=primary_voice_count,
            reason="≥3 primary voices agree",
        )

    # 2 agree — Nemotron tiebreak if enabled
    if ordered and len(ordered[0][1]) == 2 and call_nemotron_on_tie:
        nemotron_r = _call_nemotron(prompt, system_prompt, max_tokens, temperature)
        nemotron_r.parsed = _canonicalise(nemotron_r.raw_text, expect_json)
        responses.append(nemotron_r)

        winning_fingerprint = ordered[0][0]
        if nemotron_r.ok and nemotron_r.parsed is not None \
                and _fingerprint(nemotron_r.parsed) == winning_fingerprint:
            winning = ordered[0][1]
            return EnsembleVote(
                accepted=True,
                confidence="medium",
                consensus=winning[0].parsed,
                teacher_responses=responses,
                agreement_count=len(winning) + 1,
                primary_voice_count=primary_voice_count,
                nemotron_tiebreak=True,
                reason="2 primary voices + Nemotron agree",
            )
        return EnsembleVote(
            accepted=False,
            confidence="quarantine",
            consensus=None,
            teacher_responses=responses,
            agreement_count=len(ordered[0][1]),
            primary_voice_count=primary_voice_count,
            nemotron_tiebreak=True,
            reason="2 primary agree but Nemotron disagrees — quarantine",
        )

    return EnsembleVote(
        accepted=False,
        confidence="quarantine",
        consensus=None,
        teacher_responses=responses,
        agreement_count=(len(ordered[0][1]) if ordered else 0),
        primary_voice_count=primary_voice_count,
        reason="no primary majority and nemotron skipped",
    )


__all__ = [
    "TeacherResponse",
    "EnsembleVote",
    "vote",
]
