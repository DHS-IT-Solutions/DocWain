"""Claude Distillation Engine — generates training examples using Claude as a teacher.

Uses the Anthropic API to produce high-quality SFT and DPO examples across 7
document-intelligence categories. Falls back to synthetic placeholders when no
API key is available so the pipeline stays testable offline.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import formatting helpers from existing base module
# ---------------------------------------------------------------------------

try:
    from src.finetune.v2.data_generator.base import (
        format_sft_example,
        format_dpo_example,
        DOCWAIN_SYSTEM_PROMPT,
    )
except ImportError:  # pragma: no cover — only triggered outside the installed package
    DOCWAIN_SYSTEM_PROMPT = (
        "You are DocWain, an enterprise document intelligence assistant. "
        "You analyse documents with deep contextual understanding, extract "
        "structured information, identify patterns and anomalies, and provide "
        "holistic analysis grounded in evidence. You reason step-by-step before "
        "answering, state your confidence level, and cite specific sources. "
        "When information is insufficient, you say so clearly rather than guessing."
    )

    def format_sft_example(
        query: str,
        reasoning: str,
        answer: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        sys_prompt = system_prompt or DOCWAIN_SYSTEM_PROMPT
        text = (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>"
        )
        return {"text": text}

    def format_dpo_example(
        query: str,
        chosen_reasoning: str,
        chosen_answer: str,
        rejected_reasoning: str,
        rejected_answer: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        sys_prompt = system_prompt or DOCWAIN_SYSTEM_PROMPT
        prompt = (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        chosen = f"<think>\n{chosen_reasoning}\n</think>\n\n{chosen_answer}<|im_end|>"
        rejected = f"<think>\n{rejected_reasoning}\n</think>\n\n{rejected_answer}<|im_end|>"
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


from src.finetune.sprint.document_factory import generate_document  # noqa: E402

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

DISTILL_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "completeness_extraction": {
        "system": (
            "You are an expert document analyst. Given a document, generate a "
            "question that asks to extract ALL relevant fields completely, "
            "then provide thorough step-by-step reasoning and a complete answer."
        ),
        "doc_types": ["invoice", "purchase_order", "contract", "government_form"],
        "is_dpo": False,
        "multi_doc": False,
    },
    "intent_context": {
        "system": (
            "You are an expert at understanding document intent and contextual meaning. "
            "Generate a question that requires interpreting the purpose, audience, and "
            "broader context of the document. Reason about intent before answering."
        ),
        "doc_types": ["policy", "compliance_report", "meeting_notes", "audit_report"],
        "is_dpo": False,
        "multi_doc": False,
    },
    "anti_hallucination": {
        "system": (
            "You are a careful document analyst trained to answer only what is "
            "explicitly supported by the document. Generate a question that could "
            "tempt hallucination, then provide a correct answer that acknowledges "
            "missing information rather than fabricating it. Also provide a rejected "
            "answer that confidently hallucinates details."
        ),
        "doc_types": ["medical_record", "legal_filing", "insurance_claim", "contract"],
        "is_dpo": True,
        "multi_doc": False,
    },
    "ocr_vision": {
        "system": (
            "You are an expert at processing degraded, scanned, or visually complex "
            "documents. Generate a question about a low-quality document and provide "
            "reasoning that accounts for OCR noise, ambiguous characters, and layout "
            "artifacts before giving a best-effort answer."
        ),
        "doc_types": ["scanned_degraded", "government_form", "invoice"],
        "is_dpo": False,
        "multi_doc": False,
    },
    "excel_csv": {
        "system": (
            "You are a data analyst specialising in spreadsheet and tabular documents. "
            "Generate a question requiring numerical analysis, aggregation, or trend "
            "identification from tabular data. Show calculation steps in reasoning."
        ),
        "doc_types": ["spreadsheet", "financial_statement"],
        "is_dpo": False,
        "multi_doc": False,
    },
    "deep_reasoning": {
        "system": (
            "You are an expert analyst performing complex multi-step reasoning over "
            "documents. Generate a question that requires chaining multiple facts, "
            "making inferences, and drawing non-obvious conclusions. Show extended "
            "chain-of-thought reasoning."
        ),
        "doc_types": [
            "financial_statement", "audit_report", "technical_spec", "contract",
        ],
        "is_dpo": False,
        "multi_doc": False,
    },
    "cross_document": {
        "system": (
            "You are an expert at comparing and synthesising information across "
            "multiple documents. Generate a question that requires reconciling or "
            "comparing content from two different documents. Reason about each "
            "document separately before synthesising an answer."
        ),
        "doc_types": ["contract", "invoice", "policy", "resume"],
        "is_dpo": False,
        "multi_doc": True,
    },
}

# ---------------------------------------------------------------------------
# Formatting wrappers
# ---------------------------------------------------------------------------

_DIFFICULTIES = ("easy", "medium", "hard")


def format_sft(
    query: str,
    reasoning: str,
    answer: str,
    category: str,
    difficulty: str = "medium",
) -> Dict[str, Any]:
    """Wrap ``format_sft_example`` and add distillation metadata."""
    base = format_sft_example(query, reasoning, answer)
    base["category"] = category
    base["difficulty"] = difficulty
    base["source"] = "claude_distillation"
    return base


def format_dpo(
    query: str,
    chosen_reasoning: str,
    chosen_answer: str,
    rejected_reasoning: str,
    rejected_answer: str,
    category: str,
) -> Dict[str, Any]:
    """Wrap ``format_dpo_example`` and add distillation metadata."""
    base = format_dpo_example(
        query,
        chosen_reasoning,
        chosen_answer,
        rejected_reasoning,
        rejected_answer,
    )
    base["category"] = category
    base["source"] = "claude_distillation"
    return base


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

_CLAUDE_MODEL = "claude-opus-4-5"


def _call_claude(
    system: str,
    context: str,
    difficulty: str = "medium",
    is_dpo: bool = False,
) -> Dict[str, Any]:
    """Call the Anthropic API to generate a training example.

    Returns a dict with keys:
    - ``question``, ``reasoning``, ``answer`` (SFT)
    - additionally ``rejected_reasoning``, ``rejected_answer`` (DPO)

    Falls back to :func:`_synthetic_fallback` when no API key is configured
    or the API call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _synthetic_fallback(context, difficulty, is_dpo)

    try:
        import anthropic  # imported lazily so offline tests never hit this

        dpo_instruction = (
            "\n\nAlso provide a REJECTED answer that confidently makes up details "
            "not found in the document (for DPO preference training). Format:\n"
            "REJECTED_REASONING: <bad reasoning>\nREJECTED_ANSWER: <hallucinated answer>"
            if is_dpo
            else ""
        )

        difficulty_map = {
            "easy": "straightforward",
            "medium": "moderately complex",
            "hard": "complex and multi-step",
        }
        diff_desc = difficulty_map.get(difficulty, "moderately complex")

        user_message = (
            f"Here is a document to analyse:\n\n{context}\n\n"
            f"Generate a {diff_desc} question about this document, "
            f"detailed reasoning, and a thorough answer.\n\n"
            f"Format your response exactly as:\n"
            f"QUESTION: <question text>\n"
            f"REASONING: <step-by-step reasoning>\n"
            f"ANSWER: <final answer>{dpo_instruction}"
        )

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": user_message}],
            system=system,
        )
        raw = message.content[0].text
        return _parse_claude_response(raw, is_dpo)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Claude API call failed (%s); using synthetic fallback.", exc)
        return _synthetic_fallback(context, difficulty, is_dpo)


def _parse_claude_response(raw: str, is_dpo: bool) -> Dict[str, Any]:
    """Parse Claude's structured response into a dict."""

    def _extract(label: str) -> str:
        marker = f"{label}:"
        start = raw.find(marker)
        if start == -1:
            return ""
        start += len(marker)
        # find next label or end
        next_label_pos = len(raw)
        for other in ("QUESTION:", "REASONING:", "ANSWER:", "REJECTED_REASONING:", "REJECTED_ANSWER:"):
            if other == marker.rstrip():
                continue
            pos = raw.find(other, start)
            if pos != -1 and pos < next_label_pos:
                next_label_pos = pos
        return raw[start:next_label_pos].strip()

    result = {
        "question": _extract("QUESTION"),
        "reasoning": _extract("REASONING"),
        "answer": _extract("ANSWER"),
    }
    if is_dpo:
        result["rejected_reasoning"] = _extract("REJECTED_REASONING")
        result["rejected_answer"] = _extract("REJECTED_ANSWER")
    return result


def _synthetic_fallback(
    context: str,
    difficulty: str = "medium",
    is_dpo: bool = False,
) -> Dict[str, Any]:
    """Return placeholder example when Claude is not available.

    Embeds the full document context in the question so every unique
    document produces a unique training example (critical for dedup).
    """
    lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    first_line = lines[0] if lines else "the document"
    # Use enough context to make each example unique
    snippet = "\n".join(lines[:20]) if len(lines) > 5 else context

    result: Dict[str, Any] = {
        "question": (
            f"Analyze the following document and extract all key information.\n\n"
            f"{context}"
        ),
        "reasoning": (
            f"Step 1: I will read this document carefully. It begins with: {first_line}\n"
            f"Step 2: Identify all entities, dates, amounts, and relationships.\n"
            f"Step 3: Provide a {difficulty}-depth analysis covering all details."
        ),
        "answer": (
            f"Based on analysis of the document, here are the key details:\n\n"
            f"{snippet}\n\n"
            f"The document contains the information shown above. All values are extracted directly from the source."
        ),
    }
    if is_dpo:
        result["rejected_reasoning"] = "I'll quickly scan without reading carefully."
        result["rejected_answer"] = "0 items found. The document does not contain extractable information."
    return result


# ---------------------------------------------------------------------------
# Batch generators
# ---------------------------------------------------------------------------


def generate_sft_batch(
    category: str,
    count: int = 10,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate *count* SFT examples for *category* using Claude (or synthetic fallback).

    Parameters
    ----------
    category:
        One of the keys in :data:`DISTILL_CATEGORIES`.
    count:
        Number of examples to generate.
    seed:
        Random seed for deterministic document generation.
    """
    if category not in DISTILL_CATEGORIES:
        raise ValueError(f"Unknown category: {category!r}. Choose from {list(DISTILL_CATEGORIES)}")

    cfg = DISTILL_CATEGORIES[category]
    rng = random.Random(seed)
    examples: List[Dict[str, Any]] = []

    for i in range(count):
        doc_type = rng.choice(cfg["doc_types"])
        doc_seed = rng.randint(0, 2**31) if seed is None else seed + i
        doc = generate_document(doc_type, seed=doc_seed)
        context = doc.get("content", "")
        difficulty = rng.choice(_DIFFICULTIES)

        raw = _call_claude(
            system=cfg["system"],
            context=context,
            difficulty=difficulty,
            is_dpo=False,
        )
        example = format_sft(
            query=raw.get("question", ""),
            reasoning=raw.get("reasoning", ""),
            answer=raw.get("answer", ""),
            category=category,
            difficulty=difficulty,
        )
        examples.append(example)

    return examples


def generate_dpo_batch(
    category: str,
    count: int = 10,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate *count* DPO preference pairs for *category*.

    Uses the category's ``is_dpo`` system prompt when available; otherwise
    uses the standard system prompt and requests DPO output from Claude.
    """
    if category not in DISTILL_CATEGORIES:
        raise ValueError(f"Unknown category: {category!r}. Choose from {list(DISTILL_CATEGORIES)}")

    cfg = DISTILL_CATEGORIES[category]
    rng = random.Random(seed)
    examples: List[Dict[str, Any]] = []

    for i in range(count):
        doc_type = rng.choice(cfg["doc_types"])
        doc_seed = rng.randint(0, 2**31) if seed is None else seed + i
        doc = generate_document(doc_type, seed=doc_seed)
        context = doc.get("content", "")
        difficulty = rng.choice(_DIFFICULTIES)

        raw = _call_claude(
            system=cfg["system"],
            context=context,
            difficulty=difficulty,
            is_dpo=True,
        )
        example = format_dpo(
            query=raw.get("question", ""),
            chosen_reasoning=raw.get("reasoning", ""),
            chosen_answer=raw.get("answer", ""),
            rejected_reasoning=raw.get("rejected_reasoning", ""),
            rejected_answer=raw.get("rejected_answer", ""),
            category=category,
        )
        examples.append(example)

    return examples


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_examples(
    examples: List[Dict[str, Any]],
    path: Path | str,
) -> int:
    """Append *examples* to a JSONL file with deduplication.

    Deduplication is based on the ``text`` field (SFT) or ``prompt`` field (DPO).
    Returns the number of new examples written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing keys for deduplication
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get("text") or rec.get("prompt") or ""
                if key:
                    seen.add(key)
            except json.JSONDecodeError:
                continue

    written = 0
    with path.open("a", encoding="utf-8") as fh:
        for ex in examples:
            key = ex.get("text") or ex.get("prompt") or ""
            if key in seen:
                continue
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            seen.add(key)
            written += 1

    return written
