"""Reasoner — the REASON step of the DocWain Core Agent pipeline.

Calls the LLM with evidence and returns a grounded answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.generation.prompts import build_reason_prompt, build_system_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReasonerResult:
    """Outcome of a single REASON step."""

    text: str
    sources: List[Dict[str, Any]]
    grounded: bool
    thinking: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token budget lookup
# ---------------------------------------------------------------------------

_BASE_TOKENS: Dict[str, int] = {
    "lookup": 1536,
    "extract": 3072,
    "list": 3072,
    "summarize": 2048,
    "overview": 2048,
    "compare": 3072,
    "investigate": 3072,
    "aggregate": 2048,
}

# ---------------------------------------------------------------------------
# Number extraction regex — shared by grounding check
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"""
    \$[\d,]+(?:\.\d+)?    |   # dollar amounts
    \d{1,3}(?:,\d{3})+       |   # comma-separated integers
    \d+\.\d+                  |   # decimals
    \d+%                      |   # percentages
    \b\d{2,}\b                    # bare integers ≥ 2 digits
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Reasoner class
# ---------------------------------------------------------------------------


class Reasoner:
    """Generates an LLM answer from ranked evidence."""

    def __init__(self, llm_gateway: Any) -> None:
        self._llm = llm_gateway

    # -- public API ---------------------------------------------------------

    def reason(
        self,
        query: str,
        task_type: str,
        output_format: str,
        evidence: List[Dict[str, Any]],
        doc_context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]],
        use_thinking: bool = False,
        profile_domain: str = "",
        kg_context: str = "",
        profile_expertise: Optional[Dict] = None,
    ) -> ReasonerResult:
        """Run the REASON step: prompt the LLM and return a grounded result."""

        system_msg = build_system_prompt(
            profile_domain=profile_domain,
            kg_context=kg_context,
            profile_expertise=profile_expertise,
        )
        user_msg = build_reason_prompt(
            query=query,
            task_type=task_type,
            output_format=output_format,
            evidence=evidence,
            doc_context=doc_context,
            conversation_history=conversation_history,
        )

        max_tokens = self._compute_token_budget(
            task_type, len(evidence), use_thinking,
        )

        # Adaptive temperature: factual tasks need consistency, creative tasks need diversity
        _TASK_TEMPERATURE = {
            "lookup": 0.05, "extract": 0.05, "list": 0.05,
            "aggregate": 0.05, "compare": 0.1, "investigate": 0.1,
            "summarize": 0.15, "overview": 0.15,
        }
        temperature = _TASK_TEMPERATURE.get(task_type, 0.1)

        logger.info(
            "[REASONER_PROMPT] task=%s evidence_count=%d prompt_len=%d temp=%.2f query=%r",
            task_type, len(evidence), len(user_msg), temperature, query[:80],
        )

        try:
            text, metadata = self._llm.generate_with_metadata(
                user_msg,
                system=system_msg,
                think=use_thinking,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            logger.exception("LLM generation failed during REASON step")
            return ReasonerResult(
                text="Unable to generate an answer due to a processing error.",
                sources=self._extract_sources(evidence),
                grounded=False,
            )

        thinking_text = metadata.get("thinking") if metadata else None
        usage = metadata.get("usage", {}) if metadata else {}

        sources = self._extract_sources(evidence)
        grounded = self._check_grounding(text, evidence, doc_context)

        return ReasonerResult(
            text=text,
            sources=sources,
            grounded=grounded,
            thinking=thinking_text,
            usage=usage,
        )

    def reason_stream(
        self,
        query: str,
        task_type: str,
        output_format: str,
        evidence: List[Dict[str, Any]],
        doc_context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]],
        use_thinking: bool = False,
        profile_domain: str = "",
        kg_context: str = "",
    ) -> Generator[str, None, None]:
        """Stream tokens from the REASON step.

        Yields raw token strings as they arrive from the LLM.
        The caller is responsible for post-processing (cleaning, grounding).
        """
        system_msg = build_system_prompt(
            profile_domain=profile_domain,
            kg_context=kg_context,
        )
        user_msg = build_reason_prompt(
            query=query,
            task_type=task_type,
            output_format=output_format,
            evidence=evidence,
            doc_context=doc_context,
            conversation_history=conversation_history,
        )

        # Streaming bypasses thinking mode — don't inflate the token budget
        max_tokens = self._compute_token_budget(
            task_type, len(evidence), thinking=False,
        )

        # Adaptive temperature for streaming too
        _TASK_TEMPERATURE = {
            "lookup": 0.05, "extract": 0.05, "list": 0.05,
            "aggregate": 0.05, "compare": 0.1, "investigate": 0.1,
            "summarize": 0.15, "overview": 0.15,
        }
        temperature = _TASK_TEMPERATURE.get(task_type, 0.1)

        logger.info(
            "[REASONER_STREAM] task=%s evidence_count=%d prompt_len=%d max_tokens=%d temp=%.2f query=%r",
            task_type, len(evidence), len(user_msg), max_tokens, temperature, query[:80],
        )

        try:
            yield from self._llm.generate_stream(
                user_msg,
                system=system_msg,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            logger.exception("LLM streaming failed during REASON step")
            yield "Unable to generate an answer due to a processing error."

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _extract_sources(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a source list from evidence dicts."""
        sources: List[Dict[str, Any]] = []
        for item in evidence:
            sources.append({
                "source_name": item.get("source_name", "unknown"),
                "page": item.get("page"),
                "section": item.get("section"),
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "source_index": item.get("source_index"),
                "score": item.get("score"),
                "excerpt": (item.get("text", "")[:200]
                            if item.get("text") else None),
            })
        return sources

    def _check_grounding(
        self, answer: str, evidence: List[Dict[str, Any]],
        doc_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Evidence-anchored grounding check.

        Verifies that the answer's key claims are derivable from the provided
        evidence AND document intelligence context. Uses semantic overlap
        rather than strict verbatim matching, allowing expert-level synthesis
        and reasoning while catching fabrication.

        Returns False only when the answer introduces substantial content
        that cannot be traced to any evidence.
        """
        if not evidence:
            return False

        evidence_parts = []
        for item in evidence:
            text = (
                item.get("text")
                or item.get("canonical_text")
                or item.get("embedding_text")
                or item.get("content")
                or ""
            )
            evidence_parts.append(text)

        if doc_context:
            for s in doc_context.get("summaries") or []:
                if s:
                    evidence_parts.append(str(s))
            for f in doc_context.get("key_facts") or []:
                if f:
                    evidence_parts.append(str(f))
            for kv in doc_context.get("key_values") or []:
                if isinstance(kv, dict):
                    evidence_parts.append(" ".join(str(v) for v in kv.values()))
                elif kv:
                    evidence_parts.append(str(kv))
            for e in doc_context.get("entities") or []:
                if isinstance(e, dict):
                    evidence_parts.append(" ".join(str(v) for v in e.values()))
                elif e:
                    evidence_parts.append(str(e))

        evidence_text = " ".join(evidence_parts)

        if not evidence_text.strip():
            logger.warning("[Reasoner] Grounding: evidence items exist but contain no text — UNGROUNDED")
            return False

        # Short answers bypass the gates (< 40 chars — was 20). Catches concise
        # factual answers like "**philip.simon.derock@company.com**" that are
        # correctly grounded but too short to trigger meaningful word-overlap.
        if len(answer.strip()) < 40:
            return True

        # Number gate (unchanged — Task 1 confirmed it does not misfire).
        # Checks that numeric claims in the answer are traceable to evidence.
        answer_numbers = set(_NUMBER_RE.findall(answer))
        if answer_numbers:
            evidence_numbers = set(_NUMBER_RE.findall(evidence_text))
            ungrounded_nums = answer_numbers - evidence_numbers
            if ungrounded_nums:
                logger.warning(
                    "[Reasoner] Grounding: %d/%d numbers not found in evidence: %s",
                    len(ungrounded_nums), len(answer_numbers),
                    list(ungrounded_nums)[:10],
                )
            # Allow up to 20% ungrounded numbers.
            if len(ungrounded_nums) / len(answer_numbers) > 0.20:
                logger.debug(
                    "[Reasoner] Grounding: %d/%d numbers ungrounded",
                    len(ungrounded_nums), len(answer_numbers),
                )
                return False

        # Word gate (relaxed per Task 1 diagnostic):
        #   - Short answers (<10 meaningful words) trivially pass when evidence exists.
        #   - Concise "not-found" style answers (<20 meaningful words, containing
        #     explicit negation language like "not found / not specified / not
        #     present / not provided") also pass — these legitimately share zero
        #     words with on-topic evidence while being correct.
        #   - Ratio threshold lowered from 15% to 5% — paraphrased professional
        #     answers (abbreviation expansion, domain synonyms) routinely fall
        #     in the 10–15% band while being correctly grounded.
        #   - Absolute `overlap < 5` floor REMOVED. It was firing on legitimate
        #     short answers ("Not found in the documents" = 4 meaningful words
        #     = 100% ratio but tripped the floor).
        answer_words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', answer)
        )
        evidence_words = set(
            w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', evidence_text)
        )

        if len(answer_words) < 10:
            return True  # concise answer + evidence present → trust retrieval

        # Concise "not-found" answers: legitimate negation responses whose
        # vocabulary does not overlap with on-topic evidence chunks.
        if len(answer_words) < 20 and re.search(
            r'\bnot\s+(?:found|specified|present|provided|available|mentioned|listed|stated|given|included)\b',
            answer,
            re.IGNORECASE,
        ):
            return True

        if answer_words and evidence_words:
            overlap = len(answer_words & evidence_words)
            overlap_ratio = overlap / len(answer_words)
            if overlap_ratio < 0.05:
                logger.warning(
                    "[Reasoner] Grounding: only %d/%d words (%.0f%%) overlap with evidence — UNGROUNDED",
                    overlap, len(answer_words), overlap_ratio * 100,
                )
                return False

        return True

    @staticmethod
    def _compute_token_budget(
        task_type: str, evidence_count: int, thinking: bool
    ) -> int:
        """Compute a max-tokens budget based on task and context."""
        base = _BASE_TOKENS.get(task_type, 512)

        # Scale up for richer evidence sets
        if evidence_count > 10:
            base = int(base * 1.3)
        elif evidence_count > 5:
            base = int(base * 1.15)

        if thinking:
            # Qwen3's <think> blocks can consume 2000+ tokens of reasoning
            # before producing the actual answer. Double the budget to ensure
            # enough room for both thinking and answer generation.
            base = int(base * 1.5)

        return min(base, 16384)
