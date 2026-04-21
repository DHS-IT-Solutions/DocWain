"""Unified intelligence handler.

Single entry point for all intent categories. Uses the unified DocWain
vLLM instance and the generation-layer prompts (src/generation/prompts.py).
Replaces the former FastPathHandler module — no fast/smart split.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.generation.prompts import build_system_prompt
from src.serving.model_router import RouterResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_NO_RETRIEVAL_INTENTS = frozenset({"greeting", "identity", "greet", "meta", "help", "capability", "goodbye"})

_GREETING_SYSTEM = "You are DocWain."
_IDENTITY_SYSTEM = "You are DocWain."


class IntelligenceHandler:
    """Handles any intent against the unified DocWain model.

    For no-retrieval intents (greetings, identity) generates a direct
    response with no evidence lookup. For evidence-backed intents
    (lookup, list, count, extract) it performs a single Qdrant search
    then generates a grounded response.

    Complex intents (analyze, investigate, compare, summarize, etc.)
    are handled upstream by the full query pipeline; this handler
    covers only the intents that do not need a Plan -> Execute loop.
    """

    def __init__(self, vllm_manager: Any) -> None:
        self._mgr = vllm_manager

    # -- Public -----------------------------------------------------------

    def handle(
        self,
        query: str,
        router_result: RouterResult,
        profile_context: Optional[str] = None,
        retriever: Optional[Any] = None,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        intent = getattr(router_result, "intent", "") or ""
        if intent in _NO_RETRIEVAL_INTENTS:
            return self._handle_no_retrieval(query, intent)
        return self._handle_with_retrieval(
            query=query,
            router_result=router_result,
            profile_context=profile_context,
            retriever=retriever,
            subscription_id=subscription_id,
            profile_id=profile_id,
        )

    # -- Private handlers -------------------------------------------------

    def _handle_no_retrieval(self, query: str, intent: str) -> Dict[str, Any]:
        system = _GREETING_SYSTEM if intent in {"greeting", "greet"} else _IDENTITY_SYSTEM
        try:
            response = self._mgr.query(
                prompt=query,
                system_prompt=system,
                max_tokens=512,
                temperature=0.5,
            )
        except Exception as exc:
            logger.error("No-retrieval generation failed: %s", exc)
            response = ""
        if not response:
            response = self._static_fallback(intent)
        return _build_payload(response=response, sources=[], grounded=False, context_found=False)

    def _handle_with_retrieval(
        self,
        query: str,
        router_result: RouterResult,
        profile_context: Optional[str],
        retriever: Optional[Any],
        subscription_id: Optional[str],
        profile_id: Optional[str],
    ) -> Dict[str, Any]:
        evidence_chunks: List[Any] = []
        if retriever is not None and subscription_id and profile_id:
            try:
                retrieval_result = retriever.retrieve(
                    query=query,
                    subscription_id=subscription_id,
                    profile_ids=[profile_id],
                    top_k=15,
                )
                evidence_chunks = list(retrieval_result.chunks)
            except Exception as exc:
                logger.warning("Retrieval failed: %s", exc)

        evidence_text, sources = self._format_evidence(evidence_chunks)
        system = build_system_prompt(profile_domain=profile_context or "")
        user_prompt = self._build_generation_prompt(query, getattr(router_result, "intent", ""), evidence_text)

        try:
            response = self._mgr.query(
                prompt=user_prompt,
                system_prompt=system,
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            response = ""

        grounded = bool(sources) and bool(response)
        context_found = bool(sources)
        if not response and context_found:
            response = "I found relevant documents but was unable to generate a response. Please try again."
        elif not response:
            response = "I couldn't find relevant information in the documents to answer that question."

        chart_spec = self._extract_chart_spec(response)
        return _build_payload(
            response=response,
            sources=sources,
            grounded=grounded,
            context_found=context_found,
            chart_spec=chart_spec,
        )

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _format_evidence(chunks: List[Any]) -> tuple[str, List[Dict[str, Any]]]:
        """Format retrieval chunks into an evidence text block and a deduped
        source list.

        `chunks` are expected to be `src.retrieval.retriever.EvidenceChunk`
        instances (dataclass), but this method tolerates any object exposing
        the attributes listed on that dataclass. Deduplication key is the
        chunk_id itself (unique per chunk), so distinct chunks from the same
        file and page remain distinct source entries.
        """
        if not chunks:
            return "", []
        parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        seen: set = set()
        for i, chunk in enumerate(chunks):
            text = getattr(chunk, "text", "") or ""
            source_name = getattr(chunk, "source_name", "unknown")
            page_start = getattr(chunk, "page_start", None)
            chunk_id = getattr(chunk, "chunk_id", None) or getattr(chunk, "document_id", None) or f"chunk-{i}"
            snippet = text[:200] if text else ""
            parts.append(f"[SOURCE-{i + 1}] (file: {source_name}, page: {page_start})\n{text}\n")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            sources.append({
                "file_name": source_name,
                "page": page_start,
                "snippet": snippet,
            })
        return "\n".join(parts), sources

    @staticmethod
    def _build_generation_prompt(query: str, intent: str, evidence_text: str) -> str:
        if not evidence_text:
            return (
                f"The user asked: {query}\n\n"
                "No relevant evidence was found in the documents. "
                "Politely inform the user that the documents do not contain "
                "information to answer this question."
            )
        instruction = {
            "lookup": "Answer the question directly using the evidence below.",
            "list": "Provide a clear list based on the evidence below.",
            "count": "Count the relevant items from the evidence below and state the total.",
            "extract": "Extract the requested data from the evidence below in a structured format.",
        }.get(intent, "Answer the question using the evidence below.")
        return (
            f"{instruction}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            f"USER QUESTION: {query}"
        )

    @staticmethod
    def _extract_chart_spec(response: str) -> Optional[Dict[str, Any]]:
        import re
        match = re.search(r"<!--DOCWAIN_VIZ\s*\n(.*?)\n\s*-->", response, re.DOTALL)
        if not match:
            return None
        try:
            import json
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _static_fallback(intent: str) -> str:
        if intent in {"greeting", "greet"}:
            return "Hello! I'm DocWain, your document intelligence assistant. How can I help you today?"
        if intent == "identity":
            return (
                "I'm DocWain, an intelligent document analysis assistant. "
                "I can help you search, summarise, compare, and extract data "
                "from your uploaded documents."
            )
        return "I'm having trouble processing your request right now. Please try again shortly."


def _build_payload(
    response: str,
    sources: List[Dict[str, Any]],
    grounded: bool,
    context_found: bool,
    chart_spec: Optional[Dict[str, Any]] = None,
    alerts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "response": response,
        "sources": sources,
        "chart_spec": chart_spec,
        "alerts": alerts or [],
        "grounded": grounded,
        "context_found": context_found,
    }
