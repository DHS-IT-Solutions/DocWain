"""Fast path handler for simple queries routed to the 14B model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.generation.prompts import build_system_prompt
from src.serving.model_router import RouterResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Intents that need no retrieval at all.
_NO_RETRIEVAL_INTENTS = frozenset({"greeting", "identity"})

# Greetings / identity prompts.
_GREETING_SYSTEM = "You are DocWain."

_IDENTITY_SYSTEM = "You are DocWain."


class FastPathHandler:
    """Handles queries routed to the fast (14B) model path.

    For greetings and identity questions, generates a direct response with
    no retrieval.  For lookups, lists, and counts, performs a single Qdrant
    search then generates a grounded response.

    Usage::

        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager()
        handler = FastPathHandler(mgr)
        result = handler.handle(
            query="How many invoices are there?",
            router_result=RouterResult(intent="count", complexity="simple", route="fast"),
            profile_context=None,
            retriever=some_retriever,
        )
    """

    def __init__(self, vllm_manager: Any) -> None:
        self._mgr = vllm_manager

    def handle(
        self,
        query: str,
        router_result: RouterResult,
        profile_context: Optional[str] = None,
        retriever: Optional[Any] = None,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle a fast-path query and return an AnswerPayload-compatible dict.

        Args:
            query: The user's question.
            router_result: Classification result from IntentRouter.
            profile_context: Optional domain context string.
            retriever: A DocWainRetriever (or compatible) for evidence lookup.
            subscription_id: Needed for retriever queries.
            profile_id: Needed for retriever queries.
            collection_name: Qdrant collection name.

        Returns:
            Dict with keys: response, sources, chart_spec, alerts, grounded,
            context_found.
        """
        intent = router_result.intent

        # -- No-retrieval intents (greetings, identity) -----------------------
        if intent in _NO_RETRIEVAL_INTENTS:
            return self._handle_no_retrieval(query, intent)

        # -- Retrieval-backed intents (lookup, list, count, extract) ----------
        return self._handle_with_retrieval(
            query=query,
            router_result=router_result,
            profile_context=profile_context,
            retriever=retriever,
            subscription_id=subscription_id,
            profile_id=profile_id,
            collection_name=collection_name,
        )

    # -- Private handlers -----------------------------------------------------

    def _handle_no_retrieval(self, query: str, intent: str) -> Dict[str, Any]:
        """Generate a direct response for greetings / identity — no evidence."""
        system = _GREETING_SYSTEM if intent == "greeting" else _IDENTITY_SYSTEM

        try:
            response = self._mgr.query_fast(
                prompt=query,
                system_prompt=system,
                max_tokens=512,
                temperature=0.5,
            )
        except Exception as exc:
            logger.error("Fast-path no-retrieval generation failed: %s", exc)
            response = self._static_fallback(intent)

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
        collection_name: Optional[str],
    ) -> Dict[str, Any]:
        """Retrieve evidence from Qdrant, then generate a grounded answer."""

        # Retrieve evidence.
        evidence_chunks: List[Any] = []
        if retriever is not None and subscription_id and profile_id and collection_name:
            try:
                evidence_chunks = retriever.retrieve(
                    query=query,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    top_k=15,
                    collection_name=collection_name,
                )
            except Exception as exc:
                logger.warning("Fast-path retrieval failed: %s", exc)

        # Build evidence text block.
        evidence_text, sources = self._format_evidence(evidence_chunks)

        # Build the generation prompt.
        system = build_system_prompt(
            profile_domain=profile_context or "",
        )
        user_prompt = self._build_generation_prompt(query, router_result.intent, evidence_text)

        try:
            response = self._mgr.query_fast(
                prompt=user_prompt,
                system_prompt=system,
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as exc:
            logger.error("Fast-path generation failed: %s", exc)
            response = ""

        grounded = bool(sources) and bool(response)
        context_found = bool(sources)

        if not response and context_found:
            response = "I found relevant documents but was unable to generate a response. Please try again."
        elif not response:
            response = "I couldn't find relevant information in the documents to answer that question."

        # Extract chart spec if the model emitted a visualization directive.
        chart_spec = self._extract_chart_spec(response)

        return _build_payload(
            response=response,
            sources=sources,
            grounded=grounded,
            context_found=context_found,
            chart_spec=chart_spec,
        )

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _format_evidence(chunks: List[Any]) -> tuple[str, List[Dict[str, Any]]]:
        """Format evidence chunks into a text block and a sources list."""
        if not chunks:
            return "", []

        parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        seen: set = set()

        for i, chunk in enumerate(chunks):
            text = getattr(chunk, "text", None) or getattr(chunk, "snippet", "")
            file_name = getattr(chunk, "file_name", "unknown")
            page = getattr(chunk, "page", None)
            snippet = getattr(chunk, "snippet", text[:200] if text else "")
            snippet_sha = getattr(chunk, "snippet_sha", "")

            parts.append(f"[SOURCE-{i + 1}] (file: {file_name}, page: {page})\n{text}\n")

            key = (file_name, page, snippet_sha)
            if key not in seen:
                seen.add(key)
                sources.append({"file_name": file_name, "page": page, "snippet": snippet})

        evidence_text = "\n".join(parts)
        return evidence_text, sources

    @staticmethod
    def _build_generation_prompt(query: str, intent: str, evidence_text: str) -> str:
        """Build the user-facing generation prompt with evidence context."""
        if not evidence_text:
            return (
                f"The user asked: {query}\n\n"
                "No relevant evidence was found in the documents. "
                "Politely inform the user that the documents do not contain "
                "information to answer this question."
            )

        intent_instruction = {
            "lookup": "Answer the question directly using the evidence below.",
            "list": "Provide a clear list based on the evidence below.",
            "count": "Count the relevant items from the evidence below and state the total.",
            "extract": "Extract the requested data from the evidence below in a structured format.",
        }.get(intent, "Answer the question using the evidence below.")

        return (
            f"{intent_instruction}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            f"USER QUESTION: {query}"
        )

    @staticmethod
    def _extract_chart_spec(response: str) -> Optional[Dict[str, Any]]:
        """Extract a DOCWAIN_VIZ directive from the response, if present."""
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
        """Return a static fallback response when the LLM is unavailable."""
        if intent == "greeting":
            return "Hello! I'm DocWain, your document intelligence assistant. How can I help you today?"
        if intent == "identity":
            return (
                "I'm DocWain, an intelligent document analysis assistant. "
                "I can help you search, summarise, compare, and extract data "
                "from your uploaded documents."
            )
        return "I'm having trouble processing your request right now. Please try again shortly."


# -- Payload builder ----------------------------------------------------------

def _build_payload(
    response: str,
    sources: List[Dict[str, Any]],
    grounded: bool,
    context_found: bool,
    chart_spec: Optional[Dict[str, Any]] = None,
    alerts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build an AnswerPayload-compatible dict."""
    return {
        "response": response,
        "sources": sources,
        "chart_spec": chart_spec,
        "alerts": alerts or [],
        "grounded": grounded,
        "context_found": context_found,
    }
