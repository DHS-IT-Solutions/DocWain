"""Intelligent query handler — retrieval + reranking + Reasoner generation.

Uses DocWain's production intelligence layers:
- UnifiedRetriever for hybrid search (dense + keyword fallback)
- Cross-encoder reranking for precision
- Reasoner with expert system prompt for grounded, structured responses
- Conversational intent detection for natural interaction
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from teams_app.proxy.query_proxy import QueryResult

logger = logging.getLogger(__name__)


class TeamsQueryHandler:
    """Handles queries using DocWain's production intelligence stack.

    Completely self-contained — does NOT proxy to the main app.
    """

    def __init__(self, qdrant_client: Any = None):
        self._qdrant_client = qdrant_client
        self._reasoner = None
        self._cross_encoder = None
        self._cross_encoder_checked = False

    def _get_reasoner(self):
        """Lazy-initialize the Reasoner with an LLM gateway."""
        if self._reasoner is None:
            from src.generation.reasoner import Reasoner
            from src.llm.gateway import create_llm_gateway
            gateway = create_llm_gateway()
            self._reasoner = Reasoner(llm_gateway=gateway)
            logger.info("TeamsQueryHandler: Reasoner + LLM gateway initialized")
        return self._reasoner

    def _get_cross_encoder(self):
        """Lazy-load cross-encoder for reranking (returns None if unavailable)."""
        if not self._cross_encoder_checked:
            self._cross_encoder_checked = True
            try:
                from src.api.dw_newron import get_cross_encoder
                self._cross_encoder = get_cross_encoder()
                if self._cross_encoder:
                    logger.info("TeamsQueryHandler: Cross-encoder loaded for reranking")
            except Exception as exc:
                logger.debug("Cross-encoder not available: %s", exc)
        return self._cross_encoder

    def _get_qdrant_client(self):
        if self._qdrant_client is not None:
            return self._qdrant_client
        from src.api.dw_newron import get_qdrant_client
        self._qdrant_client = get_qdrant_client()
        return self._qdrant_client

    async def answer(
        self,
        query: str,
        collection_name: str,
        user_id: str = "",
    ) -> QueryResult:
        """Full intelligence pipeline: intent → retrieve → rerank → reason.

        Args:
            query: The user's question.
            collection_name: Teams-prefixed Qdrant collection to search.
            user_id: Optional user identifier.

        Returns:
            QueryResult with grounded response, sources, and metadata.
        """
        try:
            # 1. Check for conversational intent (greetings, help, etc.)
            conv_response = await self._check_conversational(query)
            if conv_response:
                return conv_response

            # 2. Retrieve relevant chunks (hybrid: dense + keyword)
            evidence = await self._retrieve(query, collection_name)

            if not evidence:
                return QueryResult(
                    response=(
                        "I couldn't find relevant information in your documents for that question. "
                        "Try rephrasing, or upload more documents."
                    ),
                    sources=[],
                    grounded=False,
                    context_found=False,
                )

            # 3. Rerank with cross-encoder for precision
            evidence = await self._rerank(query, evidence)

            # 4. Detect task type and output format
            task_type, output_format = self._classify_task(query)

            # 5. Generate response via Reasoner (production prompts + grounding)
            result = await self._reason(query, evidence, task_type, output_format)
            return result

        except Exception as exc:
            logger.error("TeamsQueryHandler.answer failed: %s", exc, exc_info=True)
            return QueryResult(
                response="I'm having trouble processing your question right now. Please try again.",
                error=str(exc),
            )

    async def _check_conversational(self, query: str) -> Optional[QueryResult]:
        """Detect conversational intent (greetings, thanks, etc.)."""
        try:
            from src.intelligence.conversational_nlp import classify_conversational_intent
            result = classify_conversational_intent(query, turn_count=0)
            if result:
                intent, confidence = result
                if confidence > 0.7:
                    response = self._conversational_response(intent)
                    if response:
                        return QueryResult(
                            response=response,
                            sources=[],
                            grounded=False,
                            context_found=False,
                            metadata={"intent": intent, "confidence": confidence},
                        )
        except Exception:
            pass  # Fall through to document query
        return None

    def _conversational_response(self, intent: str) -> Optional[str]:
        """Generate a response for conversational intents."""
        responses = {
            "GREETING": "Hello! I'm DocWain — your document intelligence assistant. Upload a document or ask me a question about your uploaded files.",
            "GREETING_RETURN": "Welcome back! What would you like to know about your documents?",
            "FAREWELL": "Goodbye! Your documents are always here when you need them.",
            "THANKS": "You're welcome! Let me know if you have more questions.",
            "PRAISE": "Thank you! Happy to help. What else would you like to know?",
            "IDENTITY": "I'm DocWain, a document intelligence assistant. I can analyze, summarize, extract, and compare information from your uploaded documents.",
            "CAPABILITY": "I can **summarize** documents, **extract** specific data (names, dates, amounts), **compare** across documents, **analyze** trends, and answer detailed questions — all grounded in your uploaded files.\n\nType **clear all** to remove old documents and start fresh.",
            "HOW_IT_WORKS": "Upload a document and I'll process it automatically. Then just ask questions in plain language — I search your documents, find relevant sections, and give you grounded answers with source references.\n\nType **clear all** to remove old documents and start fresh.",
            "USAGE_HELP": "Just type your question naturally! For example:\n- \"Summarize this document\"\n- \"What are the key dates?\"\n- \"Extract all amounts\"\n- \"Compare the two reports\"\n- **\"clear all\"** — remove old documents and start fresh",
        }
        return responses.get(intent)

    async def _retrieve(
        self, query: str, collection_name: str, top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval: dense vector search + keyword fallback."""
        # Encode query
        from src.embedding.model_loader import encode_with_fallback

        query_vector = await asyncio.to_thread(
            encode_with_fallback,
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query_vec = query_vector[0].tolist()

        client = self._get_qdrant_client()

        # Check collection exists
        try:
            collections = await asyncio.to_thread(
                lambda: client.get_collections().collections
            )
            if collection_name not in {c.name for c in collections}:
                logger.warning("Collection %s not found", collection_name)
                return []
        except Exception as exc:
            logger.warning("Failed to list collections: %s", exc)

        # Dense search
        try:
            results = await asyncio.to_thread(
                lambda: client.query_points(
                    collection_name=collection_name,
                    query=query_vec,
                    using="content_vector",
                    limit=top_k,
                    with_payload=True,
                )
            )
        except Exception as exc:
            logger.error("Dense search failed on %s: %s", collection_name, exc)
            return []

        evidence = []
        for hit in results.points:
            payload = hit.payload or {}
            # Handle both old (train_on_document) and new (teams embedder) payload schemas
            text = (
                payload.get("embedding_text")
                or payload.get("content")
                or payload.get("canonical_text")
                or payload.get("text_clean")
                or payload.get("text")
                or ""
            )
            if not text.strip():
                continue
            source = (
                payload.get("source_file")
                or payload.get("source_name")
                or payload.get("filename")
                or "Document"
            )
            evidence.append({
                "text": text,
                "score": hit.score,
                "source_name": source,
                "section_title": payload.get("section_title", ""),
                "section_path": payload.get("section_path", payload.get("section_kind", "")),
                "page": str(payload.get("page_number", payload.get("page", payload.get("page_start", "")))),
                "document_id": payload.get("document_id", ""),
                "doc_type": payload.get("doc_type", payload.get("doc_domain", "")),
                "chunk_index": payload.get("chunk_index", 0),
                "source_index": len(evidence),
            })

        # Keyword fallback if dense search found too few high-quality results
        high_quality = [e for e in evidence if e["score"] >= 0.5]
        if len(high_quality) < 3 and evidence:
            evidence = await self._keyword_boost(query, evidence)

        return evidence

    async def _keyword_boost(
        self, query: str, evidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Boost chunks that have strong keyword overlap with the query."""
        query_tokens = set(query.lower().split())
        for chunk in evidence:
            chunk_tokens = set(chunk["text"].lower().split())
            overlap = query_tokens & chunk_tokens
            if query_tokens:
                precision = len(overlap) / len(query_tokens)
                recall = len(overlap) / max(len(chunk_tokens), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-9)
                # Blend keyword F1 with dense score
                chunk["score"] = 0.7 * chunk["score"] + 0.3 * f1

        evidence.sort(key=lambda x: x["score"], reverse=True)
        return evidence

    async def _rerank(
        self, query: str, evidence: List[Dict[str, Any]], top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """Rerank with cross-encoder if available, otherwise use score-based ranking."""
        ce = self._get_cross_encoder()
        if ce is None:
            # No cross-encoder — just take top-k by score
            return evidence[:top_k]

        try:
            # Pre-filter to top 10 candidates before expensive CE
            candidates = evidence[:10]
            pairs = [(query, c["text"]) for c in candidates]

            ce_scores = await asyncio.to_thread(
                lambda: ce.predict(pairs).tolist()
            )

            for chunk, ce_score in zip(candidates, ce_scores):
                # Ensemble: 50% dense + 20% keyword + 30% cross-encoder
                chunk["score"] = 0.5 * chunk["score"] + 0.3 * ce_score

            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[:top_k]

        except Exception as exc:
            logger.debug("Cross-encoder reranking failed: %s", exc)
            return evidence[:top_k]

    def _classify_task(self, query: str) -> tuple:
        """Lightweight task type and output format classification."""
        q = query.lower().strip()

        # Task type detection
        if any(w in q for w in ("summarize", "summary", "overview", "brief", "outline")):
            task_type = "summarize"
        elif any(w in q for w in ("compare", "difference", "versus", "vs", "contrast")):
            task_type = "compare"
        elif any(w in q for w in ("extract", "find", "list all", "get all", "what are the")):
            task_type = "extract"
        elif any(w in q for w in ("how many", "total", "count", "average", "sum")):
            task_type = "aggregate"
        elif any(w in q for w in ("step", "procedure", "how to", "process", "guide")):
            task_type = "investigate"
        else:
            task_type = "lookup"

        # Output format detection
        if any(w in q for w in ("table", "tabular", "columns")):
            output_format = "table"
        elif any(w in q for w in ("list", "bullet", "items")):
            output_format = "bullets"
        elif any(w in q for w in ("step", "procedure", "numbered")):
            output_format = "numbered"
        elif task_type in ("compare",):
            output_format = "table"
        elif task_type in ("extract", "aggregate"):
            output_format = "bullets"
        else:
            output_format = "prose"

        return task_type, output_format

    async def _reason(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        task_type: str,
        output_format: str,
    ) -> QueryResult:
        """Generate a grounded response using the production Reasoner."""
        reasoner = self._get_reasoner()

        # Format evidence for the Reasoner
        formatted_evidence = []
        for e in evidence:
            formatted_evidence.append({
                "source_name": e.get("source_name", "Document"),
                "section": e.get("section_title", ""),
                "page": str(e.get("page", "")),
                "text": e["text"],
                "score": e.get("score", 0),
                "source_index": e.get("source_index", 0),
            })

        # Call Reasoner (runs LLM with production prompts)
        result = await asyncio.to_thread(
            reasoner.reason,
            query=query,
            task_type=task_type,
            output_format=output_format,
            evidence=formatted_evidence,
            doc_context=None,
            conversation_history=None,
            use_thinking=False,
        )

        # Build sources list
        seen = set()
        sources = []
        for e in evidence:
            name = e.get("source_name", "Document")
            if name not in seen:
                seen.add(name)
                sources.append({"title": name})

        return QueryResult(
            response=result.text,
            sources=sources,
            grounded=result.grounded,
            context_found=True,
            metadata={
                "task_type": task_type,
                "output_format": output_format,
                "chunks_used": len(evidence),
                "thinking": result.thinking if hasattr(result, "thinking") else None,
            },
        )
