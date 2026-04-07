"""Standalone query handler — searches Qdrant directly and generates responses via LLM.

Does NOT proxy to the main app. Self-contained retrieval + generation pipeline
for the Teams service.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from teams_app.proxy.query_proxy import QueryResult

logger = logging.getLogger(__name__)

_RAG_SYSTEM_PROMPT = (
    "You are DocWain, a document intelligence assistant.\n"
    "Answer the user's question based ONLY on the provided document context.\n"
    "If the context doesn't contain the answer, say so clearly.\n"
    "Be direct and concise. Bold key values with **markdown**."
)


class TeamsQueryHandler:
    """Handles queries by searching Qdrant and generating responses via LLM.

    Completely self-contained — does NOT proxy to the main app.
    """

    def __init__(self, qdrant_client: Any = None):
        self._qdrant_client = qdrant_client
        self._llm_gateway = None

    def _get_llm_gateway(self):
        """Lazy-initialize and return the LLM gateway (created once, reused)."""
        if self._llm_gateway is None:
            from src.llm.gateway import create_llm_gateway
            self._llm_gateway = create_llm_gateway()
            logger.info("TeamsQueryHandler: LLM gateway initialized")
        return self._llm_gateway

    def _get_qdrant_client(self):
        """Return the Qdrant client, falling back to a fresh connection."""
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
        """Search Qdrant for relevant chunks, build context, generate LLM response.

        Args:
            query: The user's question.
            collection_name: Teams-prefixed Qdrant collection to search.
            user_id: Optional user identifier for logging.

        Returns:
            QueryResult with the generated response, sources, and grounding flag.
        """
        try:
            # 1. Encode query to vector
            query_vector = await self._encode_query(query)

            # 2. Search Qdrant for top-k relevant chunks
            chunks = await self._search_qdrant(query_vector, collection_name, top_k=5)

            if not chunks:
                return QueryResult(
                    response=(
                        "I couldn't find any relevant information in your documents "
                        "for that question. Try rephrasing, or upload more documents."
                    ),
                    sources=[],
                    grounded=False,
                    context_found=False,
                )

            # 3. Build RAG prompt with retrieved context
            context_text, sources = self._build_context(chunks)
            prompt = self._build_prompt(query, context_text)

            # 4. Generate response via LLM
            response_text = await self._generate_response(prompt)

            return QueryResult(
                response=response_text,
                sources=sources,
                grounded=True,
                context_found=True,
                metadata={"chunks_used": len(chunks), "collection": collection_name},
            )

        except Exception as exc:
            logger.error("TeamsQueryHandler.answer failed: %s", exc, exc_info=True)
            return QueryResult(
                response="I'm having trouble processing your question right now. Please try again.",
                error=str(exc),
            )

    async def _encode_query(self, query: str) -> List[float]:
        """Encode query text to a dense vector using the embedding model."""
        from src.embedding.model_loader import encode_with_fallback

        vectors = await asyncio.to_thread(
            encode_with_fallback,
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors[0].tolist()

    async def _search_qdrant(
        self,
        query_vector: List[float],
        collection_name: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the Qdrant collection for relevant chunks."""
        client = self._get_qdrant_client()

        # Check if collection exists
        try:
            collections = await asyncio.to_thread(
                lambda: client.get_collections().collections
            )
            existing = {col.name for col in collections}
            if collection_name not in existing:
                logger.warning("Collection %s does not exist in Qdrant", collection_name)
                return []
        except Exception as exc:
            logger.warning("Failed to list Qdrant collections: %s", exc)
            # Proceed anyway — the search call will fail if collection is missing

        try:
            from qdrant_client.models import NamedVector

            results = await asyncio.to_thread(
                lambda: client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using="content_vector",
                    limit=top_k,
                    with_payload=True,
                )
            )
        except Exception as exc:
            logger.error("Qdrant search failed on %s: %s", collection_name, exc)
            return []

        chunks = []
        for hit in results.points:
            payload = hit.payload or {}
            chunks.append({
                "text": payload.get("content", payload.get("text", payload.get("embedding_text", ""))),
                "score": hit.score,
                "source_name": payload.get("source_file", payload.get("filename", "Unknown")),
                "section": payload.get("section_title", ""),
                "page": payload.get("page_number", ""),
                "document_id": payload.get("document_id", ""),
            })
        return chunks

    def _build_context(
        self, chunks: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Build context text and sources list from retrieved chunks.

        Returns:
            (context_text, sources) where sources is a list of dicts with "title" key.
        """
        context_parts = []
        sources = []
        seen_sources = set()

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            source_name = chunk.get("source_name", "Unknown")
            section = chunk.get("section", "")
            page = chunk.get("page", "")

            header = f"[Document {i}: {source_name}]"
            if section:
                header += f" Section: {section}"
            if page:
                header += f" Page: {page}"

            context_parts.append(f"{header}\n{text}")

            if source_name not in seen_sources:
                seen_sources.add(source_name)
                sources.append({"title": source_name})

        context_text = "\n\n---\n\n".join(context_parts)
        return context_text, sources

    def _build_prompt(self, query: str, context_text: str) -> str:
        """Build the RAG prompt from query and context."""
        return (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}"
        )

    async def _generate_response(self, prompt: str) -> str:
        """Generate a response using the LLM gateway."""
        gateway = self._get_llm_gateway()
        response = await asyncio.to_thread(
            gateway.generate,
            prompt,
            system=_RAG_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=2048,
        )
        return response
