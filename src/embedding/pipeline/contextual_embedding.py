"""Contextual Retrieval — prepend LLM-generated context to each chunk.

For each chunk, a short retrieval context is generated against the live
vLLM DocWain instance. The context is forced to mention identifiers a
search query is likely to use (invoice/PO/quote numbers, dates, parties,
amounts). The result is stored on the payload as ``chunk_context`` and
prepended to ``embedding_text`` so the vector carries those identifiers
regardless of whether they appear in the chunk body.

Gating: controlled by ``Config.ContextualRetrieval.ENABLED``. Failure is
always graceful — if vLLM is unreachable or a single generation fails,
the original ``embedding_text`` is kept, the chunk is left unchanged, and
ingestion continues.

Two public entry points, same prompt under the hood:
    * ``generate_context_for_chunk`` — single-shot, used by the one-shot
      backfill script when re-contextualising existing documents.
    * ``enrich_payloads_with_context`` — bulk, used by ``ingest_payloads``
      to contextualise freshly-transformed payloads before embedding.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = (
    "You produce one short retrieval context sentence per chunk. "
    "A retrieval context must include every identifier a search query is "
    "likely to use: document number/code/title, counterparty or vendor names, "
    "key dates, amounts, reference numbers. Do NOT describe the chunk "
    "(avoid phrases like 'this chunk contains', 'section with', 'header of'). "
    "Instead, state the facts directly as if you were indexing them. "
    "One sentence, 20-60 words. No preamble, no quotes, no bullets."
)

PROMPT_TEMPLATE = (
    "<document name=\"{doc_name}\" type=\"{doc_type}\">\n"
    "{doc_text}\n"
    "</document>\n\n"
    "<chunk index=\"{chunk_index}\">\n"
    "{chunk_text}\n"
    "</chunk>\n\n"
    "Using the document name and any identifiers visible in either the "
    "document or the chunk (invoice/PO/quote numbers, dates, parties, "
    "amounts), write the retrieval context. Include the document's "
    "identifier from its filename so this chunk is findable by that code. "
    "Do not start with 'This chunk' or similar."
)

_DOC_TEXT_CAP = 12000
_CHUNK_CAP = 2000
_MAX_TOKENS = 160
_TEMPERATURE = 0.2


def _truncate(text: str, cap: int) -> str:
    if not text or len(text) <= cap:
        return text or ""
    head = text[:cap]
    for sep in ("\n\n", ". ", "\n", " "):
        idx = head.rfind(sep)
        if idx > cap // 2:
            return head[:idx] + "…"
    return head + "…"


def _clean(text: str) -> str:
    ctx = (text or "").strip().replace("\n", " ")
    if len(ctx) >= 2 and ctx[0] in "\"'" and ctx[-1] == ctx[0]:
        ctx = ctx[1:-1].strip()
    ctx = " ".join(ctx.split())
    return ctx[:600]


def generate_context_for_chunk(
    vllm,
    *,
    doc_name: str,
    doc_type: str,
    doc_text: str,
    chunk_text: str,
    chunk_index: int = 0,
) -> str:
    """Return a single retrieval context sentence; empty string on failure."""
    prompt = PROMPT_TEMPLATE.format(
        doc_name=doc_name or "document",
        doc_type=doc_type or "document",
        chunk_index=chunk_index,
        doc_text=_truncate(doc_text, _DOC_TEXT_CAP),
        chunk_text=_truncate(chunk_text, _CHUNK_CAP),
    )
    try:
        out = vllm.query(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=_MAX_TOKENS,
            temperature=_TEMPERATURE,
            require_vllm=True,  # impact #7 — no Ollama fallback on embed path
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("contextual-retrieval: context generation failed — %s", exc)
        return ""
    return _clean(out)


def _is_enabled() -> bool:
    try:
        from src.api.config import Config
        cr = getattr(Config, "ContextualRetrieval", None)
        if cr is None:
            return False
        return bool(getattr(cr, "ENABLED", False))
    except Exception:
        return False


def _vllm_client():
    try:
        from src.serving.vllm_manager import VLLMManager
        vllm = VLLMManager()
        if vllm.health_check():
            return vllm
        logger.info("contextual-retrieval: vLLM unreachable, skipping context generation")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("contextual-retrieval: vLLM manager init failed — %s", exc)
        return None


def _chunk_sort_key(payload: Dict[str, Any]) -> int:
    idx = payload.get("chunk_index")
    if idx is None:
        nav = payload.get("navigation") or {}
        idx = nav.get("chunk_index")
    try:
        return int(idx or 0)
    except (TypeError, ValueError):
        return 0


def _reconstruct_doc_text(payloads: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for p in payloads:
        t = p.get("canonical_text") or p.get("content") or p.get("embedding_text") or ""
        if t:
            parts.append(str(t))
    return "\n\n".join(parts)


def enrich_payloads_with_context(
    payloads: List[Dict[str, Any]],
    *,
    force: bool = False,
) -> int:
    """Add contextualized embedding_text to each payload in place.

    Returns the number of payloads that were successfully enriched.
    Controlled by ``Config.ContextualRetrieval.ENABLED`` unless ``force``
    is set (used by the backfill CLI). No-op if vLLM is unavailable.
    """
    if not payloads:
        return 0
    if not force and not _is_enabled():
        logger.debug("contextual-retrieval: disabled by config")
        return 0

    vllm = _vllm_client()
    if vllm is None:
        return 0

    # Skip the chunk-level resolution filter — this helper runs over any
    # payload shape as long as it carries canonical_text / content / chunk_index
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for p in payloads:
        if (p.get("resolution") or "chunk") != "chunk":
            continue
        doc_id = str(p.get("document_id") or "")
        if not doc_id:
            continue
        by_doc.setdefault(doc_id, []).append(p)

    enriched = 0
    for doc_id, docs in by_doc.items():
        docs.sort(key=_chunk_sort_key)
        doc_name = next(
            (str(p.get("source_name") or (p.get("metadata") or {}).get("source_file") or "")
             for p in docs if p.get("source_name") or (p.get("metadata") or {}).get("source_file")),
            "document",
        )
        doc_type = next(
            (str(p.get("doc_domain") or p.get("doc_type") or "") for p in docs
             if p.get("doc_domain") or p.get("doc_type")),
            "document",
        )
        doc_text = _reconstruct_doc_text(docs)
        if not doc_text:
            continue

        for p in docs:
            chunk_text = p.get("canonical_text") or p.get("content") or ""
            if not chunk_text:
                continue
            chunk_index = _chunk_sort_key(p)
            ctx = generate_context_for_chunk(
                vllm,
                doc_name=doc_name,
                doc_type=doc_type,
                doc_text=doc_text,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
            )
            if not ctx:
                continue
            base_embedding = p.get("embedding_text") or chunk_text
            p["chunk_context"] = ctx
            p["embedding_text"] = f"{ctx}\n\n{base_embedding}"
            p["contextualized"] = True
            enriched += 1

    if enriched:
        logger.info(
            "contextual-retrieval: enriched %d/%d chunk(s) across %d doc(s)",
            enriched, sum(len(v) for v in by_doc.values()), len(by_doc),
        )
    return enriched


__all__ = [
    "generate_context_for_chunk",
    "enrich_payloads_with_context",
]
