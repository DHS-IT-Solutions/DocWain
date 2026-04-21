"""Builds enriched Qdrant payloads from extraction + screening + KG data."""

import logging

logger = logging.getLogger(__name__)


# Payload schema version. Bump when the set of top-level flat fields
# emitted here changes so downstream consumers (retriever, consistency
# check, backfill scripts) can tell new points from legacy ones.
EMBED_PIPELINE_VERSION = "v2"


def build_enriched_payload(
    chunk: dict,
    chunk_index: int,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    extraction_data: dict,
    screening_summary: dict,
    kg_node_ids: list = None,
    quality_grade: str = "C",
    source_name: str = "",
    doc_domain: str = "",
) -> dict:
    """Build enriched Qdrant point payload for a single chunk.

    The Qdrant collection defines payload indexes on a set of flat
    top-level fields (``chunk_id``, ``resolution``, ``section_id``,
    ``chunk_kind``, ``section_kind``, ``page``, ``source_name``,
    ``doc_domain``, ``embed_pipeline_version``). Retrieval filters
    (``resolution="chunk"`` in retriever.py, section/chunk filtering
    in filter_builder.py, neighbor lookups in dw_newron.py) and the
    embed-stage consistency check all read those flat fields.

    This writer used to only emit nested structures (``chunk.id``,
    ``section.id``, ``provenance.page_start``), which meant every
    filtered Qdrant query silently returned zero — including the
    consistency check that reports ``qdrant_missing`` on every
    document. Flat fields below duplicate the nested metadata so both
    legacy readers (``payload["chunk"]["id"]``) and index-aware
    readers (``payload["chunk_id"]``) see a consistent view.

    Args:
        chunk: Chunk dict with text, section info, provenance
        chunk_index: Position in document
        document_id: Document identifier
        subscription_id: Org identifier
        profile_id: Department/domain identifier
        extraction_data: Full extraction from Azure Blob
        screening_summary: Screening summary from MongoDB
        kg_node_ids: KG node IDs linked to entities in this chunk
        quality_grade: Chunk quality grade (A-F)
        source_name: Original filename (for ``source_name`` index)
        doc_domain: Document domain tag (for ``doc_domain`` index)

    Returns:
        Qdrant payload dict
    """
    chunk_text = chunk.get("text", "")
    chunk_entities = []
    chunk_entity_types = []
    chunk_importance = 0.0

    entity_scores = screening_summary.get("entity_scores", {})
    for entity in extraction_data.get("entities", []):
        entity_text = entity.get("text", "") if isinstance(entity, dict) else ""
        if entity_text and entity_text.lower() in chunk_text.lower():
            chunk_entities.append(entity_text)
            entity_type = entity.get("type", "UNKNOWN") if isinstance(entity, dict) else "UNKNOWN"
            chunk_entity_types.append(entity_type)
            score = entity_scores.get(entity_text, 0.0)
            chunk_importance = max(chunk_importance, score)

    chunk_id = f"{document_id}_chunk_{chunk_index}"
    chunk_kind = chunk.get("type", "text")
    section = chunk.get("section") or {}
    section_id = section.get("id", "") if isinstance(section, dict) else ""
    section_kind = section.get("kind") or section.get("section_kind") or "text"
    provenance = chunk.get("provenance") or {}
    page_start = provenance.get("page_start", 0) if isinstance(provenance, dict) else 0

    return {
        # Identity
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "document_id": document_id,

        # Flat indexed fields — MUST stay in sync with the Qdrant
        # collection's payload indexes. Adding/removing one here
        # without also updating ensure_collection() in vector_store
        # (and the consistency check / retriever filters) will silently
        # break filtered queries again.
        "chunk_id": chunk_id,
        "resolution": "chunk",
        "chunk_kind": chunk_kind,
        "section_id": section_id,
        "section_kind": section_kind,
        "page": int(page_start or 0),
        "source_name": str(source_name or ""),
        "doc_domain": str(doc_domain or "generic"),
        "embed_pipeline_version": EMBED_PIPELINE_VERSION,

        # Nested chunk metadata (kept for callers that still read
        # payload["chunk"]["id"], e.g. retrieval_planner / KG writers).
        "chunk": {
            "id": chunk_id,
            "index": chunk_index,
            "type": chunk_kind,
            "hash": chunk.get("hash", ""),
            "token_count": chunk.get("token_count", 0),
        },

        # Section context
        "section": chunk.get("section", {
            "id": "", "title": "", "path": [], "level": 0
        }),

        # Provenance
        "provenance": chunk.get("provenance", {
            "page_start": 0, "page_end": 0
        }),

        # Enrichment from screening
        "entities": chunk_entities,
        "entity_types": list(set(chunk_entity_types)),
        "domain_tags": screening_summary.get("domain_tags", []),
        "doc_category": screening_summary.get("doc_category", "unknown"),
        "importance_score": chunk_importance,

        # KG linkage
        "kg_node_ids": kg_node_ids or [],

        # Quality
        "quality_grade": quality_grade,

        # Source text
        "text": chunk_text
    }
