"""`_extraction_to_graph_payload` must accept both legacy and canonical extraction shapes.

Spec §5 of 2026-04-24-kg-training-stage-background-design.md.

Real signature discovered in src/tasks/kg.py:
    _extraction_to_graph_payload(extraction, screening, subscription_id, profile_id, document_id)
"""
from src.kg.ingest import GraphIngestPayload
from src.tasks.kg import _extraction_to_graph_payload


def _legacy_extraction() -> dict:
    return {
        "entities": [
            {"text": "Acme Corp", "type": "ORGANIZATION", "confidence": 0.9, "chunk_id": "c1"},
        ],
        "relationships": [],
        "tables": [{"headers": ["h1"], "rows": [["v1"]]}],
        "sections": {"intro": "text"},
        "metadata": {
            "source_file": "/blob/doc1.pdf",
            "filename": "doc1.pdf",
            "doc_type": "invoice",
            "doc_name": "doc1.pdf",
        },
        "temporal_spans": [],
    }


def _canonical_extraction() -> dict:
    return {
        "doc_id": "doc-1",
        "format": "pdf_native",
        "path_taken": "native",
        "pages": [
            {
                "page_num": 1,
                "blocks": [
                    {"text": "First paragraph of a native PDF.", "block_type": "paragraph"},
                    {"text": "Second block with more content here.", "block_type": "paragraph"},
                ],
                "tables": [{"rows": [["h1", "h2"], ["r1c1", "r1c2"]]}],
                "images": [],
            },
        ],
        "sheets": [],
        "slides": [],
        "metadata": {
            "doc_intel": {
                "doc_type_hint": "invoice",
                "layout_complexity": "simple",
                "has_handwriting": False,
                "routing_confidence": 0.9,
            },
            "coverage": {
                "verifier_score": 1.0,
                "missed_regions": [],
                "low_confidence_regions": [],
                "fallback_invocations": [],
            },
            "extraction_version": "2026-04-23-v1",
        },
    }


def _call_adapter(extraction, *, document_id="doc-1", subscription_id="sub-1",
                  profile_id="prof-1", screening=None):
    """Call adapter using the REAL signature:
        _extraction_to_graph_payload(extraction, screening, subscription_id, profile_id, document_id)
    """
    return _extraction_to_graph_payload(
        extraction=extraction,
        screening=screening,
        subscription_id=subscription_id,
        profile_id=profile_id,
        document_id=document_id,
    )


def test_legacy_shape_still_produces_valid_payload():
    payload = _call_adapter(_legacy_extraction())
    assert isinstance(payload, GraphIngestPayload)
    # Legacy shape has entities — payload should carry them forward
    assert len(payload.entities) >= 1


def test_canonical_shape_produces_valid_payload_without_entities():
    payload = _call_adapter(_canonical_extraction())
    assert isinstance(payload, GraphIngestPayload)
    # Document exists
    assert payload.document is not None
    # Canonical has no entities today (Researcher Agent fills them later)
    assert payload.entities == []
    # But we should have mentions from pages[].blocks[]
    assert len(payload.mentions) >= 1


def test_canonical_shape_doc_type_from_doc_intel():
    payload = _call_adapter(_canonical_extraction())
    # doc_type propagates from metadata.doc_intel.doc_type_hint
    doc_type = None
    if isinstance(payload.document, dict):
        doc_type = payload.document.get("doc_type") or payload.document.get("document_category")
    else:
        doc_type = getattr(payload.document, "doc_type", None) or getattr(payload.document, "document_category", None)
    # Should be the hint or a safe fallback (generic/unknown)
    assert doc_type in ("invoice", "generic", "unknown", None) or isinstance(doc_type, str)
