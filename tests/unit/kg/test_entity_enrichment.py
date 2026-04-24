"""Verify `_canonical_to_graph_payload` calls entity extraction when enabled.

Flag-gated: with `Config.KG.ENTITY_EXTRACTION_ENABLED=True`, the adapter
enriches payload.entities from a DocWain call. With flag off, behaves as today.
"""
from unittest.mock import MagicMock


def _canonical_extraction_with_text() -> dict:
    return {
        "doc_id": "doc-e1",
        "format": "pdf_native",
        "path_taken": "native",
        "pages": [
            {
                "page_num": 1,
                "blocks": [
                    {"text": "Acme Corp invoice for $1,000 dated 2025-01-15.", "block_type": "paragraph"},
                ],
                "tables": [],
                "images": [],
            }
        ],
        "sheets": [],
        "slides": [],
        "metadata": {"doc_intel": {"doc_type_hint": "invoice"}, "coverage": {}, "extraction_version": "v1"},
    }


def test_entity_enrichment_calls_docwain_and_populates_entities(monkeypatch):
    """When flag ON, extractor is called and result populates payload.entities."""
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.KG, "ENTITY_EXTRACTION_ENABLED", True, raising=False)

    # Stub the gateway.generate_with_metadata used by the extractor helper.
    from src.docwain.prompts import entity_extraction as ee
    import src.tasks.kg as kg_mod

    def fake_extract(document_text: str) -> ee.ExtractedEntities:
        return ee.ExtractedEntities(
            entities=[
                {"text": "Acme Corp", "type": "ORGANIZATION", "confidence": 0.9},
            ],
            relationships=[],
        )

    monkeypatch.setattr(kg_mod, "_call_entity_extractor", fake_extract, raising=False)

    payload = kg_mod._canonical_to_graph_payload(
        extraction=_canonical_extraction_with_text(),
        screening=None,
        subscription_id="sub-x",
        profile_id="prof-x",
        document_id="doc-e1",
    )
    assert len(payload.entities) == 1
    assert payload.entities[0]["text"] == "Acme Corp"


def test_entity_enrichment_skipped_when_flag_off(monkeypatch):
    """When flag OFF, entities stay empty (Plan 3 baseline behavior)."""
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.KG, "ENTITY_EXTRACTION_ENABLED", False, raising=False)

    import src.tasks.kg as kg_mod

    called = {"n": 0}

    def fake_extract(document_text: str):
        called["n"] += 1
        from src.docwain.prompts.entity_extraction import ExtractedEntities
        return ExtractedEntities(entities=[{"text": "SHOULD_NOT_APPEAR", "type": "OTHER"}], relationships=[])

    monkeypatch.setattr(kg_mod, "_call_entity_extractor", fake_extract, raising=False)

    payload = kg_mod._canonical_to_graph_payload(
        extraction=_canonical_extraction_with_text(),
        screening=None,
        subscription_id="sub-x",
        profile_id="prof-x",
        document_id="doc-e1",
    )
    assert payload.entities == []
    assert called["n"] == 0


def test_entity_enrichment_swallows_extractor_failures(monkeypatch):
    """When the extractor raises, payload is still valid with empty entities."""
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.KG, "ENTITY_EXTRACTION_ENABLED", True, raising=False)

    import src.tasks.kg as kg_mod

    def boom(document_text: str):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(kg_mod, "_call_entity_extractor", boom, raising=False)

    payload = kg_mod._canonical_to_graph_payload(
        extraction=_canonical_extraction_with_text(),
        screening=None,
        subscription_id="sub-x",
        profile_id="prof-x",
        document_id="doc-e1",
    )
    # Empty entities, no exception raised — KG continues with Document + mentions only.
    assert payload.entities == []
