import json

from src.docwain.prompts.entity_extraction import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ExtractedEntities,
    parse_entity_response,
)


def test_parse_entity_response_well_formed():
    text = json.dumps({
        "entities": [
            {"text": "Acme Corp", "type": "ORGANIZATION", "confidence": 0.9},
            {"text": "John Doe", "type": "PERSON", "confidence": 0.85},
        ],
        "relationships": [
            {"source": "John Doe", "target": "Acme Corp", "type": "WORKS_AT"},
        ],
    })
    out = parse_entity_response(text)
    assert isinstance(out, ExtractedEntities)
    assert len(out.entities) == 2
    assert out.entities[0]["text"] == "Acme Corp"
    assert len(out.relationships) == 1
    assert out.relationships[0]["type"] == "WORKS_AT"


def test_parse_entity_response_tolerates_code_fences():
    text = "```json\n" + json.dumps({"entities": [{"text": "X", "type": "T"}], "relationships": []}) + "\n```"
    out = parse_entity_response(text)
    assert len(out.entities) == 1


def test_parse_entity_response_returns_empty_on_garbage():
    out = parse_entity_response("not json at all")
    assert out.entities == []
    assert out.relationships == []


def test_system_prompt_non_empty():
    assert isinstance(ENTITY_EXTRACTION_SYSTEM_PROMPT, str)
    assert len(ENTITY_EXTRACTION_SYSTEM_PROMPT) > 100
    assert "entities" in ENTITY_EXTRACTION_SYSTEM_PROMPT.lower()
