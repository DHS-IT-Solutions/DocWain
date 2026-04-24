import json

from src.docwain.prompts.researcher import (
    RESEARCHER_SYSTEM_PROMPT,
    ResearcherInsights,
    parse_researcher_response,
)


def test_parse_well_formed():
    text = json.dumps({
        "summary": "Invoice from Acme for $1,000.",
        "key_facts": ["Acme Corp is the vendor", "Amount is $1,000"],
        "entities": [{"text": "Acme Corp", "type": "ORGANIZATION"}],
        "recommendations": ["Verify vendor credentials"],
        "anomalies": [],
        "questions_to_ask": ["Is this a recurring vendor?"],
        "confidence": 0.85,
    })
    out = parse_researcher_response(text)
    assert isinstance(out, ResearcherInsights)
    assert out.summary.startswith("Invoice from Acme")
    assert len(out.key_facts) == 2
    assert out.confidence == 0.85


def test_parse_garbage_returns_empty():
    out = parse_researcher_response("nonsense")
    assert out.summary == ""
    assert out.key_facts == []
    assert out.confidence == 0.0


def test_prompt_non_empty():
    assert len(RESEARCHER_SYSTEM_PROMPT) > 100
    assert "insight" in RESEARCHER_SYSTEM_PROMPT.lower() or "summary" in RESEARCHER_SYSTEM_PROMPT.lower()
