import json

from src.extraction.vision.extractor import (
    EXTRACTOR_SYSTEM_PROMPT,
    VisionExtraction,
    parse_extractor_response,
)


def test_parse_extractor_response_well_formed():
    text = json.dumps({
        "regions": [
            {"type": "text_block", "bbox": [0.1, 0.1, 0.5, 0.2], "content": "hello", "confidence": 0.95},
            {"type": "table", "bbox": [0.1, 0.4, 0.8, 0.3],
             "content": {"rows": [["a","b"], ["1","2"]]}, "confidence": 0.88},
        ],
        "reading_order": [0, 1],
        "page_confidence": 0.9,
    })
    out = parse_extractor_response(text)
    assert isinstance(out, VisionExtraction)
    assert len(out.regions) == 2
    assert out.regions[0]["type"] == "text_block"
    assert out.page_confidence == 0.9


def test_parse_extractor_response_returns_empty_on_garbage():
    out = parse_extractor_response("[not json]")
    assert out.regions == []
    assert out.page_confidence == 0.0


def test_prompt_non_empty():
    assert isinstance(EXTRACTOR_SYSTEM_PROMPT, str) and len(EXTRACTOR_SYSTEM_PROMPT) > 80
