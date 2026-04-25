import json
import pytest

from src.intelligence.knowledge.provider import (
    KnowledgeProvider,
    JsonKnowledgeProvider,
    KbNotFound,
)


def test_lookup_term(tmp_path):
    kb = {"version": "1.0", "entries": {"flood": "Flood damage exclusion clause"}}
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(kb))
    p: KnowledgeProvider = JsonKnowledgeProvider.load_from_path(str(path), kb_id="kb1")
    assert p.kb_id == "kb1"
    assert p.lookup("flood") == "Flood damage exclusion clause"


def test_lookup_missing_returns_none(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"version": "1.0", "entries": {}}))
    p = JsonKnowledgeProvider.load_from_path(str(path), kb_id="empty")
    assert p.lookup("nope") is None


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(KbNotFound):
        JsonKnowledgeProvider.load_from_path(str(tmp_path / "no.json"), kb_id="x")


def test_bundled_kbs_present_and_loadable():
    from pathlib import Path

    bundled = Path("src/intelligence/knowledge/bundled")
    expected = [
        "insurance_taxonomy_v1.json",
        "icd10_subset_v1.json",
        "hr_policies_v1.json",
        "procurement_terms_v1.json",
    ]
    for name in expected:
        path = bundled / name
        assert path.exists(), f"bundled KB missing: {name}"
        kb = JsonKnowledgeProvider.load_from_path(str(path), kb_id=name)
        assert kb.entries, f"bundled KB empty: {name}"
        assert kb.version
