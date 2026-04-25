import json

from src.intelligence.knowledge.provider import JsonKnowledgeProvider
from src.intelligence.knowledge.template_resolver import resolve_template


def _kb(tmp_path, entries):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"version": "1.0", "entries": entries}))
    return JsonKnowledgeProvider.load_from_path(str(path), kb_id="kb1")


def test_simple_lookup_directive(tmp_path):
    kb = _kb(tmp_path, {"flood": "Flood is typically excluded."})
    text = "Note: {{kb.lookup('flood')}}"
    out = resolve_template(text, kb=kb)
    assert out == "Note: Flood is typically excluded."


def test_unknown_term_replaced_with_blank(tmp_path):
    kb = _kb(tmp_path, {})
    text = "Note: {{kb.lookup('nope')}}"
    out = resolve_template(text, kb=kb)
    assert out == "Note: "


def test_no_directive_returns_text_unchanged(tmp_path):
    kb = _kb(tmp_path, {})
    text = "no directives here"
    out = resolve_template(text, kb=kb)
    assert out == text


def test_kb_none_means_directives_become_blank():
    text = "Note: {{kb.lookup('flood')}}"
    out = resolve_template(text, kb=None)
    assert out == "Note: "
