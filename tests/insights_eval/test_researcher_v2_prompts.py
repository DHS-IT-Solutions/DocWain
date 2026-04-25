from src.docwain.prompts.researcher_v2_generic import (
    build_typed_insight_prompt,
    SYSTEM_PROMPT,
)


def test_system_prompt_includes_separation_rule():
    assert "external" in SYSTEM_PROMPT.lower()
    assert "do not" in SYSTEM_PROMPT.lower() or "must not" in SYSTEM_PROMPT.lower()


def test_build_prompt_for_each_type():
    for itype in ("anomaly", "gap", "comparison", "scenario", "trend",
                  "recommendation", "conflict", "projection", "next_action"):
        prompt = build_typed_insight_prompt(
            insight_type=itype,
            domain_name="generic",
            document_text="Some doc text " * 50,
            kb_context="",
        )
        assert itype in prompt.lower() or itype.replace("_", " ") in prompt.lower()
        assert "JSON" in prompt
        assert "evidence_doc_spans" in prompt


def test_truncates_long_doc_text():
    long_text = "x" * 100_000
    prompt = build_typed_insight_prompt(
        insight_type="anomaly",
        domain_name="generic",
        document_text=long_text,
        kb_context="",
    )
    assert len(prompt) < 30_000
