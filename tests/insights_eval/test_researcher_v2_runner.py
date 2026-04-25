from src.intelligence.researcher_v2.runner import (
    run_per_doc_insight_pass,
    DocPassInputs,
    DocPassResult,
)
from src.intelligence.adapters.schema import (
    Adapter, AppliesWhen, ResearcherSection, InsightTypeConfig,
    KnowledgeConfig,
)


def _adapter() -> Adapter:
    return Adapter(
        name="generic",
        version="1.0",
        description="t",
        applies_when=AppliesWhen(),
        researcher=ResearcherSection(insight_types={
            "anomaly": InsightTypeConfig(prompt_template="prompts/generic_anomaly.md", enabled=True),
        }),
        knowledge=KnowledgeConfig(),
    )


def _llm_returning(text):
    def call(*, system, user, **_):
        return text
    return call


def test_runner_emits_validated_insights():
    llm = _llm_returning(
        '{"insights": [{"headline":"H","body":"H — body grounded in quote excludes flood",'
        '"evidence_doc_spans":[{"document_id":"DOC-1","page":1,"char_start":0,"char_end":18,'
        '"quote":"excludes flood damage"}],"confidence":0.8,"severity":"warn"}]}'
    )
    result = run_per_doc_insight_pass(DocPassInputs(
        adapter=_adapter(),
        insight_type="anomaly",
        document_id="DOC-1",
        document_text="Policy excludes flood damage and earthquake.",
        profile_id="p-1",
        subscription_id="s-1",
        kb_provider=None,
        llm_call=llm,
    ))
    assert isinstance(result, DocPassResult)
    assert len(result.insights) == 1
    insight = result.insights[0]
    assert insight.headline == "H"
    assert insight.insight_type == "anomaly"
    assert insight.adapter_version == "generic@1.0"


def test_runner_skips_disabled_type():
    a = _adapter()
    a.researcher.insight_types["anomaly"].enabled = False
    result = run_per_doc_insight_pass(DocPassInputs(
        adapter=a,
        insight_type="anomaly",
        document_id="DOC-1",
        document_text="x",
        profile_id="p-1",
        subscription_id="s-1",
        kb_provider=None,
        llm_call=_llm_returning('{"insights":[]}'),
    ))
    assert result.insights == []
    assert result.skipped_reason == "type_disabled"


def test_runner_drops_insight_with_zero_evidence():
    llm = _llm_returning(
        '{"insights":[{"headline":"H","body":"x","evidence_doc_spans":[],'
        '"confidence":0.5,"severity":"notice"}]}'
    )
    result = run_per_doc_insight_pass(DocPassInputs(
        adapter=_adapter(),
        insight_type="anomaly",
        document_id="DOC-1",
        document_text="x",
        profile_id="p-1",
        subscription_id="s-1",
        kb_provider=None,
        llm_call=llm,
    ))
    assert result.insights == []
