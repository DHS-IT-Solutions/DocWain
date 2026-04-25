from src.intelligence.researcher_v2.profile_passes import (
    run_profile_pass,
    ProfilePassInputs,
)
from src.intelligence.adapters.schema import (
    Adapter, AppliesWhen, ResearcherSection, InsightTypeConfig, KnowledgeConfig,
)


def _adapter() -> Adapter:
    return Adapter(
        name="generic",
        version="1.0",
        description="t",
        applies_when=AppliesWhen(),
        researcher=ResearcherSection(insight_types={
            "comparison": InsightTypeConfig(prompt_template="", enabled=True, requires_min_docs=2),
            "conflict": InsightTypeConfig(prompt_template="", enabled=True, requires_min_docs=2),
        }),
        knowledge=KnowledgeConfig(),
    )


def _llm_returning(payload):
    def call(*, system, user, **_):
        return payload
    return call


def test_skips_when_below_min_docs():
    docs = [{"document_id": "D1", "text": "alpha"}]
    result = run_profile_pass(ProfilePassInputs(
        adapter=_adapter(),
        insight_type="comparison",
        documents=docs,
        profile_id="p", subscription_id="s",
        kb_provider=None,
        llm_call=_llm_returning('{"insights":[]}'),
    ))
    assert result.insights == []
    assert result.skipped_reason == "below_min_docs"


def test_emits_with_2_docs():
    docs = [
        {"document_id": "D1", "text": "Premium $1800"},
        {"document_id": "D2", "text": "Premium $2400"},
    ]
    payload = (
        '{"insights":[{"headline":"D2 premium higher",'
        '"body":"Premium $2400 vs Premium $1800 represents higher cost",'
        '"evidence_doc_spans":[{"document_id":"D1","page":1,"char_start":0,"char_end":13,"quote":"Premium $1800"},'
        '{"document_id":"D2","page":1,"char_start":0,"char_end":13,"quote":"Premium $2400"}],'
        '"confidence":0.9,"severity":"notice"}]}'
    )
    result = run_profile_pass(ProfilePassInputs(
        adapter=_adapter(),
        insight_type="comparison",
        documents=docs,
        profile_id="p", subscription_id="s",
        kb_provider=None,
        llm_call=_llm_returning(payload),
    ))
    assert len(result.insights) == 1
    insight = result.insights[0]
    assert set(insight.document_ids) == {"D1", "D2"}
    assert insight.insight_type == "comparison"
