import pytest

from src.intelligence.adapters.schema import (
    Adapter,
    AppliesWhen,
    InsightTypeConfig,
    KnowledgeConfig,
    SanctionedKb,
    Watchlist,
    ActionTemplate,
    VisualizationSpec,
    parse_adapter_yaml,
)


SAMPLE_YAML = """
name: insurance
version: "1.0"
description: "Insurance policies, claims, coverage analysis"
applies_when:
  domain_classifier_labels: [insurance, policy]
  doc_type_hints: [policy_document]
  keyword_evidence_min: 3
  keywords: [policyholder, deductible, premium]
researcher:
  insight_types:
    anomaly:
      prompt_template: "prompts/insurance_anomaly.md"
      enabled: true
    gap:
      prompt_template: "prompts/insurance_gap.md"
      enabled: true
knowledge:
  sanctioned_kbs:
    - kb_id: insurance_taxonomy_v1
      ref: "blob://kbs/insurance_taxonomy_v1.json"
      describes: "Common policy types"
  citation_rule: "doc_grounded_first"
watchlists:
  - id: renewal_due
    description: "Policy renewal due soon"
    eval: "expr:doc.policy_end_date - now < 60d"
    fires_insight_type: next_action
actions:
  - action_id: generate_coverage_summary
    title: "Generate coverage summary PDF"
    action_type: artifact
    artifact_template: "templates/insurance_coverage_summary.md"
    requires_confirmation: false
visualizations:
  - viz_id: coverage_comparison_table
    insight_types: [comparison]
"""


def test_parse_yaml_returns_adapter():
    a = parse_adapter_yaml(SAMPLE_YAML)
    assert isinstance(a, Adapter)
    assert a.name == "insurance"
    assert a.version == "1.0"
    assert isinstance(a.applies_when, AppliesWhen)
    assert a.applies_when.keyword_evidence_min == 3
    assert "anomaly" in a.researcher.insight_types
    assert isinstance(a.researcher.insight_types["anomaly"], InsightTypeConfig)
    assert a.researcher.insight_types["anomaly"].enabled is True


def test_knowledge_section():
    a = parse_adapter_yaml(SAMPLE_YAML)
    assert isinstance(a.knowledge, KnowledgeConfig)
    assert len(a.knowledge.sanctioned_kbs) == 1
    assert isinstance(a.knowledge.sanctioned_kbs[0], SanctionedKb)
    assert a.knowledge.sanctioned_kbs[0].kb_id == "insurance_taxonomy_v1"


def test_watchlists_actions_visualizations():
    a = parse_adapter_yaml(SAMPLE_YAML)
    assert len(a.watchlists) == 1
    assert isinstance(a.watchlists[0], Watchlist)
    assert a.watchlists[0].id == "renewal_due"
    assert len(a.actions) == 1
    assert isinstance(a.actions[0], ActionTemplate)
    assert a.actions[0].requires_confirmation is False
    assert len(a.visualizations) == 1
    assert isinstance(a.visualizations[0], VisualizationSpec)


def test_minimal_adapter_only_name():
    minimal = (
        "name: tiny\nversion: '1.0'\ndescription: t\napplies_when: {}\n"
        "researcher:\n  insight_types: {}\n"
        "knowledge:\n  sanctioned_kbs: []\n  citation_rule: doc_grounded_first\n"
        "watchlists: []\n"
        "actions: []\n"
        "visualizations: []\n"
    )
    a = parse_adapter_yaml(minimal)
    assert a.name == "tiny"
    assert a.researcher.insight_types == {}


def test_invalid_yaml_raises():
    with pytest.raises(ValueError):
        parse_adapter_yaml("not: a: valid: yaml:::")
