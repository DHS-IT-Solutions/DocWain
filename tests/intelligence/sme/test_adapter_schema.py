"""Tests for the adapter YAML Pydantic schema."""
import pytest
from pydantic import ValidationError

from src.intelligence.sme.adapter_schema import Adapter


def _valid():
    return {
        "domain": "finance",
        "version": "1.0.0",
        "persona": {"role": "analyst", "voice": "direct", "grounding_rules": []},
        "dossier": {
            "section_weights": {"a": 0.4, "b": 0.3, "c": 0.3},
            "prompt_template": "prompts/finance_dossier.md",
        },
        "insight_detectors": [
            {"type": "trend", "rule": "qoq_gt", "params": {"t": 0.05}}
        ],
        "comparison_axes": [{"name": "period", "dimension": "temporal"}],
        "kg_inference_rules": [
            {
                "pattern": "(a)-[:X]->(b)",
                "produces": "r",
                "confidence_floor": 0.6,
                "max_hops": 3,
            }
        ],
        "recommendation_frames": [
            {
                "frame": "f",
                "template": "t",
                "requires": {"insight_types": ["trend"]},
            }
        ],
        "response_persona_prompts": {
            "diagnose": "p/d.md",
            "analyze": "p/a.md",
            "recommend": "p/r.md",
        },
        "retrieval_caps": {
            "max_pack_tokens": {
                "analyze": 6000,
                "diagnose": 5000,
                "recommend": 4500,
                "investigate": 8000,
            }
        },
        "output_caps": {
            "analyze": 1200,
            "diagnose": 1500,
            "recommend": 1000,
            "investigate": 2000,
        },
    }


def test_valid_adapter():
    a = Adapter(**_valid())
    assert a.domain == "finance"
    assert a.version == "1.0.0"


def test_rejects_unknown_field():
    d = _valid()
    d["bogus"] = 1
    with pytest.raises(ValidationError):
        Adapter(**d)


def test_section_weights_must_sum_to_one():
    d = _valid()
    d["dossier"]["section_weights"] = {"a": 0.9, "b": 0.5}
    with pytest.raises(ValidationError, match="sum to 1.0"):
        Adapter(**d)


def test_max_hops_bounded():
    d = _valid()
    d["kg_inference_rules"][0]["max_hops"] = 10
    with pytest.raises(ValidationError):
        Adapter(**d)


def test_version_semver():
    d = _valid()
    d["version"] = "not-a-semver"
    with pytest.raises(ValidationError, match="semver"):
        Adapter(**d)


def test_generic_minimal_adapter():
    d = _valid()
    d["domain"] = "generic"
    for k in (
        "insight_detectors",
        "comparison_axes",
        "kg_inference_rules",
        "recommendation_frames",
    ):
        d[k] = []
    assert Adapter(**d).domain == "generic"


def test_content_hash_defaults_none():
    a = Adapter(**_valid())
    assert a.content_hash is None
    assert a.source_path is None


def test_runtime_injected_fields_settable():
    d = _valid()
    d["content_hash"] = "abc123"
    d["source_path"] = "sme_adapters/global/finance.yaml"
    a = Adapter(**d)
    assert a.content_hash == "abc123"
    assert a.source_path == "sme_adapters/global/finance.yaml"
