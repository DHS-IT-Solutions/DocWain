"""Phase 2 tests for :class:`KGMultiHopMaterializer`.

Exercises pattern validation (ERRATA §15), rule iteration + candidate
materialization, and the trace path for invalid / skipped rows.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.builders.kg_materializer import (
    KGMultiHopMaterializer,
    _validate_pattern,
)


def _rule(pattern, produces, confidence_floor=0.7, max_hops=3):
    r = MagicMock()
    r.pattern = pattern
    r.produces = produces
    r.confidence_floor = confidence_floor
    r.max_hops = max_hops
    return r


def _adapter(rules=None):
    a = MagicMock()
    a.version = "1.2.0"
    a.kg_inference_rules = rules or [
        _rule(
            "(a)-[:PAYS]->(b)-[:FUNDS]->(c)",
            "indirectly_funds",
            0.7,
            3,
        ),
    ]
    return a


_MISSING = object()


def _row(
    src="n1",
    dst="n3",
    path=_MISSING,
    evidence=_MISSING,
    confidence=0.75,
):
    return {
        "src_node_id": src,
        "dst_node_id": dst,
        "inference_path": (
            [
                {"from": "n1", "edge": "PAYS", "to": "n2"},
                {"from": "n2", "edge": "FUNDS", "to": "n3"},
            ]
            if path is _MISSING
            else path
        ),
        "evidence": (
            [{"doc_id": "d1", "chunk_id": "c1"}]
            if evidence is _MISSING
            else evidence
        ),
        "confidence": confidence,
    }


def test_validate_pattern_accepts_allowlisted_chars():
    _validate_pattern("(a)-[:PAYS]->(b)-[:FUNDS]->(c)")
    _validate_pattern("(a:Account)-[:FUNDS]->()-[:FUNDS]->(b:Account)")


def test_validate_pattern_rejects_injection_attempt():
    with pytest.raises(ValueError, match="disallowed characters"):
        _validate_pattern(
            "(a)-[:PAYS]->(b) RETURN apoc.do.whenNotNull(x, y)"
        )


def test_validate_pattern_rejects_empty():
    with pytest.raises(ValueError):
        _validate_pattern("")
    with pytest.raises(ValueError):
        _validate_pattern(None)  # type: ignore[arg-type]


def test_build_emits_item_per_row():
    kg = MagicMock()
    kg.run_pattern.return_value = [_row()]
    trace = MagicMock()
    builder = KGMultiHopMaterializer(ctx=MagicMock(), kg=kg, trace=trace)

    items = builder.build(
        subscription_id="sub_abc",
        profile_id="prof_fin",
        adapter=_adapter(),
        version=3,
    )
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, ArtifactItem)
    assert item.artifact_type == "kg_edge"
    assert item.subscription_id == "sub_abc"
    assert item.profile_id == "prof_fin"
    assert item.metadata["from_node"] == "n1"
    assert item.metadata["to_node"] == "n3"
    assert item.metadata["relation_type"] == "indirectly_funds"
    assert len(item.inference_path) == 2
    # Hard-filter propagated to KG client
    call = kg.run_pattern.call_args
    assert call.kwargs["subscription_id"] == "sub_abc"
    assert call.kwargs["profile_id"] == "prof_fin"
    assert call.kwargs["max_hops"] == 3


def test_build_raises_on_disallowed_pattern_before_kg_query():
    kg = MagicMock()
    trace = MagicMock()
    bad_rule = _rule(
        "(a)-[:PAYS]->(b) RETURN apoc.do.whenNotNull(x, y)",
        "exfil_attempt",
    )
    builder = KGMultiHopMaterializer(ctx=MagicMock(), kg=kg, trace=trace)
    with pytest.raises(ValueError, match="disallowed characters"):
        builder.build(
            subscription_id="s",
            profile_id="p",
            adapter=_adapter(rules=[bad_rule]),
            version=1,
        )
    # Defence-in-depth: KG client never called when pattern is rejected
    kg.run_pattern.assert_not_called()


def test_build_skips_rows_with_empty_inference_path():
    kg = MagicMock()
    kg.run_pattern.return_value = [
        _row(path=[]),
        _row(src="a", dst="b", path=[{"from": "a", "edge": "X", "to": "b"}]),
    ]
    trace = MagicMock()
    builder = KGMultiHopMaterializer(ctx=MagicMock(), kg=kg, trace=trace)
    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),
        version=1,
    )
    assert len(items) == 1
    events = [c.args[0] for c in trace.append.call_args_list]
    empties = [e for e in events if e.get("stage") == "builder_kg_skip_empty_path"]
    assert len(empties) == 1


def test_build_skips_rows_over_max_hops():
    kg = MagicMock()
    kg.run_pattern.return_value = [
        _row(
            path=[
                {"from": "n1", "edge": "A", "to": "n2"},
                {"from": "n2", "edge": "B", "to": "n3"},
                {"from": "n3", "edge": "C", "to": "n4"},
                {"from": "n4", "edge": "D", "to": "n5"},
            ]
        ),
    ]
    trace = MagicMock()
    builder = KGMultiHopMaterializer(
        ctx=MagicMock(),
        kg=kg,
        trace=trace,
    )
    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(
            rules=[
                _rule(
                    "(a)-[:A]->(b)",
                    "derived",
                    0.7,
                    max_hops=3,
                )
            ]
        ),
        version=1,
    )
    assert items == []
    events = [c.args[0] for c in trace.append.call_args_list]
    over = [e for e in events if e.get("stage") == "builder_kg_skip_over_hops"]
    assert len(over) == 1
    assert over[0]["max_hops"] == 3


def test_build_skips_rows_without_evidence():
    kg = MagicMock()
    kg.run_pattern.return_value = [_row(evidence=[])]
    trace = MagicMock()
    builder = KGMultiHopMaterializer(ctx=MagicMock(), kg=kg, trace=trace)
    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),
        version=1,
    )
    assert items == []


def test_build_logs_kg_error_and_continues_with_next_rule():
    """A rule that fails in the KG client must not abort the builder."""
    kg = MagicMock()
    call_count = {"n": 0}

    def side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("neo4j down briefly")
        return [_row()]

    kg.run_pattern.side_effect = side_effect
    trace = MagicMock()
    builder = KGMultiHopMaterializer(ctx=MagicMock(), kg=kg, trace=trace)
    adapter = _adapter(
        rules=[
            _rule("(a)-[:X]->(b)", "alpha"),
            _rule("(a)-[:Y]->(b)", "beta"),
        ]
    )
    items = builder.build(
        subscription_id="s", profile_id="p", adapter=adapter, version=1
    )
    assert len(items) == 1
    events = [c.args[0] for c in trace.append.call_args_list]
    errs = [e for e in events if e.get("stage") == "builder_kg_error"]
    assert len(errs) == 1
    assert errs[0]["produces"] == "alpha"


def test_confidence_falls_back_to_rule_floor_when_missing():
    kg = MagicMock()
    row = _row()
    row.pop("confidence")
    kg.run_pattern.return_value = [row]
    trace = MagicMock()
    builder = KGMultiHopMaterializer(ctx=MagicMock(), kg=kg, trace=trace)
    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(
            rules=[_rule("(a)-[:X]->(b)", "x", confidence_floor=0.42)]
        ),
        version=1,
    )
    assert items[0].confidence == pytest.approx(0.42)


def test_text_uses_src_relation_dst_per_errata_section_3():
    kg = MagicMock()
    kg.run_pattern.return_value = [_row(src="A", dst="C")]
    trace = MagicMock()
    builder = KGMultiHopMaterializer(ctx=MagicMock(), kg=kg, trace=trace)
    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),
        version=1,
    )
    assert items[0].text == "A -[indirectly_funds]-> C"
