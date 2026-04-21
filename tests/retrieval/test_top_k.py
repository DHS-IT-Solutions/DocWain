"""Phase 3 Task 6 — adaptive per-layer top-K tests."""
from __future__ import annotations

import pytest

from src.retrieval.top_k import LayerTopK, base_top_k, compute_top_k


# ---------------------------------------------------------------------------
# Canonical table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("greeting", LayerTopK(0, 0, 0)),
        ("identity", LayerTopK(0, 0, 0)),
        ("lookup", LayerTopK(5, 0, 0)),
        ("count", LayerTopK(5, 0, 0)),
        ("extract", LayerTopK(10, 0, 2)),
        ("list", LayerTopK(10, 0, 2)),
        ("aggregate", LayerTopK(10, 0, 2)),
        ("compare", LayerTopK(12, 5, 5)),
        ("summarize", LayerTopK(12, 5, 5)),
        ("overview", LayerTopK(12, 5, 5)),
        ("analyze", LayerTopK(15, 10, 10)),
        ("diagnose", LayerTopK(15, 10, 10)),
        ("recommend", LayerTopK(15, 10, 10)),
        ("investigate", LayerTopK(15, 10, 10)),
    ],
)
def test_base_table_matches_plan(intent: str, expected: LayerTopK) -> None:
    assert base_top_k(intent) == expected


def test_unknown_intent_falls_back_to_lookup_shape() -> None:
    assert base_top_k("brand_new_intent") == LayerTopK(5, 0, 0)


def test_simple_intents_never_touch_layers_even_with_complexity() -> None:
    # Even if the caller passes heavy complexity signals, simple intents
    # must stay at (0,0,0) — Phase 3 gating rule.
    qa = {"entities": list(range(20)), "sub_queries": list(range(10))}
    for intent in ("greeting", "identity", "meta", "farewell", "thanks"):
        assert compute_top_k(intent, qa) == LayerTopK(0, 0, 0)


# ---------------------------------------------------------------------------
# Complexity scaling
# ---------------------------------------------------------------------------


def test_multi_part_query_bumps_layer_a() -> None:
    base = compute_top_k("analyze", {"sub_queries": []})
    bumped = compute_top_k("analyze", {"sub_queries": [1, 2, 3, 4, 5]})
    # Two thresholds crossed → +3 +3 = +6 on layer A.
    assert bumped.a == base.a + 6
    assert bumped.b == base.b
    assert bumped.c == base.c


def test_entity_heavy_query_bumps_layer_b() -> None:
    base = compute_top_k("analyze", None)
    bumped = compute_top_k(
        "analyze", {"entities": ["a", "b", "c", "d", "e", "f", "g"]}
    )
    assert bumped.b == base.b + 4  # both thresholds crossed
    assert bumped.a == base.a
    assert bumped.c == base.c


def test_long_temporal_span_bumps_layer_c() -> None:
    base = compute_top_k("analyze", None)
    short = compute_top_k("analyze", {"temporal_span_months": 6})
    medium = compute_top_k("analyze", {"temporal_span_months": 18})
    long = compute_top_k("analyze", {"temporal_span_months": 48})
    assert short == base
    assert medium.c == base.c + 2
    assert long.c == base.c + 5


def test_intents_without_layer_b_ignore_entity_bumps() -> None:
    # extract is ``a=10, b=0, c=2``; entity bumps must not resurrect
    # layer B because it is intentionally skipped by the gating table.
    tk = compute_top_k(
        "extract", {"entities": ["a", "b", "c", "d", "e", "f", "g", "h"]}
    )
    assert tk.b == 0


def test_intents_without_layer_c_ignore_temporal_bumps() -> None:
    tk = compute_top_k("lookup", {"temporal_span_months": 60})
    assert tk.c == 0


# ---------------------------------------------------------------------------
# Adapter caps
# ---------------------------------------------------------------------------


def test_adapter_caps_clamp_down() -> None:
    tk = compute_top_k(
        "analyze",
        {"sub_queries": [1, 2, 3, 4], "entities": ["a", "b", "c", "d", "e", "f"]},
        adapter_caps={"a": 12, "b": 8, "c": 10},
    )
    assert tk.a == 12
    assert tk.b == 8
    assert tk.c == 10


def test_adapter_caps_never_raise_above_computed() -> None:
    # Caps that are above the computed value leave it alone — caps are
    # upper bounds, not floors.
    tk = compute_top_k(
        "lookup", None, adapter_caps={"a": 50, "b": 50, "c": 50}
    )
    assert tk == LayerTopK(5, 0, 0)


def test_layer_topk_clamp_missing_keys_leave_others_alone() -> None:
    base = LayerTopK(15, 10, 10)
    clamped = base.clamp(caps={"a": 5})
    assert clamped == LayerTopK(5, 10, 10)


def test_empty_query_analysis_equals_base() -> None:
    for intent in ("lookup", "extract", "compare", "analyze"):
        assert compute_top_k(intent, None) == base_top_k(intent)


def test_query_analysis_missing_entities_counts_zero() -> None:
    tk = compute_top_k("analyze", {"sub_queries": []})
    assert tk == base_top_k("analyze")
