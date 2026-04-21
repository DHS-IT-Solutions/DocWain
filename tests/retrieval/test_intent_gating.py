"""Phase 3 Task 9 — intent-aware layer gating tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.intent_gating import GateDecision, IntentGate
from src.retrieval.unified_retriever import UnifiedRetriever


# ---------------------------------------------------------------------------
# Table-driven decision tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("greeting",    GateDecision(False, False, False)),
        ("identity",    GateDecision(False, False, False)),
        ("meta",        GateDecision(False, False, False)),
        ("farewell",    GateDecision(False, False, False)),
        ("thanks",      GateDecision(False, False, False)),
        ("lookup",      GateDecision(True,  False, False)),
        ("count",       GateDecision(True,  False, False)),
        ("extract",     GateDecision(True,  False, True)),
        ("list",        GateDecision(True,  False, True)),
        ("aggregate",   GateDecision(True,  False, True)),
        ("compare",     GateDecision(True,  True,  True)),
        ("summarize",   GateDecision(True,  True,  True)),
        ("overview",    GateDecision(True,  True,  True)),
        ("analyze",     GateDecision(True,  True,  True)),
        ("diagnose",    GateDecision(True,  True,  True)),
        ("recommend",   GateDecision(True,  True,  True)),
        ("investigate", GateDecision(True,  True,  True)),
    ],
)
def test_gate_table(intent, expected):
    assert IntentGate().decide(intent) == expected


def test_unknown_intent_is_conservative():
    d = IntentGate().decide("exotic_new_intent")
    assert d.run_a and d.run_b and d.run_c


def test_compact_mode_closes_b_c_for_analytical():
    gate = IntentGate()
    d = gate.decide("analyze", user_requested_compact=True)
    assert d.run_a is True
    assert d.run_b is False
    assert d.run_c is False


def test_compact_mode_respects_gated_a_off():
    """A compact request against ``greeting`` still skips Layer A because
    the base decision already had ``run_a=False``."""
    d = IntentGate().decide("greeting", user_requested_compact=True)
    assert d == GateDecision(False, False, False)


def test_decision_is_frozen():
    d = IntentGate().decide("analyze")
    with pytest.raises(Exception):  # FrozenInstanceError is Exception subclass
        d.run_a = False  # type: ignore[misc]


def test_custom_table_overrides_default():
    table = {
        "special": GateDecision(True, False, True),
    }
    gate = IntentGate(table=table)
    # Known override:
    assert gate.decide("special") == GateDecision(True, False, True)
    # Missing → conservative fallback.
    assert gate.decide("lookup") == GateDecision(True, True, True)


# ---------------------------------------------------------------------------
# Integration with retrieve_four_layers — gated layers are never submitted.
# ---------------------------------------------------------------------------


def test_integration_lookup_skips_b_c():
    """Simple lookup intent → only Layer A + placeholder Layer D dispatched.

    We stub every layer fn with a MagicMock and assert only the gate-enabled
    ones get called. Layer D always runs (placeholder, no cost).
    """
    ur = UnifiedRetriever(qdrant_client=MagicMock(), kg_client=MagicMock(),
                          sme=MagicMock())
    calls = {"a": 0, "b": 0, "c": 0, "d": 0}
    def a(): calls["a"] += 1; return [{"doc_id": "1", "chunk_id": "1", "text": "x"}]
    def b(): calls["b"] += 1; return []
    def c(): calls["c"] += 1; return []
    def d(): calls["d"] += 1; return []

    gate = IntentGate()
    with patch(
        "src.retrieval.unified_retriever._safe_flag_resolver",
        return_value=None,
    ):
        bundle = ur.retrieve_four_layers(
            query="What is the Q3 number?",
            subscription_id="s", profile_id="p",
            query_understanding={"intent": "lookup"},
            layer_a_fn=a, layer_b_fn=b, layer_c_fn=c, layer_d_fn=d,
            gate=gate,
        )
    assert calls == {"a": 1, "b": 0, "c": 0, "d": 1}
    # Layer B + C slots are empty because gated off, not degraded.
    assert bundle.layer_b_kg == []
    assert bundle.layer_c_sme == []
    assert "layer_b" not in bundle.degraded_layers
    assert "layer_c" not in bundle.degraded_layers


def test_integration_analyze_runs_all_three():
    ur = UnifiedRetriever(qdrant_client=MagicMock(), kg_client=MagicMock(),
                          sme=MagicMock())
    calls = {"a": 0, "b": 0, "c": 0, "d": 0}
    def a(): calls["a"] += 1; return []
    def b(): calls["b"] += 1; return []
    def c(): calls["c"] += 1; return []
    def d(): calls["d"] += 1; return []

    # Flag resolver must be "ON" for Layer C to dispatch.
    class _On:
        def is_enabled(self, sub, flag): return True

    with patch(
        "src.retrieval.unified_retriever._safe_flag_resolver", return_value=_On()
    ):
        ur.retrieve_four_layers(
            query="Analyze Q3 trend",
            subscription_id="s", profile_id="p",
            query_understanding={"intent": "analyze"},
            layer_a_fn=a, layer_b_fn=b, layer_c_fn=c, layer_d_fn=d,
            gate=IntentGate(),
        )
    assert calls == {"a": 1, "b": 1, "c": 1, "d": 1}


def test_integration_greeting_gates_everything_but_layer_d():
    ur = UnifiedRetriever(qdrant_client=MagicMock(), kg_client=MagicMock(),
                          sme=MagicMock())
    calls = {"a": 0, "b": 0, "c": 0, "d": 0}
    def a(): calls["a"] += 1; return []
    def b(): calls["b"] += 1; return []
    def c(): calls["c"] += 1; return []
    def d(): calls["d"] += 1; return []

    with patch(
        "src.retrieval.unified_retriever._safe_flag_resolver", return_value=None
    ):
        ur.retrieve_four_layers(
            query="hello",
            subscription_id="s", profile_id="p",
            query_understanding={"intent": "greeting"},
            layer_a_fn=a, layer_b_fn=b, layer_c_fn=c, layer_d_fn=d,
            gate=IntentGate(),
        )
    assert calls["a"] == 0
    assert calls["b"] == 0
    assert calls["c"] == 0


def test_integration_without_gate_runs_all_layers():
    """Backward compat: callers that don't pass ``gate`` get the original
    behaviour (all layers dispatched, Layer C still flag-gated)."""
    ur = UnifiedRetriever(qdrant_client=MagicMock(), kg_client=MagicMock(),
                          sme=MagicMock())
    calls = {"a": 0, "b": 0, "c": 0, "d": 0}
    def a(): calls["a"] += 1; return []
    def b(): calls["b"] += 1; return []
    def c(): calls["c"] += 1; return []
    def d(): calls["d"] += 1; return []

    class _On:
        def is_enabled(self, sub, flag): return True

    with patch(
        "src.retrieval.unified_retriever._safe_flag_resolver", return_value=_On()
    ):
        ur.retrieve_four_layers(
            query="hello",
            subscription_id="s", profile_id="p",
            query_understanding={"intent": "greeting"},
            layer_a_fn=a, layer_b_fn=b, layer_c_fn=c, layer_d_fn=d,
            # NO gate — should not skip any layer.
        )
    assert calls == {"a": 1, "b": 1, "c": 1, "d": 1}


def test_integration_compact_mode_override():
    ur = UnifiedRetriever(qdrant_client=MagicMock(), kg_client=MagicMock(),
                          sme=MagicMock())
    calls = {"a": 0, "b": 0, "c": 0, "d": 0}
    def a(): calls["a"] += 1; return []
    def b(): calls["b"] += 1; return []
    def c(): calls["c"] += 1; return []
    def d(): calls["d"] += 1; return []

    class _On:
        def is_enabled(self, sub, flag): return True

    with patch(
        "src.retrieval.unified_retriever._safe_flag_resolver", return_value=_On()
    ):
        ur.retrieve_four_layers(
            query="analyse it", subscription_id="s", profile_id="p",
            query_understanding={"intent": "analyze"},
            layer_a_fn=a, layer_b_fn=b, layer_c_fn=c, layer_d_fn=d,
            gate=IntentGate(),
            user_requested_compact=True,
        )
    assert calls["a"] == 1
    assert calls["b"] == 0
    assert calls["c"] == 0
