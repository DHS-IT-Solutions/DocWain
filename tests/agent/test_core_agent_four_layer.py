"""Phase 3 Task 3 — four-layer SME retrieval integration in CoreAgent.

Exercises the :meth:`CoreAgent._retrieve_four_layers` helper that
dispatches Layer A/B/C/D in parallel via
:meth:`src.retrieval.unified_retriever.UnifiedRetriever.retrieve_four_layers`.
The harness mocks every external dependency so the parallel orchestration,
profile-isolation filters, flag gating, and graceful-degradation paths are
exercised without touching Qdrant, Neo4j, Redis, or the LLM gateway.
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agent.core_agent import CoreAgent
from src.config import feature_flags as ff
from src.config.feature_flags import (
    ENABLE_SME_RETRIEVAL,
    SME_REDESIGN_ENABLED,
    init_flag_resolver,
    reset_flag_set_version,
)
from src.retrieval.types import RetrievalBundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MutableStore:
    def __init__(self, overrides: dict[str, dict[str, bool]] | None = None) -> None:
        self._by: dict[str, dict[str, bool]] = overrides or {}

    def get_subscription_overrides(self, sub: str) -> dict[str, bool]:
        return dict(self._by.get(sub, {}))

    def set_subscription_override(self, sub: str, flag: str, value: bool) -> None:
        self._by.setdefault(sub, {})[flag] = bool(value)


def _build_agent(*, sme=None, kg=None, qdrant=None, hybrid=None) -> CoreAgent:
    # Use MagicMock for llm/embedder/mongodb — the 4-layer helper never
    # invokes them. The Reasoner path isn't touched in these tests.
    llm = MagicMock()
    llm.backend = "test"
    emb = MagicMock()
    mongo = MagicMock()
    return CoreAgent(
        llm_gateway=llm,
        qdrant_client=qdrant or MagicMock(),
        embedder=emb,
        mongodb=mongo,
        sme_retriever=sme,
        sme_kg_client=kg,
        hybrid_searcher=hybrid,
    )


@pytest.fixture(autouse=True)
def _flags_master_on() -> None:
    """Master ON + ENABLE_SME_RETRIEVAL ON for sub_fin by default."""
    store = _MutableStore(
        overrides={
            "sub_fin": {SME_REDESIGN_ENABLED: True, ENABLE_SME_RETRIEVAL: True},
        }
    )
    init_flag_resolver(store=store)
    reset_flag_set_version()
    yield
    reset_flag_set_version()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_four_layers_invoked_and_returns_bundle() -> None:
    kg = MagicMock()
    kg.one_hop.return_value = [
        {"src": "n1", "dst": "n2", "type": "cites", "evidence": []},
    ]
    kg.inferred_relations.return_value = [
        {
            "src": "n1",
            "dst": "n3",
            "relation_type": "correlates_with",
            "confidence": 0.82,
            "evidence": [],
            "inference_path": [],
        },
    ]
    sme = MagicMock()
    sme.retrieve.return_value = [
        {
            "kind": "sme_artifact",
            "artifact_type": "dossier",
            "text": "Q3 revenue narrative",
            "confidence": 0.9,
            "evidence": ["d1#c1"],
            "score": 0.85,
        },
    ]
    # Override Layer A via injected fake on the qdrant bridge.
    agent = _build_agent(sme=sme, kg=kg, qdrant=MagicMock())

    # Stub the internal Layer A helper via monkey-patching on the SME
    # UnifiedRetriever path — inject via the layer_a_fn override path by
    # bypassing the helper call inside the test.
    bundle = agent._retrieve_four_layers(
        query="revenue trend",
        subscription_id="sub_fin",
        profile_id="prof_fin",
        query_understanding={"intent": "analyze", "entities": ["Q3"]},
    )

    assert isinstance(bundle, RetrievalBundle)
    # Layer A may be empty because we didn't stub qdrant — but Layer B + C fired.
    kg.one_hop.assert_called()
    # inferred_relations only fires when BOTH flags are on; ENABLE_KG_SYNTHESIZED_EDGES
    # is default off under our fixture, so inferred doesn't run — Layer B = direct only.
    assert all(row.get("kind") == "kg_direct" for row in bundle.layer_b_kg)
    sme.retrieve.assert_called_once()
    assert len(bundle.layer_c_sme) == 1
    assert bundle.layer_c_sme[0]["artifact_type"] == "dossier"
    # Layer D always returns [] in Phase 3.
    assert bundle.layer_d_url == []
    # Every layer reported per-layer wall clock.
    for name in ("layer_a", "layer_b", "layer_c", "layer_d"):
        assert name in bundle.per_layer_ms


def test_layer_c_skipped_when_flag_off() -> None:
    kg = MagicMock()
    kg.one_hop.return_value = []
    kg.inferred_relations.return_value = []
    sme = MagicMock()
    sme.retrieve.return_value = []
    # Flip ENABLE_SME_RETRIEVAL off for the subscription.
    store = _MutableStore(
        overrides={"sub_off": {SME_REDESIGN_ENABLED: True, ENABLE_SME_RETRIEVAL: False}}
    )
    init_flag_resolver(store=store)
    agent = _build_agent(sme=sme, kg=kg)
    bundle = agent._retrieve_four_layers(
        query="x",
        subscription_id="sub_off",
        profile_id="prof_off",
        query_understanding={"intent": "analyze"},
    )
    sme.retrieve.assert_not_called()
    assert bundle.layer_c_sme == []


def test_layers_run_in_parallel() -> None:
    # Inject slow layer fns; verify wall-clock elapsed is closer to max(legs)
    # than to sum(legs). Small deltas keep CI-stable.
    SLOW = 0.12

    class _SlowSME:
        def retrieve(self, **kw):
            time.sleep(SLOW)
            return [{"kind": "sme_artifact", "text": "x", "confidence": 0.8,
                     "artifact_type": "dossier"}]

    class _SlowKG:
        def one_hop(self, **kw):
            time.sleep(SLOW)
            return [{"src": "a", "dst": "b", "type": "rel", "evidence": []}]

        def inferred_relations(self, **kw):
            return []

    agent = _build_agent(sme=_SlowSME(), kg=_SlowKG())
    t0 = time.perf_counter()
    bundle = agent._retrieve_four_layers(
        query="q",
        subscription_id="sub_fin",
        profile_id="p",
        query_understanding={"intent": "analyze"},
    )
    elapsed = time.perf_counter() - t0
    # Two slow legs (B + C) at ~0.12s each → serial would be ~0.24s;
    # parallel must come in well under 0.20s.
    assert elapsed < 0.20, f"layers ran serially: {elapsed:.3f}s"
    assert len(bundle.layer_c_sme) == 1
    # One_hop row is tagged "kg_direct" by retrieve_layer_b.
    assert bundle.layer_b_kg and bundle.layer_b_kg[0]["kind"] == "kg_direct"


def test_layer_failure_does_not_kill_others() -> None:
    kg = MagicMock()
    kg.one_hop.return_value = [
        {"src": "n", "dst": "m", "type": "r", "evidence": []},
    ]
    kg.inferred_relations.return_value = []

    class _BoomSME:
        def retrieve(self, **kw):
            raise RuntimeError("qdrant down")

    agent = _build_agent(sme=_BoomSME(), kg=kg)
    bundle = agent._retrieve_four_layers(
        query="q",
        subscription_id="sub_fin",
        profile_id="p",
        query_understanding={"intent": "analyze"},
    )
    # Layer B succeeded, Layer C degraded.
    assert bundle.layer_b_kg and len(bundle.layer_b_kg) == 1
    assert bundle.layer_c_sme == []
    # Per ERRATA §11: full name only, no short "c" append.
    assert "layer_c" in bundle.degraded_layers
    assert "c" not in bundle.degraded_layers
    # Other layers are not marked degraded.
    assert "layer_a" not in bundle.degraded_layers
    assert "layer_b" not in bundle.degraded_layers
    assert "layer_d" not in bundle.degraded_layers


def test_profile_isolation_propagated_to_every_layer() -> None:
    kg = MagicMock()
    kg.one_hop.return_value = []
    kg.inferred_relations.return_value = []
    sme = MagicMock()
    sme.retrieve.return_value = []
    agent = _build_agent(sme=sme, kg=kg)
    agent._retrieve_four_layers(
        query="q",
        subscription_id="sub_fin",
        profile_id="prof_A",
        query_understanding={"intent": "analyze"},
    )
    kw_kg = kg.one_hop.call_args.kwargs
    assert kw_kg["subscription_id"] == "sub_fin"
    assert kw_kg["profile_id"] == "prof_A"
    kw_sme = sme.retrieve.call_args.kwargs
    assert kw_sme["subscription_id"] == "sub_fin"
    assert kw_sme["profile_id"] == "prof_A"


def test_no_internal_timeout_on_as_completed() -> None:
    """Memory rule — retrieve_four_layers must not pass a timeout to as_completed."""
    import concurrent.futures as cf
    from unittest.mock import patch

    captured_kwargs: list[dict] = []

    real = cf.as_completed

    def spy(*a, **kw):
        captured_kwargs.append(kw)
        return real(*a, **kw)

    kg = MagicMock()
    kg.one_hop.return_value = []
    kg.inferred_relations.return_value = []
    sme = MagicMock()
    sme.retrieve.return_value = []
    agent = _build_agent(sme=sme, kg=kg)
    with patch("src.retrieval.unified_retriever.as_completed", side_effect=spy):
        agent._retrieve_four_layers(
            query="q",
            subscription_id="sub_fin",
            profile_id="p",
            query_understanding={"intent": "analyze"},
        )
    assert captured_kwargs  # sanity
    for kw in captured_kwargs:
        assert "timeout" not in kw, f"timeout leaked: {kw}"


def test_layer_d_always_empty_in_phase3() -> None:
    agent = _build_agent(sme=None, kg=None)
    bundle = agent._retrieve_four_layers(
        query="q",
        subscription_id="sub_fin",
        profile_id="p",
        query_understanding={"intent": "analyze"},
    )
    assert bundle.layer_d_url == []


def test_missing_subscription_raises() -> None:
    agent = _build_agent()
    with pytest.raises(ValueError, match="subscription_id"):
        agent._retrieve_four_layers(
            query="q",
            subscription_id="",
            profile_id="p",
            query_understanding={"intent": "analyze"},
        )


def test_missing_profile_raises() -> None:
    agent = _build_agent()
    with pytest.raises(ValueError, match="profile_id"):
        agent._retrieve_four_layers(
            query="q",
            subscription_id="s",
            profile_id="",
            query_understanding={"intent": "analyze"},
        )


def test_layer_c_is_off_when_layer_c_dependency_unset() -> None:
    # No sme_retriever injected → Layer C returns [] silently even with flag on.
    kg = MagicMock()
    kg.one_hop.return_value = []
    kg.inferred_relations.return_value = []
    agent = _build_agent(sme=None, kg=kg)
    bundle = agent._retrieve_four_layers(
        query="q",
        subscription_id="sub_fin",
        profile_id="p",
        query_understanding={"intent": "analyze"},
    )
    assert bundle.layer_c_sme == []
    assert "layer_c" not in bundle.degraded_layers  # empty ≠ degraded
