"""Tests for :class:`SMESynthesizer` — control-flow only in Phase 1.

The builders all return empty lists, so we verify the synthesizer
orchestrates the trace / verifier / storage dance correctly rather than any
synthesis behavior.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.adapter_schema import Adapter
from src.intelligence.sme.synthesizer import SMESynthesizer, SynthesizerDeps
from src.intelligence.sme.verifier import Verdict


@pytest.fixture
def deps() -> SynthesizerDeps:
    trace_writer = MagicMock()
    d = SynthesizerDeps(
        adapter_loader=MagicMock(),
        storage=MagicMock(),
        verifier=MagicMock(),
        trace_writer=trace_writer,
        builders={
            name: MagicMock()
            for name in (
                "dossier",
                "insight",
                "comparison",
                "kg_edge",
                "recommendation",
            )
        },
    )
    adapter = MagicMock(spec=Adapter)
    adapter.version = "1.0.0"
    adapter.domain = "generic"
    adapter.content_hash = "h"
    adapter.source_path = "x"
    d.adapter_loader.load.return_value = adapter
    d.adapter_loader.last_load_metadata.return_value = {
        "version": "1.0.0",
        "content_hash": "h",
        "source_path": "x",
    }
    for builder in d.builders.values():
        builder.build.return_value = []
    d.verifier.verify_batch.return_value = []
    return d


def _run(deps: SynthesizerDeps) -> dict[str, int]:
    return SMESynthesizer(deps).run(
        subscription_id="s",
        profile_id="p",
        profile_domain="generic",
        synthesis_version=1,
    )


def test_opens_and_closes_trace(deps: SynthesizerDeps) -> None:
    _run(deps)
    deps.trace_writer.open.assert_called_once_with(
        subscription_id="s", profile_id="p", synthesis_id="s:p:1"
    )
    deps.trace_writer.close.assert_called_once()


def test_calls_all_five_builders(deps: SynthesizerDeps) -> None:
    _run(deps)
    for builder in deps.builders.values():
        builder.build.assert_called_once()


def test_verifies_and_persists_per_builder(deps: SynthesizerDeps) -> None:
    _run(deps)
    assert deps.verifier.verify_batch.call_count == 5
    assert deps.storage.persist_items.call_count == 5


def test_records_adapter_version_and_hash(deps: SynthesizerDeps) -> None:
    _run(deps)
    events = [call.args[0] for call in deps.trace_writer.append.call_args_list]
    assert any(e.get("stage") == "start" for e in events)
    assert any(e.get("adapter_version") == "1.0.0" for e in events)
    assert any(e.get("adapter_hash") == "h" for e in events)


def test_emits_builder_complete_and_complete_events(deps: SynthesizerDeps) -> None:
    _run(deps)
    events = [call.args[0] for call in deps.trace_writer.append.call_args_list]
    stages = [e["stage"] for e in events]
    assert stages[0] == "start"
    assert stages[-1] == "complete"
    assert stages.count("builder_complete") == 5


def test_close_runs_even_on_exception(deps: SynthesizerDeps) -> None:
    deps.adapter_loader.load.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        _run(deps)
    deps.trace_writer.close.assert_called_once()


def test_verifier_drops_logged(deps: SynthesizerDeps) -> None:
    from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef

    item = ArtifactItem(
        item_id="dropme",
        artifact_type="insight",
        subscription_id="s",
        profile_id="p",
        text="t",
        evidence=[EvidenceRef(doc_id="d", chunk_id="c")],
        confidence=0.5,
    )

    def _builder_build(**_kwargs):
        return [item]

    # Only the "insight" builder returns an item; it's dropped by the verifier.
    deps.builders["insight"].build.side_effect = _builder_build
    deps.verifier.verify_batch.side_effect = lambda items, _ctx: [
        Verdict(
            item_id=it.item_id,
            passed=False,
            adjusted_item=None,
            failing_check="evidence_validity",
            drop_reason="stub",
        )
        for it in items
    ]
    _run(deps)
    events = [call.args[0] for call in deps.trace_writer.append.call_args_list]
    drops = [e for e in events if e["stage"] == "verifier_drop"]
    assert len(drops) == 1
    assert drops[0]["item_id"] == "dropme"
    assert drops[0]["failing_check"] == "evidence_validity"


def test_run_returns_per_builder_counts(deps: SynthesizerDeps) -> None:
    counts = _run(deps)
    assert counts == {
        "dossier": 0,
        "insight": 0,
        "comparison": 0,
        "kg_edge": 0,
        "recommendation": 0,
    }


def test_phase2_threads_insight_items_to_recommendation_builder() -> None:
    """Phase 2 orchestrator wires verified Insight Index items into the
    Recommendation Bank builder as ``insight_items=...``. The dependency is
    detected by introspecting the builder signature, so a real Phase 2
    builder exposes the kwarg while MagicMock doubles continue to match the
    Phase 1 contract.

    We validate the wiring by hand-rolling a class whose ``build`` signature
    carries the kwarg — the orchestrator should propagate the verified
    ``insight`` items into ``recommendation.build(insight_items=[...])``.
    """
    from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef

    captured: dict[str, Any] = {}

    class _RealBuilder:
        artifact_type = "recommendation"

        def build(
            self,
            *,
            subscription_id: str,
            profile_id: str,
            adapter: Adapter,
            version: int,
            insight_items: list[ArtifactItem] | None = None,
        ) -> list[ArtifactItem]:
            captured["insight_items"] = insight_items
            return []

    trace_writer = MagicMock()
    adapter = MagicMock(spec=Adapter)
    adapter.version = "1.0.0"
    adapter.domain = "generic"
    adapter.content_hash = "h"
    adapter.source_path = "x"
    loader = MagicMock()
    loader.load.return_value = adapter
    loader.last_load_metadata.return_value = {
        "version": "1.0.0",
        "content_hash": "h",
        "source_path": "x",
    }

    insight_item = ArtifactItem(
        item_id="insight:1",
        artifact_type="insight",
        subscription_id="s",
        profile_id="p",
        text="rev rose",
        evidence=[EvidenceRef(doc_id="d1", chunk_id="c1")],
        confidence=0.7,
    )

    insight_builder = MagicMock()
    insight_builder.build.return_value = [insight_item]

    def _verify(items, ctx):
        return [
            Verdict(item_id=it.item_id, passed=True, adjusted_item=it)
            for it in items
        ]

    verifier = MagicMock()
    verifier.verify_batch.side_effect = _verify
    storage = MagicMock()

    # Order matters: insight MUST precede recommendation so the accepted
    # insight items are available when the recommendation builder runs.
    dossier = MagicMock()
    dossier.build.return_value = []
    comparison = MagicMock()
    comparison.build.return_value = []
    kg = MagicMock()
    kg.build.return_value = []

    deps = SynthesizerDeps(
        adapter_loader=loader,
        storage=storage,
        verifier=verifier,
        trace_writer=trace_writer,
        builders={
            "dossier": dossier,
            "insight": insight_builder,
            "comparison": comparison,
            "kg_edge": kg,
            "recommendation": _RealBuilder(),
        },
    )
    SMESynthesizer(deps).run(
        subscription_id="s",
        profile_id="p",
        profile_domain="generic",
        synthesis_version=1,
    )
    assert captured["insight_items"] == [insight_item]


def test_phase2_mock_recommendation_builder_receives_no_insight_items_kwarg() -> (
    None
):
    """MagicMock builders do NOT expose an explicit ``insight_items`` kwarg
    (introspection returns ``*args, **kwargs`` which we deliberately decline
    per the helper's comment). Phase 1 callers therefore keep calling
    ``builder.build(subscription_id=, profile_id=, adapter=, version=)``
    without extra args.
    """
    trace_writer = MagicMock()
    adapter = MagicMock(spec=Adapter)
    adapter.version = "1.0.0"
    adapter.domain = "g"
    adapter.content_hash = "h"
    adapter.source_path = "x"
    loader = MagicMock()
    loader.load.return_value = adapter
    verifier = MagicMock()
    verifier.verify_batch.return_value = []
    storage = MagicMock()
    rec = MagicMock()
    rec.build.return_value = []
    deps = SynthesizerDeps(
        adapter_loader=loader,
        storage=storage,
        verifier=verifier,
        trace_writer=trace_writer,
        builders={
            "dossier": MagicMock(),
            "insight": MagicMock(),
            "comparison": MagicMock(),
            "kg_edge": MagicMock(),
            "recommendation": rec,
        },
    )
    for builder in deps.builders.values():
        builder.build.return_value = []
    SMESynthesizer(deps).run(
        subscription_id="s",
        profile_id="p",
        profile_domain="g",
        synthesis_version=1,
    )
    # The recommendation mock received the base kwargs only — no
    # ``insight_items`` key.
    call = rec.build.call_args
    assert "insight_items" not in call.kwargs
