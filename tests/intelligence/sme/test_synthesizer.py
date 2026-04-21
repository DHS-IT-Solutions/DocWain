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
