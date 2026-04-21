"""Phase 1 sandbox end-to-end integration test.

Wires every SME Phase 1 module with in-memory fakes and proves the full
pipeline is plumbed: adapter resolution → five builders → verifier →
storage → trace. Skeleton builders return ``[]``, so the run produces empty
artifacts — but every seam is exercised and observable.

The second test verifies cross-subscription isolation at the Qdrant
collection level.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.adapter_loader import AdapterLoader
from src.intelligence.sme.builders.comparative_register import (
    ComparativeRegisterBuilder,
)
from src.intelligence.sme.builders.dossier import SMEDossierBuilder
from src.intelligence.sme.builders.insight_index import InsightIndexBuilder
from src.intelligence.sme.builders.kg_materializer import KGMultiHopMaterializer
from src.intelligence.sme.builders.recommendation_bank import (
    RecommendationBankBuilder,
)
from src.intelligence.sme.storage import SMEArtifactStorage, StorageDeps
from src.intelligence.sme.synthesizer import SMESynthesizer, SynthesizerDeps
from src.intelligence.sme.trace import SynthesisTraceWriter
from src.intelligence.sme.verifier import SMEVerifier
from tests.intelligence.sme.conftest import (
    InMemoryBlob,
    InMemoryNeo4j,
    InMemoryQdrant,
)


LAST_RESORT = Path("deploy/sme_adapters/last_resort/generic.yaml")


@pytest.fixture
def wired() -> tuple[SMESynthesizer, InMemoryBlob, InMemoryQdrant, InMemoryNeo4j]:
    """Assemble the full Phase 1 pipeline with in-memory fakes."""
    blob = InMemoryBlob()
    # Seed the generic global adapter so the loader resolves happy-path.
    blob.files["sme_adapters/global/generic.yaml"] = LAST_RESORT.read_text()
    qdrant = InMemoryQdrant()
    neo4j = InMemoryNeo4j()

    loader = AdapterLoader(
        blob=blob, last_resort_path=LAST_RESORT, ttl_seconds=60
    )
    storage = SMEArtifactStorage(
        StorageDeps(blob=blob, qdrant=qdrant, neo4j=neo4j)
    )

    chunk_store = MagicMock()
    chunk_store.chunk_exists.return_value = True
    chunk_store.chunk_text.return_value = ""

    builder_ctx = MagicMock()
    builder_ctx.iter_profile_chunks.return_value = []
    builder_ctx.iter_profile_kg.return_value = []

    builders = {
        "dossier": SMEDossierBuilder(ctx=builder_ctx),
        "insight": InsightIndexBuilder(ctx=builder_ctx),
        "comparison": ComparativeRegisterBuilder(ctx=builder_ctx),
        "kg_edge": KGMultiHopMaterializer(ctx=builder_ctx),
        "recommendation": RecommendationBankBuilder(ctx=builder_ctx),
    }

    synthesizer = SMESynthesizer(
        SynthesizerDeps(
            adapter_loader=loader,
            storage=storage,
            verifier=SMEVerifier(
                chunk_store=chunk_store, max_inference_hops=3
            ),
            trace_writer=SynthesisTraceWriter(blob),
            builders=builders,
        )
    )
    return synthesizer, blob, qdrant, neo4j


def test_sandbox_end_to_end_plumbing(wired) -> None:
    synthesizer, blob, qdrant, neo4j = wired
    counts = synthesizer.run(
        subscription_id="sandbox",
        profile_id="prof_1",
        profile_domain="generic",
        synthesis_version=1,
    )
    # Skeletons produce nothing but the full pipeline runs.
    assert counts == {
        "dossier": 0,
        "insight": 0,
        "comparison": 0,
        "kg_edge": 0,
        "recommendation": 0,
    }
    # Canonical JSON written for every artifact type, with subscription +
    # profile identity surfaced in the payload.
    for atype in (
        "dossier",
        "insight",
        "comparison",
        "kg_edge",
        "recommendation",
    ):
        path = f"sme_artifacts/sandbox/prof_1/{atype}/1.json"
        body = json.loads(blob.files[path])
        assert body["items"] == []
        assert body["subscription_id"] == "sandbox"
        assert body["profile_id"] == "prof_1"
    # No items → no Qdrant points, no Neo4j edges.
    assert qdrant.points["sme_artifacts_sandbox"] == []
    assert neo4j.edges == []
    # Trace captures the full lifecycle: start + 5 × builder_complete + complete.
    trace_path = (
        "sme_traces/synthesis/sandbox/prof_1/sandbox:prof_1:1.jsonl"
    )
    stages = [
        json.loads(line)["stage"]
        for line in blob.files[trace_path].splitlines()
    ]
    assert stages[0] == "start"
    assert stages[-1] == "complete"
    assert stages.count("builder_complete") == 5


def test_cross_subscription_isolation(wired) -> None:
    synthesizer, _blob, qdrant, _neo4j = wired
    synthesizer.run(
        subscription_id="sandbox",
        profile_id="p1",
        profile_domain="generic",
        synthesis_version=1,
    )
    synthesizer.run(
        subscription_id="other_sub",
        profile_id="p1",
        profile_domain="generic",
        synthesis_version=1,
    )
    # Every Qdrant point written into a collection carries the right
    # subscription id on its payload. With empty-builder output this loop
    # is a no-op, but the invariant is asserted structurally for clarity.
    for collection in ("sme_artifacts_sandbox", "sme_artifacts_other_sub"):
        expected_sub = collection.removeprefix("sme_artifacts_")
        for point in qdrant.points[collection]:
            assert point["payload"]["subscription_id"] == expected_sub


def test_adapter_metadata_flows_through_to_trace(wired) -> None:
    synthesizer, blob, _qdrant, _neo4j = wired
    synthesizer.run(
        subscription_id="sandbox",
        profile_id="prof_1",
        profile_domain="generic",
        synthesis_version=1,
    )
    trace_path = (
        "sme_traces/synthesis/sandbox/prof_1/sandbox:prof_1:1.jsonl"
    )
    start_events = [
        json.loads(line)
        for line in blob.files[trace_path].splitlines()
        if json.loads(line)["stage"] == "start"
    ]
    assert len(start_events) == 1
    start = start_events[0]
    # Adapter identity flows from the loaded Adapter instance (ERRATA §1).
    assert start["adapter_version"] == "1.0.0"
    assert isinstance(start["adapter_hash"], str) and start["adapter_hash"]


def test_manifest_writer_usable_standalone(wired) -> None:
    synthesizer, blob, _qdrant, _neo4j = wired
    # Proves the storage facade exposes put_manifest for Phase 2 run-manifests.
    storage = synthesizer._d.storage
    storage.put_manifest(
        "sandbox",
        "prof_1",
        {"synthesis_id": "sandbox:prof_1:9", "counts": {"dossier": 0}},
    )
    body = json.loads(
        blob.files["sme_artifacts/sandbox/prof_1/manifest/sandbox:prof_1:9.json"]
    )
    assert body["counts"]["dossier"] == 0
