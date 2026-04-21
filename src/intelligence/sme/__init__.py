"""DocWain SME (Subject Matter Expert) synthesis package.

Ingestion-time synthesis of domain-aware artifacts, per spec
docs/superpowers/specs/2026-04-20-docwain-profile-sme-reasoning-design.md.

Phase 2 public surface (ERRATA §6): the synthesizer, storage facade,
verifier, trace writer, the five builders, the unified artifact item
contract, and the incremental-synthesis helpers. Feature-flag helpers
stay in ``src.config.feature_flags`` (ERRATA §4) — not re-exported here.
"""

from src.intelligence.sme.adapter_loader import AdapterLoader, get_adapter_loader
from src.intelligence.sme.adapter_schema import Adapter
from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef
from src.intelligence.sme.builders._base import ArtifactBuilder, BuilderContext
from src.intelligence.sme.builders.comparative_register import (
    ComparativeRegisterBuilder,
)
from src.intelligence.sme.builders.dossier import SMEDossierBuilder
from src.intelligence.sme.builders.insight_index import InsightIndexBuilder
from src.intelligence.sme.builders.kg_materializer import KGMultiHopMaterializer
from src.intelligence.sme.builders.recommendation_bank import (
    RecommendationBankBuilder,
)
from src.intelligence.sme.input_hash import (
    ChunkRef,
    InputHashInputs,
    compute_input_hash,
    compute_input_hash_for_profile,
    input_hash_unchanged,
)
from src.intelligence.sme.storage import (
    BlobStore,
    Neo4jBridge,
    QdrantBridge,
    SMEArtifactStorage,
    StorageDeps,
)
from src.intelligence.sme.synthesizer import SMESynthesizer, SynthesizerDeps
from src.intelligence.sme.trace import SynthesisTraceWriter
from src.intelligence.sme.verifier import SMEVerifier, Verdict, VerifierContext

__all__ = [
    # Adapter loader
    "AdapterLoader",
    "Adapter",
    "get_adapter_loader",
    # Artifact models
    "ArtifactItem",
    "EvidenceRef",
    # Builder base + five implementations
    "ArtifactBuilder",
    "BuilderContext",
    "SMEDossierBuilder",
    "InsightIndexBuilder",
    "ComparativeRegisterBuilder",
    "KGMultiHopMaterializer",
    "RecommendationBankBuilder",
    # Storage facade
    "SMEArtifactStorage",
    "StorageDeps",
    "BlobStore",
    "QdrantBridge",
    "Neo4jBridge",
    # Synthesizer
    "SMESynthesizer",
    "SynthesizerDeps",
    # Verifier + context
    "SMEVerifier",
    "VerifierContext",
    "Verdict",
    # Trace writer
    "SynthesisTraceWriter",
    # Incremental synthesis
    "ChunkRef",
    "InputHashInputs",
    "compute_input_hash",
    "compute_input_hash_for_profile",
    "input_hash_unchanged",
]
