# DocWain Intelligence Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DocWain a highly intelligent document platform by completing the V2 vision-grafted model training pipeline and fundamentally upgrading the document processing pipeline (extraction, KG, embeddings, visualization).

**Architecture:** Two pillars executed in parallel where possible. Pillar 1 completes the V2 6-phase training pipeline (Phase 1 exists, Phase 4 exists, implement 2/2.5/3/3.5). Pillar 2 upgrades extraction (4-engine ensemble + intelligent merger), KG (LLM-driven with ontology), embeddings (3-signal hybrid retrieval), and visualization (model-native insights + charts). Foundation work (GPU config, ontology) unblocks both pillars.

**Tech Stack:** Python 3.12, PyTorch, Unsloth, TRL (SFTTrainer/DPOTrainer), SigLIP, Qwen3-14B, SPLADE v3, BGE-large, Qdrant, Neo4j, Plotly, XGBoost, FastAPI, Celery

**Spec:** `docs/superpowers/specs/2026-04-01-docwain-intelligence-upgrade-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `src/utils/gpu.py` | GPU auto-detection, tiered config (A100/T4/CPU) |
| `src/utils/vram_manager.py` | Dynamic model loading/offloading, VRAM budgeting |
| `src/finetune/v2/train_phase2_5_dpo.py` | DPO contrastive preference training |
| `src/finetune/v2/train_phase3_5_insights.py` | Insight generation SFT |
| `src/extraction/triage.py` | Adaptive document triage (route to correct engines) |
| `src/extraction/preprocessor.py` | Image enhancement, page classification, language detection |
| `src/extraction/v2_extractor.py` | V2 model as fourth extraction engine |
| `src/extraction/validator.py` | Post-extraction validation + re-extraction loop |
| `src/kg/llm_entity_extractor.py` | LLM-driven entity + relationship extraction |
| `src/kg/entity_resolver.py` | Hierarchical entity resolution + alias merging |
| `src/kg/ontology.py` | Domain relationship type definitions |
| `src/kg/quality.py` | KG completeness + evidence scoring |
| `src/embedding/sparse.py` | SPLADE v3 sparse embeddings |
| `src/embedding/v2_embeddings.py` | V2 model hidden-state embeddings |
| `src/embedding/chunking/semantic_chunker.py` | V2-driven semantic boundary detection |
| `src/embedding/kg_enrichment.py` | Prepend KG context to chunk text before embedding |
| `src/embedding/feedback.py` | Retrieval quality tracking + hard negative mining |
| `src/retrieval/fusion.py` | Three-signal RRF fusion (BGE + SPLADE + V2) |
| `src/visualization/insights.py` | Insight categorization + visualization routing |
| `src/visualization/dashboard.py` | Multi-document dashboard response composition |

### Modified Files

| File | Change |
|---|---|
| `src/finetune/v2/train_phase2.py` | Replace stubs with Unsloth+TRL training loop, add CoT + curriculum |
| `src/finetune/v2/train_phase3.py` | Replace stubs with Unsloth+TRL training loop, add confidence + viz |
| `src/finetune/v2/pipeline.py` | Add phase2_5 and phase3_5 to pipeline phases + markers |
| `src/finetune/v2/merge_promote.py` | Update merge for 4 LoRA adapters, expand regression suite |
| `src/finetune/v2/dataset_preprocess.py` | Add CoT format, DPO format, insight format helpers |
| `src/finetune/agentic_orchestrator.py` | Retarget to V2 14B, enforce data policy filter |
| `src/extraction/engine.py` | Add V2 extractor + triage integration |
| `src/extraction/merger.py` | Rewrite with weighted agreement + conflict resolution |
| `src/extraction/models.py` | Add TriageResult, ValidationResult, QualityScorecard models |
| `src/kg/ingest.py` | Incremental enrichment, cross-doc inference |
| `src/kg/entity_extractor.py` | LLM-primary path with regex+spaCy validation |
| `src/embedding/pipeline/qdrant_ingestion.py` | Add sparse + V2 vector support |
| `src/embedding/orchestrator.py` | Integrate V2 embeddings + feedback loop |
| `src/visualization/enhancer.py` | Parse V2 `<viz>` directives, integrate insights |
| `src/retrieval/unified_retriever.py` | Integrate 3-signal fusion |

### Test Files

| File | Tests |
|---|---|
| `tests/unit/utils/test_gpu.py` | GPU detection, tiered configs, fallback |
| `tests/unit/utils/test_vram_manager.py` | Model loading/offloading, priority, budget |
| `tests/unit/finetune/v2/test_train_phase2.py` | Phase 2 training config, curriculum staging, CoT format |
| `tests/unit/finetune/v2/test_train_phase2_5_dpo.py` | DPO data construction, preference pairs |
| `tests/unit/finetune/v2/test_train_phase3.py` | Phase 3 tool-calling, confidence calibration |
| `tests/unit/finetune/v2/test_train_phase3_5_insights.py` | Insight categories, dataset construction |
| `tests/unit/finetune/v2/test_pipeline.py` | Pipeline phases including 2.5 and 3.5 |
| `tests/unit/finetune/v2/test_merge_promote.py` | 4-adapter merge sequence |
| `tests/unit/finetune/v2/test_dataset_preprocess.py` | CoT, DPO, insight format helpers |
| `tests/unit/extraction/test_triage.py` | Document type classification, engine routing |
| `tests/unit/extraction/test_preprocessor.py` | Image enhancement, page classification |
| `tests/unit/extraction/test_v2_extractor.py` | V2 extraction with think reasoning |
| `tests/unit/extraction/test_merger.py` | Weighted agreement, conflict resolution |
| `tests/unit/extraction/test_validator.py` | Self-consistency, schema validation, re-extraction |
| `tests/unit/kg/test_llm_entity_extractor.py` | LLM entity + relationship extraction |
| `tests/unit/kg/test_entity_resolver.py` | Alias resolution, cross-doc linking |
| `tests/unit/kg/test_ontology.py` | Domain schemas, relationship types |
| `tests/unit/kg/test_quality.py` | Completeness scoring, gap detection |
| `tests/unit/embedding/test_sparse.py` | SPLADE encoding, sparse vector format |
| `tests/unit/embedding/test_v2_embeddings.py` | V2 hidden-state extraction, projection |
| `tests/unit/embedding/test_semantic_chunker.py` | Semantic boundaries, hierarchical chunks |
| `tests/unit/embedding/test_kg_enrichment.py` | KG context prepending |
| `tests/unit/embedding/test_feedback.py` | Retrieval tracking, hard negative mining |
| `tests/unit/retrieval/test_fusion.py` | RRF fusion, weight tuning |
| `tests/unit/visualization/test_insights.py` | Insight routing, severity classification |
| `tests/unit/visualization/test_dashboard.py` | Multi-doc dashboard composition |

---

## Task Dependency Graph

```
Task 1 (GPU) ──────────────┬──> Task 4 (Phase 2) ──> Task 5 (Phase 2.5 DPO) ──> Task 6 (Phase 3) ──> Task 7 (Phase 3.5) ──> Task 9 (Merge) ──> Task 10 (Pipeline) ──> Task 28 (Daily Loop)
                            │
Task 2 (VRAM) ─────────────┤
                            │
Task 3 (Ontology) ─────────┼──> Task 15 (LLM Entity) ──> Task 16 (Resolver) ──> Task 17 (KG Quality) ──> Task 18 (KG Ingest)
                            │
Task 8 (Dataset Preprocess) ┘
                            
Task 11 (Triage) ──> Task 12 (Preprocessor) ──> Task 14 (V2 Extractor*) ──> Task 13 (Merger) ──> Task 25 (Validator)

Task 19 (Sparse) ──┐
Task 20 (V2 Embed*)├──> Task 23 (Fusion) ──> Task 24 (Feedback)
Task 21 (Chunker) ─┤
Task 22 (KG Enrich)┘

Task 26 (Insights) ──> Task 27 (Dashboard) ──> Task 29 (Enhancer Upgrade)

* Tasks 14 and 20 depend on V2 model being trained (Task 10), but the code/interfaces can be built beforehand.
```

**Parallel execution groups (no dependencies between groups):**
- Group A: Tasks 1, 2, 3, 8, 11, 19, 21, 26
- Group B: Tasks 4, 12, 15, 22, 27 (after Group A)
- Group C: Tasks 5, 13, 16, 20, 23, 29 (after Group B)
- Group D: Tasks 6, 14, 17, 24, 25 (after Group C)
- Group E: Tasks 7, 18, 28 (after Group D)
- Group F: Tasks 9, 10 (after Group E)

---

## Task 1: GPU Detection Module

**Files:**
- Create: `src/utils/gpu.py`
- Test: `tests/unit/utils/test_gpu.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/utils/test_gpu.py
import pytest
from unittest.mock import patch, MagicMock


def test_detect_gpu_returns_gpu_config():
    from src.utils.gpu import detect_gpu, GPUConfig
    config = detect_gpu()
    assert isinstance(config, GPUConfig)
    assert isinstance(config.available, bool)
    assert isinstance(config.name, str)
    assert isinstance(config.vram_mb, int)
    assert isinstance(config.is_high_memory, bool)
    assert isinstance(config.use_4bit_quantization, bool)
    assert isinstance(config.recommended_embedding_batch_size, int)
    assert isinstance(config.recommended_training_batch_size, int)
    assert isinstance(config.max_concurrent_models, int)


def test_a100_80gb_config():
    """A100 80GB should use full precision, no quantization."""
    from src.utils.gpu import _build_config_for_gpu
    config = _build_config_for_gpu(name="NVIDIA A100-SXM4-80GB", vram_mb=81920, cuda_version="12.4")
    assert config.is_high_memory is True
    assert config.use_4bit_quantization is False
    assert config.recommended_embedding_batch_size == 256
    assert config.recommended_training_batch_size == 4
    assert config.max_concurrent_models == 3


def test_t4_16gb_config():
    """T4 16GB should use aggressive quantization."""
    from src.utils.gpu import _build_config_for_gpu
    config = _build_config_for_gpu(name="Tesla T4", vram_mb=16384, cuda_version="12.0")
    assert config.is_high_memory is False
    assert config.use_4bit_quantization is True
    assert config.recommended_embedding_batch_size == 64
    assert config.recommended_training_batch_size == 1
    assert config.max_concurrent_models == 1


def test_a100_40gb_config():
    """A100 40GB should use selective quantization."""
    from src.utils.gpu import _build_config_for_gpu
    config = _build_config_for_gpu(name="NVIDIA A100-PCIE-40GB", vram_mb=40960, cuda_version="12.0")
    assert config.is_high_memory is True
    assert config.use_4bit_quantization is False
    assert config.recommended_embedding_batch_size == 128
    assert config.recommended_training_batch_size == 2
    assert config.max_concurrent_models == 2


def test_cpu_fallback():
    """When no GPU is available, return CPU config."""
    from src.utils.gpu import _build_cpu_config
    config = _build_cpu_config()
    assert config.available is False
    assert config.use_4bit_quantization is True
    assert config.recommended_embedding_batch_size == 16
    assert config.recommended_training_batch_size == 1
    assert config.max_concurrent_models == 1


def test_detect_gpu_no_cuda(monkeypatch):
    """When torch.cuda is unavailable, fallback to CPU config."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from src.utils.gpu import detect_gpu
    config = detect_gpu()
    assert config.available is False
    assert config.name == "CPU"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/utils/test_gpu.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.gpu'`

- [ ] **Step 3: Write implementation**

```python
# src/utils/gpu.py
"""GPU detection and hardware-adaptive configuration for DocWain."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUConfig:
    name: str
    vram_mb: int
    cuda_version: str
    is_high_memory: bool
    available: bool
    use_4bit_quantization: bool
    recommended_embedding_batch_size: int
    recommended_training_batch_size: int
    max_concurrent_models: int


def _get_cuda_version() -> str:
    try:
        import torch
        return torch.version.cuda or "unknown"
    except Exception:
        return "unknown"


def _build_config_for_gpu(
    name: str, vram_mb: int, cuda_version: Optional[str] = None
) -> GPUConfig:
    cuda_ver = cuda_version or _get_cuda_version()

    if vram_mb >= 70000:  # A100 80GB class
        return GPUConfig(
            name=name,
            vram_mb=vram_mb,
            cuda_version=cuda_ver,
            is_high_memory=True,
            available=True,
            use_4bit_quantization=False,
            recommended_embedding_batch_size=256,
            recommended_training_batch_size=4,
            max_concurrent_models=3,
        )
    elif vram_mb >= 35000:  # A100 40GB class
        return GPUConfig(
            name=name,
            vram_mb=vram_mb,
            cuda_version=cuda_ver,
            is_high_memory=True,
            available=True,
            use_4bit_quantization=False,
            recommended_embedding_batch_size=128,
            recommended_training_batch_size=2,
            max_concurrent_models=2,
        )
    elif vram_mb >= 20000:  # A10/V100 class
        return GPUConfig(
            name=name,
            vram_mb=vram_mb,
            cuda_version=cuda_ver,
            is_high_memory=False,
            available=True,
            use_4bit_quantization=True,
            recommended_embedding_batch_size=96,
            recommended_training_batch_size=1,
            max_concurrent_models=2,
        )
    else:  # T4 16GB or smaller
        return GPUConfig(
            name=name,
            vram_mb=vram_mb,
            cuda_version=cuda_ver,
            is_high_memory=False,
            available=True,
            use_4bit_quantization=True,
            recommended_embedding_batch_size=64,
            recommended_training_batch_size=1,
            max_concurrent_models=1,
        )


def _build_cpu_config() -> GPUConfig:
    return GPUConfig(
        name="CPU",
        vram_mb=0,
        cuda_version="none",
        is_high_memory=False,
        available=False,
        use_4bit_quantization=True,
        recommended_embedding_batch_size=16,
        recommended_training_batch_size=1,
        max_concurrent_models=1,
    )


def detect_gpu() -> GPUConfig:
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("No CUDA GPU available, using CPU config")
            return _build_cpu_config()

        props = torch.cuda.get_device_properties(0)
        name = props.name
        vram_mb = props.total_mem // (1024 * 1024)
        cuda_ver = torch.version.cuda or "unknown"

        config = _build_config_for_gpu(name, vram_mb, cuda_ver)
        logger.info(
            "GPU detected: %s (%d MB VRAM, CUDA %s, 4bit=%s, batch_embed=%d, batch_train=%d)",
            config.name,
            config.vram_mb,
            config.cuda_version,
            config.use_4bit_quantization,
            config.recommended_embedding_batch_size,
            config.recommended_training_batch_size,
        )
        return config

    except Exception as exc:
        logger.warning("GPU detection failed: %s — falling back to CPU config", exc)
        return _build_cpu_config()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/utils/test_gpu.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/gpu.py tests/unit/utils/test_gpu.py
git commit -m "feat: add GPU detection module with tiered hardware configs"
```

---

## Task 2: VRAM Memory Manager

**Files:**
- Create: `src/utils/vram_manager.py`
- Test: `tests/unit/utils/test_vram_manager.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/utils/test_vram_manager.py
import pytest
from unittest.mock import patch, MagicMock


def test_vram_manager_registers_model():
    from src.utils.vram_manager import VRAMManager
    mgr = VRAMManager(total_vram_mb=81920)
    mgr.register_model("v2", estimated_vram_mb=28000, priority=1)
    assert "v2" in mgr.registered_models
    assert mgr.registered_models["v2"]["estimated_vram_mb"] == 28000


def test_vram_manager_load_model():
    from src.utils.vram_manager import VRAMManager
    mgr = VRAMManager(total_vram_mb=81920)
    mgr.register_model("bge", estimated_vram_mb=2000, priority=2)
    loaded = mgr.request_load("bge")
    assert loaded is True
    assert "bge" in mgr.loaded_models


def test_vram_manager_rejects_over_budget():
    from src.utils.vram_manager import VRAMManager
    mgr = VRAMManager(total_vram_mb=1000, max_utilization=0.9)
    mgr.register_model("huge", estimated_vram_mb=1000, priority=1)
    loaded = mgr.request_load("huge")
    assert loaded is False


def test_vram_manager_evicts_low_priority():
    from src.utils.vram_manager import VRAMManager
    mgr = VRAMManager(total_vram_mb=5000, max_utilization=0.9)
    mgr.register_model("training", estimated_vram_mb=3000, priority=3)
    mgr.register_model("inference", estimated_vram_mb=3000, priority=1)
    mgr.request_load("training")
    assert "training" in mgr.loaded_models
    # inference (higher priority = lower number) should evict training
    loaded = mgr.request_load("inference")
    assert loaded is True
    assert "inference" in mgr.loaded_models
    assert "training" not in mgr.loaded_models


def test_vram_manager_mode_switch():
    from src.utils.vram_manager import VRAMManager, ExecutionMode
    mgr = VRAMManager(total_vram_mb=81920)
    mgr.register_model("v2", estimated_vram_mb=28000, priority=1)
    mgr.register_model("bge", estimated_vram_mb=2000, priority=2)
    mgr.register_model("splade", estimated_vram_mb=1500, priority=2)
    mgr.register_model("extraction", estimated_vram_mb=5000, priority=2)

    plan = mgr.get_mode_plan(ExecutionMode.QUERY_ANSWERING)
    assert "v2" in plan["load"]
    assert "bge" in plan["load"]
    assert "splade" in plan["load"]


def test_vram_manager_available_budget():
    from src.utils.vram_manager import VRAMManager
    mgr = VRAMManager(total_vram_mb=10000, max_utilization=0.9)
    assert mgr.available_vram_mb == 9000
    mgr.register_model("m1", estimated_vram_mb=3000, priority=1)
    mgr.request_load("m1")
    assert mgr.available_vram_mb == 6000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/utils/test_vram_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/utils/vram_manager.py
"""VRAM memory manager for dynamic model loading/offloading."""
from __future__ import annotations

import enum
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ExecutionMode(enum.Enum):
    DOCUMENT_PROCESSING = "document_processing"
    QUERY_ANSWERING = "query_answering"
    TRAINING = "training"


# Which model groups to load per execution mode
_MODE_MODELS: Dict[ExecutionMode, Set[str]] = {
    ExecutionMode.QUERY_ANSWERING: {"v2", "bge", "splade", "reranker"},
    ExecutionMode.DOCUMENT_PROCESSING: {"v2", "extraction", "bge"},
    ExecutionMode.TRAINING: {"v2"},
}


class VRAMManager:
    def __init__(
        self,
        total_vram_mb: int = 81920,
        max_utilization: float = 0.9,
    ):
        self._total_vram_mb = total_vram_mb
        self._max_utilization = max_utilization
        self._budget_mb = int(total_vram_mb * max_utilization)
        self._lock = threading.Lock()
        self.registered_models: Dict[str, Dict[str, Any]] = {}
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self._used_vram_mb = 0

    @property
    def available_vram_mb(self) -> int:
        return self._budget_mb - self._used_vram_mb

    def register_model(
        self,
        name: str,
        estimated_vram_mb: int,
        priority: int,
        load_fn: Any = None,
        unload_fn: Any = None,
    ) -> None:
        with self._lock:
            self.registered_models[name] = {
                "estimated_vram_mb": estimated_vram_mb,
                "priority": priority,
                "load_fn": load_fn,
                "unload_fn": unload_fn,
            }

    def request_load(self, name: str) -> bool:
        with self._lock:
            if name in self.loaded_models:
                return True

            if name not in self.registered_models:
                logger.warning("Model %s not registered", name)
                return False

            info = self.registered_models[name]
            needed = info["estimated_vram_mb"]

            # Try to fit without eviction
            if needed <= self.available_vram_mb:
                return self._do_load(name, info)

            # Evict lower-priority models to make room
            evictable = sorted(
                [
                    (n, m)
                    for n, m in self.loaded_models.items()
                    if self.registered_models[n]["priority"] > info["priority"]
                ],
                key=lambda x: self.registered_models[x[0]]["priority"],
                reverse=True,
            )

            freed = 0
            to_evict = []
            for evict_name, evict_info in evictable:
                to_evict.append(evict_name)
                freed += evict_info["estimated_vram_mb"]
                if self.available_vram_mb + freed >= needed:
                    break

            if self.available_vram_mb + freed < needed:
                logger.warning(
                    "Cannot load %s (%d MB): only %d MB available after eviction",
                    name, needed, self.available_vram_mb + freed,
                )
                return False

            for evict_name in to_evict:
                self._do_unload(evict_name)

            return self._do_load(name, info)

    def request_unload(self, name: str) -> bool:
        with self._lock:
            if name not in self.loaded_models:
                return True
            self._do_unload(name)
            return True

    def get_mode_plan(self, mode: ExecutionMode) -> Dict[str, List[str]]:
        desired = _MODE_MODELS.get(mode, set())
        registered_desired = desired & set(self.registered_models.keys())
        currently_loaded = set(self.loaded_models.keys())

        to_load = registered_desired - currently_loaded
        to_unload = currently_loaded - registered_desired

        return {
            "load": sorted(to_load),
            "unload": sorted(to_unload),
            "keep": sorted(currently_loaded & registered_desired),
        }

    def switch_mode(self, mode: ExecutionMode) -> Dict[str, List[str]]:
        plan = self.get_mode_plan(mode)
        for name in plan["unload"]:
            self.request_unload(name)
        for name in plan["load"]:
            self.request_load(name)
        logger.info("Switched to %s mode: %s", mode.value, plan)
        return plan

    def _do_load(self, name: str, info: Dict[str, Any]) -> bool:
        vram = info["estimated_vram_mb"]
        load_fn = info.get("load_fn")
        if load_fn:
            try:
                load_fn()
            except Exception as exc:
                logger.error("Failed to load %s: %s", name, exc)
                return False

        self.loaded_models[name] = {"estimated_vram_mb": vram}
        self._used_vram_mb += vram
        logger.info("Loaded %s (%d MB, %d MB remaining)", name, vram, self.available_vram_mb)
        return True

    def _do_unload(self, name: str) -> None:
        info = self.loaded_models.pop(name, None)
        if info:
            self._used_vram_mb -= info["estimated_vram_mb"]
            unload_fn = self.registered_models.get(name, {}).get("unload_fn")
            if unload_fn:
                try:
                    unload_fn()
                except Exception as exc:
                    logger.warning("Unload callback failed for %s: %s", name, exc)
            logger.info("Unloaded %s (%d MB freed)", name, info["estimated_vram_mb"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/utils/test_vram_manager.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/vram_manager.py tests/unit/utils/test_vram_manager.py
git commit -m "feat: add VRAM memory manager with priority-based eviction"
```

---

## Task 3: Domain Ontology

**Files:**
- Create: `src/kg/ontology.py`
- Test: `tests/unit/kg/test_ontology.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/kg/test_ontology.py
import pytest


def test_get_domain_returns_relationship_types():
    from src.kg.ontology import get_domain_relationships
    legal = get_domain_relationships("legal")
    assert "signatory_of" in legal
    assert "governed_by" in legal


def test_all_domains_exist():
    from src.kg.ontology import DOMAINS, get_domain_relationships
    expected = {"legal", "financial", "hr", "medical", "generic"}
    assert set(DOMAINS) == expected
    for domain in DOMAINS:
        rels = get_domain_relationships(domain)
        assert len(rels) > 0


def test_unknown_domain_returns_generic():
    from src.kg.ontology import get_domain_relationships
    rels = get_domain_relationships("unknown_domain")
    generic = get_domain_relationships("generic")
    assert rels == generic


def test_relationship_has_schema():
    from src.kg.ontology import get_relationship_schema
    schema = get_relationship_schema("signatory_of")
    assert schema["domain"] == "legal"
    assert "source_types" in schema
    assert "target_types" in schema


def test_all_relationship_types_unique():
    from src.kg.ontology import ALL_RELATIONSHIPS
    names = [r["name"] for r in ALL_RELATIONSHIPS]
    assert len(names) == len(set(names))


def test_detect_domain_from_entities():
    from src.kg.ontology import detect_domain
    entities = [
        {"type": "PERSON", "name": "John"},
        {"type": "CLAUSE", "name": "Section 5.2"},
    ]
    assert detect_domain(entities) == "legal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/kg/test_ontology.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/kg/ontology.py
"""Domain relationship ontology for DocWain knowledge graph."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

_LEGAL_RELATIONSHIPS = [
    {"name": "party_to", "source_types": ["PERSON", "ORGANIZATION"], "target_types": ["CONTRACT", "AGREEMENT"]},
    {"name": "signatory_of", "source_types": ["PERSON"], "target_types": ["CONTRACT", "AGREEMENT"]},
    {"name": "governed_by", "source_types": ["CONTRACT", "AGREEMENT"], "target_types": ["LAW", "REGULATION"]},
    {"name": "amends", "source_types": ["DOCUMENT"], "target_types": ["DOCUMENT"]},
    {"name": "supersedes", "source_types": ["DOCUMENT"], "target_types": ["DOCUMENT"]},
    {"name": "terminates", "source_types": ["DOCUMENT"], "target_types": ["CONTRACT", "AGREEMENT"]},
    {"name": "effective_from", "source_types": ["CONTRACT", "AGREEMENT"], "target_types": ["DATE"]},
    {"name": "expires_on", "source_types": ["CONTRACT", "AGREEMENT"], "target_types": ["DATE"]},
]

_FINANCIAL_RELATIONSHIPS = [
    {"name": "invoiced_by", "source_types": ["INVOICE"], "target_types": ["ORGANIZATION", "PERSON"]},
    {"name": "paid_to", "source_types": ["PAYMENT"], "target_types": ["ORGANIZATION", "PERSON"]},
    {"name": "line_item_of", "source_types": ["LINE_ITEM"], "target_types": ["INVOICE"]},
    {"name": "totals_to", "source_types": ["INVOICE", "LINE_ITEM"], "target_types": ["AMOUNT"]},
    {"name": "billed_on", "source_types": ["INVOICE"], "target_types": ["DATE"]},
    {"name": "due_on", "source_types": ["INVOICE"], "target_types": ["DATE"]},
]

_HR_RELATIONSHIPS = [
    {"name": "employed_by", "source_types": ["PERSON"], "target_types": ["ORGANIZATION"]},
    {"name": "reports_to", "source_types": ["PERSON"], "target_types": ["PERSON"]},
    {"name": "holds_certification", "source_types": ["PERSON"], "target_types": ["CERTIFICATION", "SKILL"]},
    {"name": "worked_during", "source_types": ["PERSON"], "target_types": ["DATE_RANGE"]},
    {"name": "role_of", "source_types": ["PERSON"], "target_types": ["ROLE"]},
]

_MEDICAL_RELATIONSHIPS = [
    {"name": "diagnosed_with", "source_types": ["PERSON"], "target_types": ["MEDICAL_TERM", "CONDITION"]},
    {"name": "prescribed", "source_types": ["PERSON"], "target_types": ["MEDICATION"]},
    {"name": "treated_by", "source_types": ["PERSON"], "target_types": ["PERSON", "ORGANIZATION"]},
    {"name": "allergic_to", "source_types": ["PERSON"], "target_types": ["SUBSTANCE", "MEDICATION"]},
    {"name": "admitted_on", "source_types": ["PERSON"], "target_types": ["DATE"]},
]

_GENERIC_RELATIONSHIPS = [
    {"name": "related_to", "source_types": ["ENTITY"], "target_types": ["ENTITY"]},
    {"name": "mentioned_in", "source_types": ["ENTITY"], "target_types": ["DOCUMENT"]},
    {"name": "part_of", "source_types": ["ENTITY"], "target_types": ["ENTITY"]},
    {"name": "located_at", "source_types": ["ENTITY", "ORGANIZATION"], "target_types": ["LOCATION"]},
]

_DOMAIN_MAP: Dict[str, List[Dict[str, Any]]] = {
    "legal": _LEGAL_RELATIONSHIPS,
    "financial": _FINANCIAL_RELATIONSHIPS,
    "hr": _HR_RELATIONSHIPS,
    "medical": _MEDICAL_RELATIONSHIPS,
    "generic": _GENERIC_RELATIONSHIPS,
}

DOMAINS: List[str] = list(_DOMAIN_MAP.keys())

ALL_RELATIONSHIPS: List[Dict[str, Any]] = []
_NAME_TO_SCHEMA: Dict[str, Dict[str, Any]] = {}

for _domain, _rels in _DOMAIN_MAP.items():
    for _rel in _rels:
        enriched = {**_rel, "domain": _domain}
        ALL_RELATIONSHIPS.append(enriched)
        _NAME_TO_SCHEMA[_rel["name"]] = enriched

# Entity types that hint at specific domains
_DOMAIN_HINTS: Dict[str, Set[str]] = {
    "legal": {"CLAUSE", "CONTRACT", "AGREEMENT", "LAW", "REGULATION"},
    "financial": {"INVOICE", "PAYMENT", "LINE_ITEM", "AMOUNT"},
    "hr": {"CERTIFICATION", "ROLE", "SKILL"},
    "medical": {"MEDICAL_TERM", "CONDITION", "MEDICATION", "SUBSTANCE"},
}


def get_domain_relationships(domain: str) -> List[str]:
    rels = _DOMAIN_MAP.get(domain, _DOMAIN_MAP["generic"])
    return [r["name"] for r in rels]


def get_relationship_schema(name: str) -> Optional[Dict[str, Any]]:
    return _NAME_TO_SCHEMA.get(name)


def detect_domain(entities: List[Dict[str, Any]]) -> str:
    entity_types = {e.get("type", "").upper() for e in entities}
    best_domain = "generic"
    best_overlap = 0
    for domain, hint_types in _DOMAIN_HINTS.items():
        overlap = len(entity_types & hint_types)
        if overlap > best_overlap:
            best_overlap = overlap
            best_domain = domain
    return best_domain
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/kg/test_ontology.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/kg/ontology.py tests/unit/kg/test_ontology.py
git commit -m "feat: add domain relationship ontology for KG typed relationships"
```

---

## Task 4: Phase 2 — Document Intelligence SFT Training Loop

**Files:**
- Modify: `src/finetune/v2/train_phase2.py:74-140`
- Modify: `src/finetune/v2/dataset_preprocess.py` (add CoT format)
- Test: `tests/unit/finetune/v2/test_train_phase2.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/finetune/v2/test_train_phase2.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_phase2_config_defaults():
    from src.finetune.v2.train_phase2 import Phase2Config
    cfg = Phase2Config()
    assert cfg.lora_r == 64
    assert cfg.lora_alpha == 128
    assert cfg.learning_rate == 2e-5
    assert cfg.epochs == 8
    assert cfg.dataset_mix == {"table": 0.40, "layout": 0.25, "ocr": 0.20, "cross_ref": 0.15}
    assert cfg.curriculum_stages == 4


def test_phase2_config_lora_targets():
    from src.finetune.v2.train_phase2 import Phase2Config
    cfg = Phase2Config()
    expected = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert cfg.lora_target_modules == expected


def test_phase2_curriculum_stage_epochs():
    from src.finetune.v2.train_phase2 import Phase2Config, _get_curriculum_epochs
    cfg = Phase2Config(epochs=8, curriculum_stages=4)
    stages = _get_curriculum_epochs(cfg)
    assert len(stages) == 4
    assert stages[0] == (1, 2)   # clean docs
    assert stages[1] == (3, 4)   # noisy scans
    assert stages[2] == (5, 6)   # complex layouts
    assert stages[3] == (7, 8)   # adversarial


def test_phase2_builds_sft_training_args():
    from src.finetune.v2.train_phase2 import Phase2Config, _build_training_args
    cfg = Phase2Config()
    args = _build_training_args(cfg, output_dir=Path("/tmp/test"))
    assert args["per_device_train_batch_size"] == 4
    assert args["gradient_accumulation_steps"] == 8
    assert args["learning_rate"] == 2e-5
    assert args["bf16"] is True
    assert args["max_seq_length"] == 4096


def test_cot_format_wraps_reasoning():
    from src.finetune.v2.dataset_preprocess import format_cot_sft
    result = format_cot_sft(
        question="What is the total?",
        reasoning="I see a table with 3 rows. Sum of column 3 is 1500.",
        answer="The total is $1,500.",
    )
    msgs = result["messages"]
    assistant_msg = msgs[-1]["content"]
    assert "<think>" in assistant_msg
    assert "</think>" in assistant_msg
    assert "The total is $1,500." in assistant_msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_train_phase2.py -v`
Expected: FAIL — `Phase2Config` has old defaults (lora_r=16, epochs=2)

- [ ] **Step 3: Update Phase2Config and add CoT format**

First, update `src/finetune/v2/dataset_preprocess.py` — add CoT format helper after the existing `format_no_tool_sft` function:

```python
# Add to src/finetune/v2/dataset_preprocess.py after format_no_tool_sft (after line 162)

def format_cot_sft(
    question: str,
    reasoning: str,
    answer: str,
    *,
    image_path: str | None = None,
    tools_json: str | None = None,
) -> dict[str, Any]:
    """Format a chain-of-thought training example with <think> block."""
    msgs: list[dict[str, Any]] = [_system_message(tools_json)]

    user_content = question
    if image_path:
        user_content = f"<image>{image_path}</image>\n{question}"
    msgs.append({"role": "user", "content": user_content})

    assistant_content = f"<think>\n{reasoning}\n</think>\n\n{answer}"
    msgs.append({"role": "assistant", "content": assistant_content})

    return {"messages": msgs}


def format_dpo_pair(
    question: str,
    chosen_reasoning: str,
    chosen_answer: str,
    rejected_reasoning: str,
    rejected_answer: str,
    *,
    image_path: str | None = None,
    tools_json: str | None = None,
) -> dict[str, Any]:
    """Format a DPO preference pair with chosen and rejected responses."""
    msgs_prompt: list[dict[str, Any]] = [_system_message(tools_json)]

    user_content = question
    if image_path:
        user_content = f"<image>{image_path}</image>\n{question}"
    msgs_prompt.append({"role": "user", "content": user_content})

    chosen = f"<think>\n{chosen_reasoning}\n</think>\n\n{chosen_answer}"
    rejected = f"<think>\n{rejected_reasoning}\n</think>\n\n{rejected_answer}"

    return {
        "prompt": msgs_prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def format_insight_sft(
    question: str,
    reasoning: str,
    insight_category: str,
    insight_text: str,
    answer: str,
    *,
    viz_directive: str | None = None,
    tools_json: str | None = None,
) -> dict[str, Any]:
    """Format an insight generation training example."""
    msgs: list[dict[str, Any]] = [_system_message(tools_json)]
    msgs.append({"role": "user", "content": question})

    parts = [f"<think>\n{reasoning}\n</think>"]
    parts.append(f"\n**Insight ({insight_category}):** {insight_text}")
    if viz_directive:
        parts.append(f"\n{viz_directive}")
    parts.append(f"\n{answer}")

    msgs.append({"role": "assistant", "content": "\n".join(parts)})
    return {"messages": msgs}
```

- [ ] **Step 4: Update Phase2Config and implement training loop**

Replace the Phase2Config and run_phase2 in `src/finetune/v2/train_phase2.py`:

```python
# src/finetune/v2/train_phase2.py — full replacement of config and run function

# Update Phase2Config (replace lines 26-67)
@dataclass
class Phase2Config:
    """Configuration for Phase 2: Document Intelligence SFT."""
    base_model: str = "unsloth/Qwen3-14B"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.10
    epochs: int = 8
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    bf16: bool = True
    curriculum_stages: int = 4
    dataset_mix: dict = field(default_factory=lambda: {
        "table": 0.40, "layout": 0.25, "ocr": 0.20, "cross_ref": 0.15,
    })
    checkpoint_steps: int = 500
    output_dir: Path = field(default_factory=lambda: Path("runs/v2/phase2"))
    # QA gates
    gate_docvqa_accuracy: float = 0.75
    gate_table_f1: float = 0.80
    gate_layout_map: float = 0.70


def _get_curriculum_epochs(config: Phase2Config) -> list[tuple[int, int]]:
    """Return (start_epoch, end_epoch) pairs for each curriculum stage."""
    epochs_per_stage = config.epochs // config.curriculum_stages
    stages = []
    for i in range(config.curriculum_stages):
        start = i * epochs_per_stage + 1
        end = (i + 1) * epochs_per_stage
        stages.append((start, end))
    return stages


def _build_training_args(config: Phase2Config, output_dir: Path) -> dict:
    """Build SFTTrainer-compatible training arguments."""
    return {
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "num_train_epochs": config.epochs,
        "bf16": config.bf16,
        "max_seq_length": config.max_seq_length,
        "output_dir": str(output_dir),
        "save_steps": config.checkpoint_steps,
        "logging_steps": 50,
        "save_total_limit": 4,
        "report_to": "none",
    }


# Replace run_phase2 (lines 74-140)
def run_phase2(
    config: Optional[Phase2Config] = None,
    *,
    phase1_checkpoint: Optional[Path] = None,
) -> Path:
    """Execute Phase 2: Document Intelligence SFT with CoT + curriculum learning."""
    config = config or Phase2Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Phase 2: Document Intelligence SFT — output=%s", config.output_dir)

    # 1. Load vision-grafted model from Phase 1
    from src.finetune.v2.vision_graft import VisionGraftedModel, GraftConfig

    graft_cfg = GraftConfig(text_model=config.base_model)
    model = VisionGraftedModel(graft_cfg)
    model.load_vision_encoder()
    model.load_projection(checkpoint=phase1_checkpoint / "projection.pt" if phase1_checkpoint else None)
    model.load_text_model()

    # 2. Apply LoRA with Phase 2 config
    model.add_lora(r=config.lora_r, lora_alpha=config.lora_alpha)

    # 3. Unfreeze projection (trainable in Phase 2)
    if model._projection is not None:
        for p in model._projection.parameters():
            p.requires_grad = True

    # 4. Build curriculum-staged datasets
    from src.finetune.v2.dataset_preprocess import format_cot_sft
    curriculum_labels = ["clean", "noisy", "complex", "adversarial"]
    curriculum_stages = _get_curriculum_epochs(config)

    # 5. Training loop — one SFTTrainer per curriculum stage
    from trl import SFTTrainer, SFTConfig

    for stage_idx, (start_epoch, end_epoch) in enumerate(curriculum_stages):
        stage_name = curriculum_labels[stage_idx] if stage_idx < len(curriculum_labels) else f"stage_{stage_idx}"
        stage_epochs = end_epoch - start_epoch + 1
        logger.info("Curriculum stage %d/%d: %s (epochs %d-%d)",
                     stage_idx + 1, len(curriculum_stages), stage_name, start_epoch, end_epoch)

        # Load stage-specific dataset (JSONL with CoT format)
        dataset_path = config.output_dir / f"dataset_{stage_name}.jsonl"
        if not dataset_path.exists():
            logger.warning("Dataset not found: %s — skipping stage", dataset_path)
            continue

        from datasets import load_dataset
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")

        training_args = _build_training_args(config, config.output_dir / stage_name)
        training_args["num_train_epochs"] = stage_epochs

        sft_config = SFTConfig(**training_args)

        trainer = SFTTrainer(
            model=model._text_model,
            tokenizer=model._tokenizer,
            train_dataset=dataset,
            args=sft_config,
        )

        trainer.train()

        # Save stage checkpoint
        stage_checkpoint = config.output_dir / f"checkpoint_{stage_name}"
        model.save_all(stage_checkpoint)
        logger.info("Stage %s checkpoint saved to %s", stage_name, stage_checkpoint)

    # 6. Save final Phase 2 output
    model.save_all(config.output_dir)
    marker = config.output_dir / ".phase2_complete"
    marker.touch()
    logger.info("Phase 2 complete: %s", config.output_dir)
    return config.output_dir
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_train_phase2.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/finetune/v2/train_phase2.py src/finetune/v2/dataset_preprocess.py tests/unit/finetune/v2/test_train_phase2.py
git commit -m "feat: implement Phase 2 SFT training loop with CoT and curriculum learning"
```

---

## Task 5: Phase 2.5 — DPO Contrastive Preference Training

**Files:**
- Create: `src/finetune/v2/train_phase2_5_dpo.py`
- Test: `tests/unit/finetune/v2/test_train_phase2_5_dpo.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/finetune/v2/test_train_phase2_5_dpo.py
import pytest
from pathlib import Path


def test_dpo_config_defaults():
    from src.finetune.v2.train_phase2_5_dpo import DPOPhaseConfig
    cfg = DPOPhaseConfig()
    assert cfg.beta == 0.1
    assert cfg.learning_rate == 5e-6
    assert cfg.epochs == 3
    assert cfg.per_device_batch_size == 2
    assert cfg.gradient_accumulation_steps == 16
    assert cfg.max_prompt_length == 2048
    assert cfg.max_response_length == 2048
    assert cfg.gate_hallucination_rate == 0.05
    assert cfg.gate_extraction_f1_improvement == 0.05


def test_dpo_config_effective_batch():
    from src.finetune.v2.train_phase2_5_dpo import DPOPhaseConfig
    cfg = DPOPhaseConfig()
    assert cfg.per_device_batch_size * cfg.gradient_accumulation_steps == 32


def test_build_dpo_training_args():
    from src.finetune.v2.train_phase2_5_dpo import DPOPhaseConfig, _build_dpo_training_args
    cfg = DPOPhaseConfig()
    args = _build_dpo_training_args(cfg, output_dir=Path("/tmp/dpo"))
    assert args["beta"] == 0.1
    assert args["learning_rate"] == 5e-6
    assert args["bf16"] is True


def test_corrupt_extraction_produces_errors():
    from src.finetune.v2.train_phase2_5_dpo import corrupt_extraction
    good = {
        "entities": [{"name": "Acme Corp", "type": "ORGANIZATION", "confidence": 0.95}],
        "tables": [{"headers": ["Item", "Qty"], "data": [["Widget", "100"]]}],
        "fields": {"total": "$1,500", "date": "2025-03-15"},
    }
    bad = corrupt_extraction(good, seed=42)
    # Corrupted version should differ from original
    assert bad != good
    # Should still have the same structure
    assert "entities" in bad
    assert "tables" in bad
    assert "fields" in bad
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_train_phase2_5_dpo.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/finetune/v2/train_phase2_5_dpo.py
"""Phase 2.5: DPO contrastive preference training for extraction quality."""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DPOPhaseConfig:
    """Configuration for Phase 2.5: DPO preference training."""
    beta: float = 0.1
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.10
    epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_prompt_length: int = 2048
    max_response_length: int = 2048
    bf16: bool = True
    output_dir: Path = field(default_factory=lambda: Path("runs/v2/phase2_5_dpo"))
    # QA gates
    gate_hallucination_rate: float = 0.05
    gate_extraction_f1_improvement: float = 0.05


def _build_dpo_training_args(config: DPOPhaseConfig, output_dir: Path) -> dict:
    return {
        "beta": config.beta,
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "num_train_epochs": config.epochs,
        "bf16": config.bf16,
        "max_prompt_length": config.max_prompt_length,
        "max_length": config.max_prompt_length + config.max_response_length,
        "output_dir": str(output_dir),
        "save_steps": 200,
        "logging_steps": 25,
        "report_to": "none",
    }


def corrupt_extraction(
    good: Dict[str, Any], *, seed: int = 0
) -> Dict[str, Any]:
    """Programmatically corrupt a good extraction to produce a realistic bad example."""
    rng = random.Random(seed)
    bad = copy.deepcopy(good)

    corruption_types = ["drop_entity", "hallucinate_value", "break_table", "wrong_field"]
    chosen = rng.sample(corruption_types, k=min(2, len(corruption_types)))

    for corruption in chosen:
        if corruption == "drop_entity" and bad.get("entities"):
            idx = rng.randrange(len(bad["entities"]))
            bad["entities"].pop(idx)

        elif corruption == "hallucinate_value" and bad.get("fields"):
            keys = list(bad["fields"].keys())
            if keys:
                key = rng.choice(keys)
                bad["fields"][key] = f"HALLUCINATED_{rng.randint(1000, 9999)}"

        elif corruption == "break_table" and bad.get("tables"):
            table = rng.choice(bad["tables"])
            if table.get("data") and len(table["data"]) > 0:
                row_idx = rng.randrange(len(table["data"]))
                if table["data"][row_idx]:
                    col_idx = rng.randrange(len(table["data"][row_idx]))
                    table["data"][row_idx][col_idx] = ""

        elif corruption == "wrong_field" and bad.get("fields"):
            keys = list(bad["fields"].keys())
            if len(keys) >= 2:
                k1, k2 = rng.sample(keys, 2)
                bad["fields"][k1], bad["fields"][k2] = bad["fields"][k2], bad["fields"][k1]

    return bad


def run_phase2_5(
    config: Optional[DPOPhaseConfig] = None,
    *,
    phase2_dir: Optional[Path] = None,
) -> Path:
    """Execute Phase 2.5: DPO contrastive preference training."""
    config = config or DPOPhaseConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Phase 2.5: DPO — output=%s", config.output_dir)

    # 1. Load model from Phase 2 checkpoint
    from src.finetune.v2.vision_graft import VisionGraftedModel, GraftConfig

    graft_cfg = GraftConfig()
    model = VisionGraftedModel(graft_cfg)
    model.load_vision_encoder()
    model.load_projection(checkpoint=phase2_dir / "projection.pt" if phase2_dir else None)
    model.load_text_model()
    model.add_lora()

    # Load Phase 2 LoRA weights
    if phase2_dir:
        from peft import PeftModel
        model._text_model = PeftModel.from_pretrained(
            model._text_model, str(phase2_dir / "lora_adapter")
        )

    # 2. Load DPO dataset
    dataset_path = config.output_dir / "dpo_pairs.jsonl"
    if not dataset_path.exists():
        logger.warning("DPO dataset not found: %s", dataset_path)
        marker = config.output_dir / ".phase2_5_complete"
        marker.touch()
        return config.output_dir

    from datasets import load_dataset
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # 3. Train with DPOTrainer
    from trl import DPOTrainer, DPOConfig

    dpo_args = _build_dpo_training_args(config, config.output_dir)
    dpo_config = DPOConfig(**dpo_args)

    trainer = DPOTrainer(
        model=model._text_model,
        tokenizer=model._tokenizer,
        train_dataset=dataset,
        args=dpo_config,
    )

    trainer.train()

    # 4. Save
    model.save_all(config.output_dir)
    marker = config.output_dir / ".phase2_5_complete"
    marker.touch()
    logger.info("Phase 2.5 DPO complete: %s", config.output_dir)
    return config.output_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_train_phase2_5_dpo.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/finetune/v2/train_phase2_5_dpo.py tests/unit/finetune/v2/test_train_phase2_5_dpo.py
git commit -m "feat: add Phase 2.5 DPO contrastive preference training"
```

---

## Task 6: Phase 3 — Tool-Calling SFT Training Loop

**Files:**
- Modify: `src/finetune/v2/train_phase3.py:28-162`
- Test: `tests/unit/finetune/v2/test_train_phase3.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/finetune/v2/test_train_phase3.py
import pytest
from pathlib import Path


def test_phase3_config_defaults():
    from src.finetune.v2.train_phase3 import Phase3Config
    cfg = Phase3Config()
    assert cfg.lora_r == 64
    assert cfg.lora_alpha == 128
    assert cfg.learning_rate == 1e-5
    assert cfg.epochs == 5
    assert cfg.max_seq_length == 4096
    assert cfg.gate_tool_accuracy == 0.85
    assert cfg.gate_arg_correctness == 0.90
    assert cfg.gate_false_positive_rate == 0.10


def test_phase3_config_source_weights():
    from src.finetune.v2.train_phase3 import Phase3Config
    cfg = Phase3Config()
    assert cfg.source_weights["synthetic"] == 0.40
    assert cfg.source_weights["toolbench"] == 0.25
    assert cfg.source_weights["gorilla"] == 0.20
    assert cfg.source_weights["nexusraven"] == 0.15
    assert abs(sum(cfg.source_weights.values()) - 1.0) < 1e-6


def test_phase3_freezes_projection():
    from src.finetune.v2.train_phase3 import Phase3Config
    cfg = Phase3Config()
    assert cfg.freeze_projection is True


def test_phase3_training_args():
    from src.finetune.v2.train_phase3 import Phase3Config, _build_training_args
    cfg = Phase3Config()
    args = _build_training_args(cfg, Path("/tmp/phase3"))
    assert args["per_device_train_batch_size"] == 4
    assert args["gradient_accumulation_steps"] == 8
    assert args["learning_rate"] == 1e-5
    assert args["bf16"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_train_phase3.py -v`
Expected: FAIL — old config has different defaults

- [ ] **Step 3: Update Phase3Config and implement training loop**

Replace `src/finetune/v2/train_phase3.py` config and run function with updated values (lora_r=64, lora_alpha=128, learning_rate=1e-5, epochs=5, max_seq_length=4096, freeze_projection=True) and implement the SFTTrainer training loop following the same pattern as Phase 2 but with frozen projection and tool-calling dataset from `build_tool_calling_dataset()`.

The training loop:
1. Load model from Phase 2.5 checkpoint
2. Apply LoRA (r=64, alpha=128)
3. Freeze projection
4. Load tool-calling dataset (synthetic + external benchmarks)
5. Train with SFTTrainer
6. Save checkpoint with `.phase3_complete` marker

Add `_build_training_args` function matching the test expectations.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_train_phase3.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/finetune/v2/train_phase3.py tests/unit/finetune/v2/test_train_phase3.py
git commit -m "feat: implement Phase 3 tool-calling SFT with confidence calibration"
```

---

## Task 7: Phase 3.5 — Insight Generation SFT

**Files:**
- Create: `src/finetune/v2/train_phase3_5_insights.py`
- Test: `tests/unit/finetune/v2/test_train_phase3_5_insights.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/finetune/v2/test_train_phase3_5_insights.py
import pytest
from pathlib import Path


def test_insight_config_defaults():
    from src.finetune.v2.train_phase3_5_insights import InsightPhaseConfig
    cfg = InsightPhaseConfig()
    assert cfg.learning_rate == 1e-5
    assert cfg.epochs == 4
    assert cfg.gate_insight_precision == 0.80
    assert cfg.gate_insight_recall == 0.60


def test_insight_categories():
    from src.finetune.v2.train_phase3_5_insights import INSIGHT_CATEGORIES
    expected = {"pattern_recognition", "anomaly_detection", "trend_analysis",
                "comparative_analysis", "gap_analysis"}
    assert set(INSIGHT_CATEGORIES) == expected


def test_insight_format_helper():
    from src.finetune.v2.dataset_preprocess import format_insight_sft
    result = format_insight_sft(
        question="Summarize vendor performance",
        reasoning="Comparing delivery dates across 12 invoices shows consistent delays.",
        insight_category="pattern_recognition",
        insight_text="Vendor X delivers 5-7 days late consistently across all 12 invoices",
        answer="Based on analysis of 12 invoices...",
    )
    msgs = result["messages"]
    assistant_msg = msgs[-1]["content"]
    assert "<think>" in assistant_msg
    assert "pattern_recognition" in assistant_msg
    assert "Vendor X delivers" in assistant_msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_train_phase3_5_insights.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/finetune/v2/train_phase3_5_insights.py
"""Phase 3.5: Insight generation SFT training."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

INSIGHT_CATEGORIES = [
    "pattern_recognition",
    "anomaly_detection",
    "trend_analysis",
    "comparative_analysis",
    "gap_analysis",
]


@dataclass
class InsightPhaseConfig:
    """Configuration for Phase 3.5: Insight generation SFT."""
    lora_r: int = 64
    lora_alpha: int = 128
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.10
    epochs: int = 4
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    bf16: bool = True
    output_dir: Path = field(default_factory=lambda: Path("runs/v2/phase3_5_insights"))
    # QA gates
    gate_insight_precision: float = 0.80
    gate_insight_recall: float = 0.60


def _build_training_args(config: InsightPhaseConfig, output_dir: Path) -> dict:
    return {
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "num_train_epochs": config.epochs,
        "bf16": config.bf16,
        "max_seq_length": config.max_seq_length,
        "output_dir": str(output_dir),
        "save_steps": 300,
        "logging_steps": 25,
        "report_to": "none",
    }


def run_phase3_5(
    config: Optional[InsightPhaseConfig] = None,
    *,
    phase3_dir: Optional[Path] = None,
) -> Path:
    """Execute Phase 3.5: Insight generation SFT."""
    config = config or InsightPhaseConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Phase 3.5: Insight Generation SFT — output=%s", config.output_dir)

    # 1. Load model from Phase 3 checkpoint
    from src.finetune.v2.vision_graft import VisionGraftedModel, GraftConfig

    graft_cfg = GraftConfig()
    model = VisionGraftedModel(graft_cfg)
    model.load_vision_encoder()
    model.load_projection(checkpoint=phase3_dir / "projection.pt" if phase3_dir else None)
    model.load_text_model()
    model.add_lora(r=config.lora_r, lora_alpha=config.lora_alpha)

    # Load Phase 3 LoRA weights
    if phase3_dir:
        from peft import PeftModel
        model._text_model = PeftModel.from_pretrained(
            model._text_model, str(phase3_dir / "lora_adapter")
        )

    # Freeze projection (same as Phase 3)
    if model._projection is not None:
        for p in model._projection.parameters():
            p.requires_grad = False

    # 2. Load insight dataset
    dataset_path = config.output_dir / "insight_training.jsonl"
    if not dataset_path.exists():
        logger.warning("Insight dataset not found: %s", dataset_path)
        marker = config.output_dir / ".phase3_5_complete"
        marker.touch()
        return config.output_dir

    from datasets import load_dataset
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # 3. Train with SFTTrainer
    from trl import SFTTrainer, SFTConfig

    training_args = _build_training_args(config, config.output_dir)
    sft_config = SFTConfig(**training_args)

    trainer = SFTTrainer(
        model=model._text_model,
        tokenizer=model._tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    trainer.train()

    # 4. Save
    model.save_all(config.output_dir)
    marker = config.output_dir / ".phase3_5_complete"
    marker.touch()
    logger.info("Phase 3.5 complete: %s", config.output_dir)
    return config.output_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_train_phase3_5_insights.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/finetune/v2/train_phase3_5_insights.py tests/unit/finetune/v2/test_train_phase3_5_insights.py
git commit -m "feat: add Phase 3.5 insight generation SFT training"
```

---

## Task 8: Update Dataset Preprocess with New Format Helpers

**Files:**
- Modify: `src/finetune/v2/dataset_preprocess.py`
- Test: `tests/unit/finetune/v2/test_dataset_preprocess.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/finetune/v2/test_dataset_preprocess.py
import pytest


def test_format_cot_sft_structure():
    from src.finetune.v2.dataset_preprocess import format_cot_sft
    result = format_cot_sft(
        question="Extract the total from this invoice",
        reasoning="The table has columns Item, Qty, Price, Total. Last row shows grand total of $5,000.",
        answer="The invoice total is $5,000.",
    )
    assert "messages" in result
    msgs = result["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert "<think>" in msgs[2]["content"]
    assert "</think>" in msgs[2]["content"]
    assert "$5,000" in msgs[2]["content"]


def test_format_cot_sft_with_image():
    from src.finetune.v2.dataset_preprocess import format_cot_sft
    result = format_cot_sft(
        question="What is this document?",
        reasoning="I see a header saying 'Invoice' and a table.",
        answer="This is an invoice.",
        image_path="/tmp/doc.png",
    )
    user_msg = result["messages"][1]["content"]
    assert "<image>" in user_msg


def test_format_dpo_pair_structure():
    from src.finetune.v2.dataset_preprocess import format_dpo_pair
    result = format_dpo_pair(
        question="Extract entities",
        chosen_reasoning="I see John Smith (PERSON) and Acme Corp (ORG).",
        chosen_answer="Entities: John Smith, Acme Corp",
        rejected_reasoning="I see some names.",
        rejected_answer="Entities: John, Corp",
    )
    assert "prompt" in result
    assert "chosen" in result
    assert "rejected" in result
    assert "<think>" in result["chosen"]
    assert "<think>" in result["rejected"]


def test_format_insight_sft_structure():
    from src.finetune.v2.dataset_preprocess import format_insight_sft
    result = format_insight_sft(
        question="Analyze vendor performance",
        reasoning="Comparing 12 invoices shows late deliveries.",
        insight_category="pattern_recognition",
        insight_text="Vendor delivers 5-7 days late consistently",
        answer="Analysis shows...",
        viz_directive='<viz type="line" title="Delivery Delays" />',
    )
    msgs = result["messages"]
    assistant = msgs[-1]["content"]
    assert "pattern_recognition" in assistant
    assert "<viz" in assistant
    assert "<think>" in assistant
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_dataset_preprocess.py -v`
Expected: FAIL — new format functions don't exist yet

- [ ] **Step 3: Implementation was already added in Task 4 Step 3**

The `format_cot_sft`, `format_dpo_pair`, and `format_insight_sft` functions were added to `dataset_preprocess.py` in Task 4. Verify they're in place.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_dataset_preprocess.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/finetune/v2/test_dataset_preprocess.py
git commit -m "test: add tests for CoT, DPO, and insight format helpers"
```

---

## Task 9: Update Pipeline Orchestrator + Merge & Promote

**Files:**
- Modify: `src/finetune/v2/pipeline.py:12-71`
- Modify: `src/finetune/v2/merge_promote.py`
- Test: `tests/unit/finetune/v2/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/finetune/v2/test_pipeline.py
import pytest


def test_pipeline_has_six_phases():
    from src.finetune.v2.pipeline import V2Pipeline
    p = V2Pipeline()
    assert p.phases == ["phase1", "phase2", "phase2_5", "phase3", "phase3_5", "phase4"]


def test_phase_markers_include_new_phases():
    from src.finetune.v2.pipeline import PHASE_MARKERS
    assert "phase2_5" in PHASE_MARKERS
    assert "phase3_5" in PHASE_MARKERS
    assert PHASE_MARKERS["phase2_5"] == ".phase2_5_complete"
    assert PHASE_MARKERS["phase3_5"] == ".phase3_5_complete"


def test_pipeline_status_includes_new_phases(tmp_path):
    from src.finetune.v2.pipeline import V2Pipeline
    p = V2Pipeline(base_dir=tmp_path)
    status = p.status()
    assert status["next_phase"] == "phase1"
    # Mark phases 1-2 complete
    (tmp_path / ".phase1_complete").touch()
    (tmp_path / ".phase2_complete").touch()
    status = p.status()
    assert status["next_phase"] == "phase2_5"


def test_merge_promote_regression_includes_new_criteria():
    from src.finetune.v2.merge_promote import get_new_capability_criteria
    criteria = get_new_capability_criteria()
    assert "insight_precision" in criteria
    assert "confidence_calibration_ece" in criteria
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_pipeline.py -v`
Expected: FAIL — pipeline has 4 phases, not 6

- [ ] **Step 3: Update pipeline.py**

In `src/finetune/v2/pipeline.py`:
- Update `PHASE_MARKERS` dict to add `"phase2_5": ".phase2_5_complete"` and `"phase3_5": ".phase3_5_complete"`
- Update `V2Pipeline.phases` list to `["phase1", "phase2", "phase2_5", "phase3", "phase3_5", "phase4"]`

- [ ] **Step 4: Update merge_promote.py**

In `src/finetune/v2/merge_promote.py`:
- Add `"insight_precision": 0.80` and `"confidence_calibration_ece": 0.10` to `get_new_capability_criteria()`
- Update `run_phase4` merge sequence comment to document merging 4 LoRA adapters (Phase 2 + 2.5 + 3 + 3.5)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/v2/test_pipeline.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/finetune/v2/pipeline.py src/finetune/v2/merge_promote.py tests/unit/finetune/v2/test_pipeline.py
git commit -m "feat: update pipeline to 6 phases, expand merge regression criteria"
```

---

## Task 10: Update Extraction Models

**Files:**
- Modify: `src/extraction/models.py`
- Test: `tests/unit/extraction/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/extraction/test_models.py
import pytest


def test_triage_result_fields():
    from src.extraction.models import TriageResult
    tr = TriageResult(
        document_type="scanned",
        engine_weights={"structural": 0.3, "semantic": 0.5, "vision": 0.9, "v2": 0.8},
        preprocessing_directives=["deskew", "denoise"],
        page_types={"1": "body", "2": "table"},
        confidence=0.85,
    )
    assert tr.document_type == "scanned"
    assert tr.engine_weights["v2"] == 0.8
    assert "deskew" in tr.preprocessing_directives


def test_validation_result_fields():
    from src.extraction.models import ValidationResult
    vr = ValidationResult(
        passed=False,
        failed_checks=["total_mismatch"],
        field_confidences={"total": 0.4, "date": 0.95},
        retry_recommended=True,
    )
    assert vr.passed is False
    assert "total_mismatch" in vr.failed_checks


def test_quality_scorecard_fields():
    from src.extraction.models import QualityScorecard
    qs = QualityScorecard(
        overall_confidence=0.82,
        engine_contributions={"structural": 0.3, "v2": 0.7},
        conflict_count=2,
        conflict_log=["date field: structural=2025-01, v2=2025-01-15, resolved=v2"],
    )
    assert qs.overall_confidence == 0.82
    assert qs.conflict_count == 2


def test_extraction_result_existing_fields():
    from src.extraction.models import ExtractionResult
    er = ExtractionResult(
        document_id="doc1",
        subscription_id="sub1",
        profile_id="prof1",
        clean_text="hello",
    )
    assert er.document_id == "doc1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_models.py -v`
Expected: FAIL — `TriageResult`, `ValidationResult`, `QualityScorecard` don't exist

- [ ] **Step 3: Add new dataclasses to models.py**

Add after the existing `ExtractionResult` class in `src/extraction/models.py`:

```python
@dataclass
class TriageResult:
    """Result from adaptive document triage."""
    document_type: str  # "clean_digital", "scanned", "handwritten", "mixed", "table_heavy"
    engine_weights: Dict[str, float]  # engine_name -> weight (0.0-1.0)
    preprocessing_directives: List[str]  # ["deskew", "denoise", "upscale"]
    page_types: Dict[str, str]  # page_number -> type ("body", "table", "form", etc.)
    confidence: float = 0.0


@dataclass
class ValidationResult:
    """Result from post-extraction validation."""
    passed: bool
    failed_checks: List[str]  # ["total_mismatch", "date_chronology"]
    field_confidences: Dict[str, float]  # field_name -> confidence
    retry_recommended: bool = False


@dataclass
class QualityScorecard:
    """Extraction quality scorecard per document."""
    overall_confidence: float
    engine_contributions: Dict[str, float]  # engine_name -> contribution weight
    conflict_count: int = 0
    conflict_log: List[str] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_models.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/extraction/models.py tests/unit/extraction/test_models.py
git commit -m "feat: add TriageResult, ValidationResult, QualityScorecard models"
```

---

## Task 11: Adaptive Document Triage

**Files:**
- Create: `src/extraction/triage.py`
- Test: `tests/unit/extraction/test_triage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/extraction/test_triage.py
import pytest


def test_triage_clean_digital_pdf():
    from src.extraction.triage import DocumentTriager
    triager = DocumentTriager()
    result = triager.triage(
        file_type="pdf",
        has_text_layer=True,
        dpi=300,
        noise_score=0.05,
        page_count=5,
    )
    assert result.document_type == "clean_digital"
    assert result.engine_weights["vision"] < result.engine_weights["structural"]
    assert "deskew" not in result.preprocessing_directives


def test_triage_scanned_document():
    from src.extraction.triage import DocumentTriager
    triager = DocumentTriager()
    result = triager.triage(
        file_type="pdf",
        has_text_layer=False,
        dpi=150,
        noise_score=0.4,
        page_count=3,
    )
    assert result.document_type == "scanned"
    assert result.engine_weights["vision"] >= 0.8
    assert "denoise" in result.preprocessing_directives


def test_triage_low_dpi_triggers_upscale():
    from src.extraction.triage import DocumentTriager
    triager = DocumentTriager()
    result = triager.triage(
        file_type="png",
        has_text_layer=False,
        dpi=100,
        noise_score=0.1,
        page_count=1,
    )
    assert "upscale" in result.preprocessing_directives


def test_triage_table_heavy():
    from src.extraction.triage import DocumentTriager
    triager = DocumentTriager()
    result = triager.triage(
        file_type="xlsx",
        has_text_layer=True,
        dpi=300,
        noise_score=0.0,
        page_count=2,
    )
    assert result.document_type == "table_heavy"
    assert result.engine_weights["v2"] >= 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_triage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/extraction/triage.py
"""Adaptive document triage — routes documents to optimal extraction engines."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.extraction.models import TriageResult

logger = logging.getLogger(__name__)

_TABLE_EXTENSIONS = {"xlsx", "xls", "csv", "tsv"}
_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp"}

# Default engine weight profiles per document type
_WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "clean_digital": {"structural": 0.9, "semantic": 0.8, "vision": 0.3, "v2": 0.7},
    "scanned":       {"structural": 0.3, "semantic": 0.5, "vision": 0.9, "v2": 0.8},
    "handwritten":   {"structural": 0.2, "semantic": 0.3, "vision": 0.7, "v2": 0.9},
    "mixed":         {"structural": 0.6, "semantic": 0.7, "vision": 0.7, "v2": 0.8},
    "table_heavy":   {"structural": 0.8, "semantic": 0.6, "vision": 0.5, "v2": 0.9},
}


class DocumentTriager:
    def triage(
        self,
        file_type: str,
        has_text_layer: bool = True,
        dpi: int = 300,
        noise_score: float = 0.0,
        page_count: int = 1,
        *,
        page_images: Optional[List[bytes]] = None,
    ) -> TriageResult:
        ext = file_type.lower().lstrip(".")

        # Classify document type
        doc_type = self._classify(ext, has_text_layer, dpi, noise_score)

        # Determine preprocessing
        directives = self._preprocessing_directives(dpi, noise_score, has_text_layer, doc_type)

        # Get engine weights
        weights = dict(_WEIGHT_PROFILES.get(doc_type, _WEIGHT_PROFILES["mixed"]))

        # Build page types (default all to "body" — real impl would analyze per-page)
        page_types = {str(i + 1): "body" for i in range(page_count)}

        confidence = 0.9 if doc_type in ("clean_digital", "table_heavy") else 0.7

        result = TriageResult(
            document_type=doc_type,
            engine_weights=weights,
            preprocessing_directives=directives,
            page_types=page_types,
            confidence=confidence,
        )
        logger.info("Triage: type=%s, directives=%s, weights=%s", doc_type, directives, weights)
        return result

    def _classify(
        self, ext: str, has_text_layer: bool, dpi: int, noise_score: float
    ) -> str:
        if ext in _TABLE_EXTENSIONS:
            return "table_heavy"

        if not has_text_layer:
            if noise_score > 0.5:
                return "handwritten"
            return "scanned"

        if ext in _IMAGE_EXTENSIONS:
            return "scanned"

        if has_text_layer and noise_score < 0.15:
            return "clean_digital"

        return "mixed"

    def _preprocessing_directives(
        self, dpi: int, noise_score: float, has_text_layer: bool, doc_type: str
    ) -> List[str]:
        directives = []

        if dpi < 150:
            directives.append("upscale")

        if noise_score > 0.2:
            directives.append("denoise")

        if doc_type in ("scanned", "handwritten"):
            directives.append("deskew")

        if noise_score > 0.3 and not has_text_layer:
            directives.append("contrast")

        return directives
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_triage.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/extraction/triage.py tests/unit/extraction/test_triage.py
git commit -m "feat: add adaptive document triage with engine weight routing"
```

---

## Task 12: Pre-Processing Intelligence

**Files:**
- Create: `src/extraction/preprocessor.py`
- Test: `tests/unit/extraction/test_preprocessor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/extraction/test_preprocessor.py
import pytest
from unittest.mock import MagicMock
import numpy as np


def test_preprocessor_skips_when_no_directives():
    from src.extraction.preprocessor import DocumentPreprocessor
    pp = DocumentPreprocessor()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pp.preprocess(img, directives=[])
    assert result.shape == img.shape


def test_preprocessor_deskew():
    from src.extraction.preprocessor import DocumentPreprocessor
    pp = DocumentPreprocessor()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    result = pp.preprocess(img, directives=["deskew"])
    assert result is not None
    assert result.shape[0] > 0 and result.shape[1] > 0


def test_preprocessor_denoise():
    from src.extraction.preprocessor import DocumentPreprocessor
    pp = DocumentPreprocessor()
    # Create noisy image
    rng = np.random.RandomState(42)
    img = rng.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    result = pp.preprocess(img, directives=["denoise"])
    # Denoised image should be smoother (lower variance)
    assert result.var() <= img.var()


def test_preprocessor_contrast():
    from src.extraction.preprocessor import DocumentPreprocessor
    pp = DocumentPreprocessor()
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = pp.preprocess(img, directives=["contrast"])
    assert result is not None


def test_detect_language_english():
    from src.extraction.preprocessor import detect_language
    lang = detect_language("This is a simple English sentence about contracts.")
    assert lang == "en"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_preprocessor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/extraction/preprocessor.py
"""Pre-processing intelligence for document images."""
from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect primary language of text. Returns ISO 639-1 code."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


class DocumentPreprocessor:
    def preprocess(
        self,
        image: np.ndarray,
        directives: List[str],
    ) -> np.ndarray:
        if not directives:
            return image

        result = image.copy()

        for directive in directives:
            if directive == "deskew":
                result = self._deskew(result)
            elif directive == "denoise":
                result = self._denoise(result)
            elif directive == "contrast":
                result = self._enhance_contrast(result)
            elif directive == "upscale":
                result = self._upscale(result)
            else:
                logger.warning("Unknown preprocessing directive: %s", directive)

        return result

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) < 10:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, 9, 75, 75)
        return cv2.bilateralFilter(image, 9, 75, 75)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(l_channel)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def _upscale(self, image: np.ndarray, factor: int = 2) -> np.ndarray:
        h, w = image.shape[:2]
        return cv2.resize(image, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_preprocessor.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/extraction/preprocessor.py tests/unit/extraction/test_preprocessor.py
git commit -m "feat: add document preprocessing with deskew, denoise, contrast, upscale"
```

---

## Task 13: Intelligent Extraction Merger (Rewrite)

**Files:**
- Modify: `src/extraction/merger.py` (full rewrite)
- Test: `tests/unit/extraction/test_merger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/extraction/test_merger.py
import pytest


def test_weighted_merge_uses_triage_weights():
    from src.extraction.merger import ExtractionMerger
    from src.extraction.models import TriageResult
    merger = ExtractionMerger()
    triage = TriageResult(
        document_type="scanned",
        engine_weights={"structural": 0.3, "semantic": 0.5, "vision": 0.9, "v2": 0.8},
        preprocessing_directives=[],
        page_types={},
        confidence=0.8,
    )
    result = merger.merge(
        document_id="doc1",
        subscription_id="sub1",
        profile_id="prof1",
        structural={"entities": [{"text": "John", "type": "PERSON", "confidence": 0.7}]},
        semantic={"entities": [{"text": "John Smith", "type": "PERSON", "confidence": 0.8}]},
        vision={"entities": [{"text": "John Smith", "type": "PERSON", "confidence": 0.9}]},
        v2={"entities": [{"text": "John Smith", "type": "PERSON", "confidence": 0.95}]},
        triage=triage,
    )
    # V2 + vision should dominate for scanned docs
    john = [e for e in result.entities if "john" in e.text.lower()]
    assert len(john) >= 1
    assert john[0].confidence > 0.8


def test_conflict_logged_in_scorecard():
    from src.extraction.merger import ExtractionMerger
    from src.extraction.models import TriageResult
    merger = ExtractionMerger()
    triage = TriageResult(
        document_type="clean_digital",
        engine_weights={"structural": 0.9, "semantic": 0.8, "vision": 0.3, "v2": 0.7},
        preprocessing_directives=[],
        page_types={},
    )
    result = merger.merge(
        document_id="doc1",
        subscription_id="sub1",
        profile_id="prof1",
        structural={"entities": [{"text": "2025-01", "type": "DATE", "confidence": 0.9}]},
        semantic={"entities": []},
        vision={"entities": [{"text": "2025-01-15", "type": "DATE", "confidence": 0.8}]},
        v2={"entities": [{"text": "2025-01-15", "type": "DATE", "confidence": 0.95}]},
        triage=triage,
    )
    assert result.metadata.get("quality_scorecard") is not None


def test_merge_without_v2_still_works():
    from src.extraction.merger import ExtractionMerger
    merger = ExtractionMerger()
    result = merger.merge(
        document_id="doc1",
        subscription_id="sub1",
        profile_id="prof1",
        structural={"entities": [{"text": "Test", "type": "ORG", "confidence": 0.8}]},
        semantic={"entities": []},
        vision={"entities": []},
    )
    assert len(result.entities) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_merger.py -v`
Expected: FAIL — merger.merge() doesn't accept `v2` or `triage` params

- [ ] **Step 3: Rewrite merger.py**

Rewrite `src/extraction/merger.py` with:
- Updated `merge()` signature accepting optional `v2: dict` and `triage: TriageResult` params
- Weighted confidence calculation using triage engine weights when available
- Entity deduplication using fuzzy matching (rapidfuzz) with weighted confidence
- Conflict detection and logging into `QualityScorecard`
- Backward compatibility when `v2` and `triage` are not provided

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_merger.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/extraction/merger.py tests/unit/extraction/test_merger.py
git commit -m "feat: rewrite extraction merger with weighted agreement and conflict resolution"
```

---

## Task 14: V2 Extraction Engine

**Files:**
- Create: `src/extraction/v2_extractor.py`
- Test: `tests/unit/extraction/test_v2_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/extraction/test_v2_extractor.py
import pytest
from unittest.mock import patch, MagicMock


def test_v2_extractor_returns_structured_output():
    from src.extraction.v2_extractor import V2Extractor
    extractor = V2Extractor(ollama_host="http://localhost:11434", model="docwain:v2")
    # Mock the Ollama call
    with patch("src.extraction.v2_extractor._call_v2_model") as mock_call:
        mock_call.return_value = {
            "think": "I see a table with 3 columns on page 1.",
            "entities": [{"text": "Acme Corp", "type": "ORGANIZATION", "confidence": 0.95}],
            "tables": [],
            "fields": {"date": "2025-01-15"},
            "confidence": 0.92,
        }
        result = extractor.extract(document_bytes=b"fake", file_type="pdf", page_images=[])
        assert "entities" in result
        assert result["entities"][0]["text"] == "Acme Corp"
        assert result["confidence"] == 0.92


def test_v2_extractor_includes_think_reasoning():
    from src.extraction.v2_extractor import V2Extractor
    extractor = V2Extractor()
    with patch("src.extraction.v2_extractor._call_v2_model") as mock_call:
        mock_call.return_value = {
            "think": "Analyzing document structure...",
            "entities": [],
            "tables": [],
            "fields": {},
            "confidence": 0.5,
        }
        result = extractor.extract(document_bytes=b"fake", file_type="pdf")
        assert "think" in result


def test_v2_extractor_prompt_includes_doc_type():
    from src.extraction.v2_extractor import _build_extraction_prompt
    prompt = _build_extraction_prompt(doc_type="scanned", page_type="table")
    assert "scanned" in prompt.lower()
    assert "table" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_v2_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/extraction/v2_extractor.py
"""V2 model as fourth extraction engine — vision + reasoning."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _build_extraction_prompt(
    doc_type: str = "unknown",
    page_type: str = "body",
) -> str:
    return (
        f"You are extracting structured information from a {doc_type} document. "
        f"This page is classified as: {page_type}. "
        "First reason about what you see in <think> tags, then extract:\n"
        "1. All entities (name, type, confidence)\n"
        "2. All tables (headers, rows)\n"
        "3. Key-value fields (dates, amounts, IDs)\n"
        "4. Flag any inconsistencies (e.g., totals that don't add up)\n"
        "Respond in JSON with keys: think, entities, tables, fields, confidence."
    )


def _call_v2_model(
    prompt: str,
    images: Optional[List[bytes]] = None,
    *,
    ollama_host: str = "http://localhost:11434",
    model: str = "docwain:v2",
    timeout: int = 120,
) -> Dict[str, Any]:
    """Call V2 model via Ollama for extraction."""
    try:
        import ollama
        client = ollama.Client(host=ollama_host)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": 8192, "temperature": 0.1},
        )
        content = response["message"]["content"]
        # Parse JSON from response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
        return {"think": content, "entities": [], "tables": [], "fields": {}, "confidence": 0.3}
    except Exception as exc:
        logger.error("V2 extraction failed: %s", exc)
        return {"think": f"Error: {exc}", "entities": [], "tables": [], "fields": {}, "confidence": 0.0}


class V2Extractor:
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "docwain:v2",
    ):
        self.ollama_host = ollama_host
        self.model = model

    def extract(
        self,
        document_bytes: bytes,
        file_type: str,
        page_images: Optional[List[bytes]] = None,
        *,
        doc_type: str = "unknown",
        page_type: str = "body",
        text_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = _build_extraction_prompt(doc_type=doc_type, page_type=page_type)

        if text_content:
            prompt += f"\n\nAvailable text content:\n{text_content[:4000]}"

        result = _call_v2_model(
            prompt,
            images=page_images,
            ollama_host=self.ollama_host,
            model=self.model,
        )

        logger.info(
            "V2 extraction: entities=%d, tables=%d, confidence=%.2f",
            len(result.get("entities", [])),
            len(result.get("tables", [])),
            result.get("confidence", 0),
        )
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_v2_extractor.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/extraction/v2_extractor.py tests/unit/extraction/test_v2_extractor.py
git commit -m "feat: add V2 model as fourth extraction engine with think reasoning"
```

---

## Task 15: LLM-Driven Entity & Relationship Extraction

**Files:**
- Create: `src/kg/llm_entity_extractor.py`
- Test: `tests/unit/kg/test_llm_entity_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/kg/test_llm_entity_extractor.py
import pytest
from unittest.mock import patch


def test_extract_entities_and_relationships():
    from src.kg.llm_entity_extractor import LLMEntityExtractor
    extractor = LLMEntityExtractor()
    with patch("src.kg.llm_entity_extractor._call_llm") as mock:
        mock.return_value = {
            "entities": [
                {"name": "John Smith", "type": "PERSON", "aliases": ["J. Smith"], "confidence": 0.95},
                {"name": "Acme Corp", "type": "ORGANIZATION", "aliases": ["Acme"], "confidence": 0.90},
            ],
            "relationships": [
                {"source": "John Smith", "target": "Acme Corp", "type": "signatory_of",
                 "evidence": "signed by John Smith on behalf of Acme Corporation",
                 "confidence": 0.92},
            ],
        }
        result = extractor.extract("John Smith signed the contract for Acme Corp.")
        assert len(result["entities"]) == 2
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["type"] == "signatory_of"


def test_validate_against_regex_spacy():
    from src.kg.llm_entity_extractor import LLMEntityExtractor
    extractor = LLMEntityExtractor()
    llm_entities = [
        {"name": "John Smith", "type": "PERSON", "confidence": 0.95},
        {"name": "mystery_value", "type": "PERSON", "confidence": 0.6},
    ]
    validated = extractor.validate_entities(llm_entities, "John Smith works at Acme.")
    # John Smith should be validated (cross-check with spaCy/regex)
    assert any(e["name"] == "John Smith" for e in validated)


def test_extract_with_domain_hint():
    from src.kg.llm_entity_extractor import LLMEntityExtractor
    extractor = LLMEntityExtractor()
    with patch("src.kg.llm_entity_extractor._call_llm") as mock:
        mock.return_value = {"entities": [], "relationships": []}
        extractor.extract("Some text", domain="legal")
        call_args = mock.call_args[0][0]
        assert "legal" in call_args.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/kg/test_llm_entity_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/kg/llm_entity_extractor.py
"""LLM-driven entity and relationship extraction for knowledge graph."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.kg.ontology import get_domain_relationships, detect_domain

logger = logging.getLogger(__name__)


def _build_extraction_prompt(text: str, domain: str = "generic") -> str:
    rel_types = get_domain_relationships(domain)
    return (
        f"Extract all entities and relationships from this {domain} document text.\n\n"
        f"Text: {text[:3000]}\n\n"
        "For each entity, provide: name, type (PERSON, ORGANIZATION, LOCATION, DATE, AMOUNT, "
        "CLAUSE, SKILL, MEDICAL_TERM, etc.), aliases (alternative names), confidence (0.0-1.0).\n\n"
        f"For relationships, use these types when applicable: {', '.join(rel_types)}\n"
        "Each relationship needs: source, target, type, evidence (quote from text), confidence, "
        "and optional temporal_bounds.\n\n"
        "Respond in JSON: {\"entities\": [...], \"relationships\": [...]}"
    )


def _call_llm(
    prompt: str,
    *,
    ollama_host: str = "http://localhost:11434",
    model: str = "docwain:v2",
) -> Dict[str, Any]:
    try:
        import ollama
        client = ollama.Client(host=ollama_host)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": 8192, "temperature": 0.1},
        )
        content = response["message"]["content"]
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
        return {"entities": [], "relationships": []}
    except Exception as exc:
        logger.error("LLM entity extraction failed: %s", exc)
        return {"entities": [], "relationships": []}


class LLMEntityExtractor:
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "docwain:v2",
    ):
        self.ollama_host = ollama_host
        self.model = model

    def extract(
        self,
        text: str,
        domain: str = "generic",
    ) -> Dict[str, Any]:
        prompt = _build_extraction_prompt(text, domain)
        result = _call_llm(prompt, ollama_host=self.ollama_host, model=self.model)

        entities = result.get("entities", [])
        relationships = result.get("relationships", [])

        logger.info("LLM extracted %d entities, %d relationships", len(entities), len(relationships))
        return {"entities": entities, "relationships": relationships}

    def validate_entities(
        self,
        llm_entities: List[Dict[str, Any]],
        original_text: str,
    ) -> List[Dict[str, Any]]:
        """Cross-validate LLM entities against regex + spaCy."""
        from src.kg.entity_extractor import EntityExtractor

        regex_extractor = EntityExtractor()
        regex_entities = regex_extractor.extract_with_metadata(original_text)
        regex_names = {e.normalized_name for e in regex_entities}

        validated = []
        for entity in llm_entities:
            name = entity.get("name", "")
            normalized = name.lower().strip()

            # Entity is validated if regex/spaCy also found it
            if normalized in regex_names:
                entity["cross_validated"] = True
                validated.append(entity)
            elif entity.get("confidence", 0) >= 0.8:
                # High-confidence LLM entities pass without cross-validation
                entity["cross_validated"] = False
                validated.append(entity)
            else:
                logger.debug("Low-confidence entity rejected: %s (%.2f)", name, entity.get("confidence", 0))

        return validated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/kg/test_llm_entity_extractor.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/kg/llm_entity_extractor.py tests/unit/kg/test_llm_entity_extractor.py
git commit -m "feat: add LLM-driven entity and relationship extraction for KG"
```

---

## Task 16: Hierarchical Entity Resolution

**Files:**
- Create: `src/kg/entity_resolver.py`
- Test: `tests/unit/kg/test_entity_resolver.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/kg/test_entity_resolver.py
import pytest


def test_resolve_aliases():
    from src.kg.entity_resolver import EntityResolver
    resolver = EntityResolver()
    entities = [
        {"name": "John Smith", "type": "PERSON", "confidence": 0.95, "doc_id": "doc1"},
        {"name": "J. Smith", "type": "PERSON", "confidence": 0.80, "doc_id": "doc2"},
        {"name": "Mr. Smith", "type": "PERSON", "confidence": 0.70, "doc_id": "doc1"},
    ]
    resolved = resolver.resolve(entities)
    # All three should merge into one canonical entity
    assert len(resolved) == 1
    assert resolved[0]["canonical_name"] == "John Smith"  # highest confidence
    assert set(resolved[0]["aliases"]) >= {"J. Smith", "Mr. Smith"}


def test_different_types_not_merged():
    from src.kg.entity_resolver import EntityResolver
    resolver = EntityResolver()
    entities = [
        {"name": "Smith", "type": "PERSON", "confidence": 0.8, "doc_id": "doc1"},
        {"name": "Smith", "type": "ORGANIZATION", "confidence": 0.8, "doc_id": "doc1"},
    ]
    resolved = resolver.resolve(entities)
    assert len(resolved) == 2


def test_cross_document_linking():
    from src.kg.entity_resolver import EntityResolver
    resolver = EntityResolver()
    entities = [
        {"name": "Acme Corporation", "type": "ORGANIZATION", "confidence": 0.95, "doc_id": "doc1"},
        {"name": "Acme Corp", "type": "ORGANIZATION", "confidence": 0.90, "doc_id": "doc2"},
        {"name": "ACME", "type": "ORGANIZATION", "confidence": 0.80, "doc_id": "doc3"},
    ]
    resolved = resolver.resolve(entities)
    assert len(resolved) == 1
    assert len(resolved[0]["doc_ids"]) == 3


def test_confidence_propagation():
    from src.kg.entity_resolver import EntityResolver
    resolver = EntityResolver()
    entities = [
        {"name": "Acme Corp", "type": "ORGANIZATION", "confidence": 0.95, "doc_id": "doc1"},
        {"name": "ACME", "type": "ORGANIZATION", "confidence": 0.50, "doc_id": "doc2"},
    ]
    resolved = resolver.resolve(entities)
    # Low-confidence mention should be boosted by high-confidence match
    assert resolved[0]["confidence"] > 0.50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/kg/test_entity_resolver.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/kg/entity_resolver.py
"""Hierarchical entity resolution with alias merging and cross-doc linking."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

_FUZZY_THRESHOLD = 75  # rapidfuzz partial_ratio threshold


class EntityResolver:
    def __init__(self, fuzzy_threshold: float = _FUZZY_THRESHOLD):
        self._threshold = fuzzy_threshold

    def resolve(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve entities into canonical groups with aliases."""
        groups: List[Dict[str, Any]] = []

        for entity in entities:
            name = entity.get("name", "")
            etype = entity.get("type", "")
            confidence = entity.get("confidence", 0.0)
            doc_id = entity.get("doc_id", "")

            matched_group = self._find_matching_group(groups, name, etype)

            if matched_group is not None:
                self._merge_into_group(matched_group, name, confidence, doc_id)
            else:
                groups.append({
                    "canonical_name": name,
                    "type": etype,
                    "aliases": set(),
                    "doc_ids": {doc_id} if doc_id else set(),
                    "confidence": confidence,
                    "mention_count": 1,
                    "max_confidence": confidence,
                })

        # Convert sets to lists for serialization
        return [
            {
                **g,
                "aliases": sorted(g["aliases"]),
                "doc_ids": sorted(g["doc_ids"]),
            }
            for g in groups
        ]

    def _find_matching_group(
        self, groups: List[Dict[str, Any]], name: str, etype: str
    ) -> Dict[str, Any] | None:
        name_lower = name.lower().strip()

        for group in groups:
            if group["type"] != etype:
                continue

            canonical_lower = group["canonical_name"].lower().strip()

            # Exact match
            if name_lower == canonical_lower:
                return group

            # Fuzzy match against canonical name
            if fuzz.partial_ratio(name_lower, canonical_lower) >= self._threshold:
                return group

            # Fuzzy match against aliases
            for alias in group["aliases"]:
                if fuzz.partial_ratio(name_lower, alias.lower()) >= self._threshold:
                    return group

        return None

    def _merge_into_group(
        self, group: Dict[str, Any], name: str, confidence: float, doc_id: str
    ) -> None:
        # Add as alias if different from canonical
        if name.lower().strip() != group["canonical_name"].lower().strip():
            group["aliases"].add(name)

        # Promote to canonical if higher confidence
        if confidence > group["max_confidence"]:
            old_canonical = group["canonical_name"]
            group["canonical_name"] = name
            group["aliases"].add(old_canonical)
            group["aliases"].discard(name)
            group["max_confidence"] = confidence

        # Propagate confidence (weighted average)
        count = group["mention_count"]
        group["confidence"] = (group["confidence"] * count + confidence) / (count + 1)
        group["mention_count"] = count + 1

        if doc_id:
            group["doc_ids"].add(doc_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/kg/test_entity_resolver.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/kg/entity_resolver.py tests/unit/kg/test_entity_resolver.py
git commit -m "feat: add hierarchical entity resolution with alias merging"
```

---

## Task 17: KG Quality & Completeness Scoring

**Files:**
- Create: `src/kg/quality.py`
- Test: `tests/unit/kg/test_quality.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/kg/test_quality.py
import pytest


def test_entity_completeness_name_only():
    from src.kg.quality import score_entity_completeness
    score = score_entity_completeness({"name": "John", "type": "PERSON"})
    assert score == pytest.approx(0.2, abs=0.05)


def test_entity_completeness_full_profile():
    from src.kg.quality import score_entity_completeness
    score = score_entity_completeness({
        "name": "John Smith",
        "type": "PERSON",
        "aliases": ["J. Smith"],
        "relationships": [{"type": "employed_by", "target": "Acme"}],
        "doc_ids": ["doc1", "doc2"],
        "temporal_bounds": {"start": "2024-01"},
    })
    assert score >= 0.8


def test_relationship_evidence_score():
    from src.kg.quality import score_relationship_evidence
    # Single document mention
    assert score_relationship_evidence(doc_count=1) < 0.5
    # Multiple corroborating documents
    assert score_relationship_evidence(doc_count=3) >= 0.7
    assert score_relationship_evidence(doc_count=5) >= 0.9


def test_detect_gaps():
    from src.kg.quality import detect_gaps
    entities = [
        {"name": "John Smith", "type": "PERSON", "relationships": [
            {"type": "role_of", "target": "CEO"},
        ]},
        {"name": "Acme Corp", "type": "ORGANIZATION", "relationships": []},
    ]
    gaps = detect_gaps(entities)
    # Should flag: Person has role but no employer, Org has no employees/contracts
    assert len(gaps) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/kg/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/kg/quality.py
"""KG quality and completeness scoring."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Attributes that contribute to entity completeness
_COMPLETENESS_WEIGHTS = {
    "name": 0.20,
    "type": 0.00,  # Always present, no bonus
    "aliases": 0.15,
    "relationships": 0.25,
    "doc_ids": 0.15,
    "temporal_bounds": 0.15,
    "confidence": 0.10,
}

# Expected relationships per entity type
_EXPECTED_RELATIONSHIPS = {
    "PERSON": {"employed_by", "reports_to", "role_of", "signatory_of"},
    "ORGANIZATION": {"employs", "party_to", "located_at"},
}


def score_entity_completeness(entity: Dict[str, Any]) -> float:
    score = 0.0
    for attr, weight in _COMPLETENESS_WEIGHTS.items():
        value = entity.get(attr)
        if value is not None:
            if isinstance(value, (list, set, dict)):
                if len(value) > 0:
                    score += weight
            elif isinstance(value, str):
                if value.strip():
                    score += weight
            else:
                score += weight
    return min(score, 1.0)


def score_relationship_evidence(doc_count: int) -> float:
    if doc_count <= 0:
        return 0.0
    if doc_count == 1:
        return 0.3
    if doc_count == 2:
        return 0.6
    if doc_count <= 4:
        return 0.7 + (doc_count - 3) * 0.1
    return min(0.9 + (doc_count - 5) * 0.02, 1.0)


def detect_gaps(entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    gaps = []
    for entity in entities:
        etype = entity.get("type", "")
        name = entity.get("name", "unknown")
        relationships = entity.get("relationships", [])
        rel_types = {r.get("type", "") for r in relationships}
        expected = _EXPECTED_RELATIONSHIPS.get(etype, set())

        if expected and not rel_types & expected:
            gaps.append({
                "entity": name,
                "type": etype,
                "gap": f"No expected relationships found. Expected at least one of: {', '.join(sorted(expected))}",
            })

        if etype == "PERSON" and "role_of" in rel_types and "employed_by" not in rel_types:
            gaps.append({
                "entity": name,
                "type": etype,
                "gap": "Has role but no employer relationship",
            })

    return gaps
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/kg/test_quality.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/kg/quality.py tests/unit/kg/test_quality.py
git commit -m "feat: add KG quality scoring with completeness and gap detection"
```

---

## Task 18: Update KG Ingest for Incremental Enrichment

**Files:**
- Modify: `src/kg/ingest.py`
- Test: `tests/unit/kg/test_ingest_enrichment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/kg/test_ingest_enrichment.py
import pytest
from unittest.mock import MagicMock


def test_build_graph_payload_with_typed_relationships():
    from src.kg.ingest import build_graph_payload
    payload = build_graph_payload(
        embeddings_payload={"chunks": [{"text": "test", "metadata": {}}]},
        subscription_id="sub1",
        profile_id="prof1",
        document_id="doc1",
        doc_name="test.pdf",
        typed_relationships=[
            {"source": "John Smith", "target": "Acme Corp", "type": "signatory_of",
             "confidence": 0.92},
        ],
    )
    assert payload is not None
    assert len(payload.typed_relationships) >= 1
    assert payload.typed_relationships[0]["type"] == "signatory_of"


def test_cross_doc_inference_trigger():
    from src.kg.ingest import should_run_cross_doc_inference
    # After 10 document ingestions, should trigger
    assert should_run_cross_doc_inference(doc_count=10) is True
    assert should_run_cross_doc_inference(doc_count=5) is False
    assert should_run_cross_doc_inference(doc_count=20) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/kg/test_ingest_enrichment.py -v`
Expected: FAIL — `should_run_cross_doc_inference` doesn't exist

- [ ] **Step 3: Add to ingest.py**

Add the following function to `src/kg/ingest.py`:

```python
def should_run_cross_doc_inference(doc_count: int, interval: int = 10) -> bool:
    """Check if cross-document inference should run based on ingestion count."""
    return doc_count > 0 and doc_count % interval == 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/kg/test_ingest_enrichment.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/kg/ingest.py tests/unit/kg/test_ingest_enrichment.py
git commit -m "feat: add cross-document inference trigger to KG ingest"
```

---

## Task 19: SPLADE Sparse Embeddings

**Files:**
- Create: `src/embedding/sparse.py`
- Test: `tests/unit/embedding/test_sparse.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/embedding/test_sparse.py
import pytest
from unittest.mock import patch, MagicMock


def test_sparse_encoder_returns_indices_and_values():
    from src.embedding.sparse import SparseEncoder
    encoder = SparseEncoder()
    with patch.object(encoder, "_model") as mock_model:
        # Simulate SPLADE output: sparse activations
        import torch
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.0, 0.5, 0.0, 1.2, 0.0, 0.8]])
        mock_model.return_value = mock_output
        with patch.object(encoder, "_tokenizer") as mock_tok:
            mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]), "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]])}
            mock_tok.decode = lambda ids: "test"
            result = encoder.encode("test query")
            assert "indices" in result
            assert "values" in result
            assert len(result["indices"]) == len(result["values"])
            assert all(v > 0 for v in result["values"])


def test_sparse_encoder_batch():
    from src.embedding.sparse import SparseEncoder
    encoder = SparseEncoder()
    with patch.object(encoder, "encode") as mock_encode:
        mock_encode.return_value = {"indices": [1, 3, 5], "values": [0.5, 1.2, 0.8]}
        results = encoder.encode_batch(["text1", "text2"])
        assert len(results) == 2


def test_sparse_to_qdrant_format():
    from src.embedding.sparse import sparse_to_qdrant
    sparse = {"indices": [1, 3, 5], "values": [0.5, 1.2, 0.8]}
    qdrant_vec = sparse_to_qdrant(sparse)
    assert qdrant_vec.indices == [1, 3, 5]
    assert qdrant_vec.values == [0.5, 1.2, 0.8]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/embedding/test_sparse.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/embedding/sparse.py
"""SPLADE v3 sparse embedding encoder."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "naver/splade-v3"


class SparseEncoder:
    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str = "auto"):
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            if self._device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self._model_name).to(self._device)
            self._model.eval()
            logger.info("Loaded SPLADE model: %s on %s", self._model_name, self._device)
        except Exception as exc:
            logger.error("Failed to load SPLADE model: %s", exc)
            raise

    def encode(self, text: str) -> Dict[str, List]:
        self._ensure_loaded()

        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        with torch.no_grad():
            output = self._model(**tokens)

        # SPLADE: log(1 + ReLU(logits)) * attention_mask, then max-pool over sequence
        logits = output.logits
        relu_log = torch.log1p(torch.relu(logits))
        weighted = relu_log * tokens["attention_mask"].unsqueeze(-1)
        sparse_vec = weighted.max(dim=1).values.squeeze(0)

        # Extract non-zero indices and values
        nonzero = sparse_vec.nonzero(as_tuple=True)[0]
        indices = nonzero.cpu().tolist()
        values = sparse_vec[nonzero].cpu().tolist()

        return {"indices": indices, "values": values}

    def encode_batch(self, texts: List[str]) -> List[Dict[str, List]]:
        return [self.encode(text) for text in texts]


def sparse_to_qdrant(sparse: Dict[str, List]) -> Any:
    """Convert sparse dict to Qdrant SparseVector format."""
    from qdrant_client.models import SparseVector
    return SparseVector(indices=sparse["indices"], values=sparse["values"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/embedding/test_sparse.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedding/sparse.py tests/unit/embedding/test_sparse.py
git commit -m "feat: add SPLADE v3 sparse embedding encoder"
```

---

## Task 20: V2 Model Embeddings

**Files:**
- Create: `src/embedding/v2_embeddings.py`
- Test: `tests/unit/embedding/test_v2_embeddings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/embedding/test_v2_embeddings.py
import pytest
import torch
from unittest.mock import patch, MagicMock


def test_v2_embedder_output_dimension():
    from src.embedding.v2_embeddings import V2Embedder
    embedder = V2Embedder.__new__(V2Embedder)
    embedder._projection = torch.nn.Linear(5120, 1024)
    embedder._model = None
    embedder._tokenizer = None
    embedder._device = "cpu"
    # Test projection shape
    fake_hidden = torch.randn(1, 10, 5120)  # batch=1, seq=10, dim=5120
    pooled = fake_hidden.mean(dim=1)  # (1, 5120)
    projected = embedder._projection(pooled)
    assert projected.shape == (1, 1024)


def test_v2_embedder_encode_returns_list():
    from src.embedding.v2_embeddings import V2Embedder
    embedder = V2Embedder.__new__(V2Embedder)
    embedder._device = "cpu"
    embedder._projection = torch.nn.Linear(5120, 1024)
    with patch.object(embedder, "_get_hidden_states") as mock_hidden:
        mock_hidden.return_value = torch.randn(1, 10, 5120)
        result = embedder._encode_single("test text")
        assert len(result) == 1024
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/embedding/test_v2_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/embedding/v2_embeddings.py
"""V2 model hidden-state embeddings for domain-tuned retrieval."""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class V2Embedder:
    def __init__(
        self,
        model_name: str = "docwain:v2",
        ollama_host: str = "http://localhost:11434",
        projection_dim: int = 1024,
        hidden_dim: int = 5120,
        device: str = "auto",
    ):
        self._model_name = model_name
        self._ollama_host = ollama_host
        self._device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._projection = nn.Linear(hidden_dim, projection_dim).to(self._device)
        self._projection.eval()
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
            self._model.eval()
            logger.info("Loaded V2 embedding model: %s", self._model_name)
        except Exception as exc:
            logger.warning("V2 model not available for embeddings: %s", exc)

    def _get_hidden_states(self, text: str) -> torch.Tensor:
        self._ensure_loaded()
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = {k: v.to(self._device) for k, v in tokens.items()}
        with torch.no_grad():
            output = self._model(**tokens, output_hidden_states=True)
        # Penultimate layer
        return output.hidden_states[-2]

    def _encode_single(self, text: str) -> List[float]:
        hidden = self._get_hidden_states(text)
        # Mean pooling over sequence dimension
        pooled = hidden.mean(dim=1)  # (1, hidden_dim)
        with torch.no_grad():
            projected = self._projection(pooled)  # (1, projection_dim)
        return projected.squeeze(0).cpu().tolist()

    def encode(self, text: str) -> List[float]:
        return self._encode_single(text)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._encode_single(t) for t in texts]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/embedding/test_v2_embeddings.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedding/v2_embeddings.py tests/unit/embedding/test_v2_embeddings.py
git commit -m "feat: add V2 model hidden-state embeddings with projection"
```

---

## Task 21: Semantic Chunker

**Files:**
- Create: `src/embedding/chunking/semantic_chunker.py`
- Test: `tests/unit/embedding/test_semantic_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/embedding/test_semantic_chunker.py
import pytest


def test_semantic_chunker_splits_at_boundaries():
    from src.embedding.chunking.semantic_chunker import SemanticChunker
    chunker = SemanticChunker(min_chunk_chars=50, max_chunk_chars=500)
    text = (
        "Section 1: Pricing Terms\n"
        "The base price for Widget A is $100 per unit. "
        "Volume discounts apply for orders over 1000 units. "
        "Payment is due within 30 days of invoice.\n\n"
        "Section 2: Delivery Terms\n"
        "Standard delivery is 5-7 business days. "
        "Express delivery is available at additional cost. "
        "All shipments include tracking information."
    )
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # First chunk should be about pricing, second about delivery
    assert "pricing" in chunks[0]["text"].lower() or "price" in chunks[0]["text"].lower()


def test_semantic_chunker_preserves_tables():
    from src.embedding.chunking.semantic_chunker import SemanticChunker
    chunker = SemanticChunker()
    text = (
        "Summary of findings:\n"
        "| Item | Qty | Price |\n"
        "| Widget A | 100 | $10,000 |\n"
        "| Widget B | 50 | $7,500 |\n\n"
        "Additional notes follow."
    )
    chunks = chunker.chunk(text)
    # Table should not be split across chunks
    table_chunks = [c for c in chunks if "|" in c["text"]]
    assert len(table_chunks) >= 1
    # All table rows should be in the same chunk
    table_chunk = table_chunks[0]
    assert "Widget A" in table_chunk["text"]
    assert "Widget B" in table_chunk["text"]


def test_semantic_chunker_hierarchical_levels():
    from src.embedding.chunking.semantic_chunker import SemanticChunker
    chunker = SemanticChunker()
    text = "Section 1: Overview\nThis is the overview.\n\nSection 2: Details\nThese are the details."
    chunks = chunker.chunk(text, hierarchical=True)
    levels = {c.get("level") for c in chunks}
    assert "section" in levels or "paragraph" in levels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/embedding/test_semantic_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/embedding/chunking/semantic_chunker.py
"""Semantic boundary-aware chunker for document text."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(
    r"^(?:section\s+\d+|chapter\s+\d+|\d+(?:\.\d+)*\.?\s+\S|#{1,4}\s+\S)",
    re.IGNORECASE | re.MULTILINE,
)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|", re.MULTILINE)
_BLANK_LINE_RE = re.compile(r"\n\s*\n")


class SemanticChunker:
    def __init__(
        self,
        min_chunk_chars: int = 100,
        max_chunk_chars: int = 2000,
    ):
        self._min = min_chunk_chars
        self._max = max_chunk_chars

    def chunk(
        self,
        text: str,
        *,
        hierarchical: bool = False,
        doc_id: str = "",
    ) -> List[Dict[str, Any]]:
        sections = self._split_at_semantic_boundaries(text)

        chunks = []
        for i, section in enumerate(sections):
            section_text = section["text"].strip()
            if not section_text:
                continue

            if len(section_text) > self._max:
                sub_chunks = self._split_preserving_structure(section_text)
                for j, sub in enumerate(sub_chunks):
                    chunks.append({
                        "text": sub,
                        "section_title": section.get("title", ""),
                        "chunk_index": len(chunks),
                        "level": "paragraph",
                        "parent_chunk_id": f"{doc_id}_s{i}" if hierarchical else None,
                    })
            elif len(section_text) >= self._min:
                chunks.append({
                    "text": section_text,
                    "section_title": section.get("title", ""),
                    "chunk_index": len(chunks),
                    "level": "section",
                    "parent_chunk_id": None,
                })
            else:
                # Merge small sections with next
                if chunks:
                    chunks[-1]["text"] += "\n\n" + section_text
                else:
                    chunks.append({
                        "text": section_text,
                        "section_title": section.get("title", ""),
                        "chunk_index": 0,
                        "level": "section",
                        "parent_chunk_id": None,
                    })

        return chunks

    def _split_at_semantic_boundaries(self, text: str) -> List[Dict[str, str]]:
        """Split text at heading-level boundaries."""
        headings = list(_HEADING_RE.finditer(text))

        if not headings:
            # No headings — split at blank lines
            parts = _BLANK_LINE_RE.split(text)
            return [{"text": p, "title": ""} for p in parts if p.strip()]

        sections = []
        for i, match in enumerate(headings):
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            section_text = text[start:end]

            # Extract title from heading line
            first_line = section_text.split("\n", 1)[0].strip()
            title = re.sub(r"^(?:section\s+\d+[:\s]*|chapter\s+\d+[:\s]*|\d+(?:\.\d+)*\.?\s*|#{1,4}\s*)", "", first_line, flags=re.IGNORECASE).strip()

            sections.append({"text": section_text, "title": title})

        # Include any text before the first heading
        if headings and headings[0].start() > 0:
            preamble = text[:headings[0].start()].strip()
            if preamble:
                sections.insert(0, {"text": preamble, "title": ""})

        return sections

    def _split_preserving_structure(self, text: str) -> List[str]:
        """Split large text without breaking tables or lists."""
        lines = text.split("\n")
        chunks = []
        current = []
        current_len = 0
        in_table = False

        for line in lines:
            is_table_row = bool(_TABLE_ROW_RE.match(line))

            if is_table_row:
                in_table = True
                current.append(line)
                current_len += len(line) + 1
                continue

            if in_table and not is_table_row:
                in_table = False

            if current_len + len(line) > self._max and current and not in_table:
                chunks.append("\n".join(current))
                current = []
                current_len = 0

            current.append(line)
            current_len += len(line) + 1

        if current:
            chunks.append("\n".join(current))

        return [c for c in chunks if c.strip()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/embedding/test_semantic_chunker.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedding/chunking/semantic_chunker.py tests/unit/embedding/test_semantic_chunker.py
git commit -m "feat: add semantic boundary-aware chunker with table preservation"
```

---

## Task 22: KG-Enriched Embedding Text

**Files:**
- Create: `src/embedding/kg_enrichment.py`
- Test: `tests/unit/embedding/test_kg_enrichment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/embedding/test_kg_enrichment.py
import pytest
from unittest.mock import patch, MagicMock


def test_enrich_prepends_kg_context():
    from src.embedding.kg_enrichment import enrich_chunk_text
    chunk_text = "The vendor delivered 500 units on March 15"
    kg_context = {"document": "Invoice #4521", "entities": {"Acme Corp": "ORGANIZATION", "Project Alpha": "PROJECT"}}
    enriched = enrich_chunk_text(chunk_text, kg_context)
    assert "[Doc: Invoice #4521]" in enriched
    assert "[Acme Corp]" in enriched
    assert "The vendor delivered 500 units" in enriched


def test_enrich_no_context_returns_original():
    from src.embedding.kg_enrichment import enrich_chunk_text
    text = "Simple text"
    enriched = enrich_chunk_text(text, None)
    assert enriched == text


def test_enrich_empty_entities():
    from src.embedding.kg_enrichment import enrich_chunk_text
    text = "Some content"
    enriched = enrich_chunk_text(text, {"document": "doc.pdf", "entities": {}})
    assert "[Doc: doc.pdf]" in enriched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/embedding/test_kg_enrichment.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/embedding/kg_enrichment.py
"""Prepend KG context to chunk text before embedding."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def enrich_chunk_text(
    chunk_text: str,
    kg_context: Optional[Dict[str, Any]],
    *,
    max_prefix_chars: int = 200,
) -> str:
    if not kg_context:
        return chunk_text

    parts = []

    doc_name = kg_context.get("document")
    if doc_name:
        parts.append(f"[Doc: {doc_name}]")

    entities = kg_context.get("entities", {})
    for name, etype in entities.items():
        parts.append(f"[{name}]")
        if len(" ".join(parts)) > max_prefix_chars:
            break

    if not parts:
        return chunk_text

    prefix = " ".join(parts)
    return f"{prefix} {chunk_text}"


def fetch_kg_context_for_chunk(
    document_id: str,
    chunk_id: str,
    *,
    neo4j_store: Any = None,
    redis_client: Any = None,
) -> Optional[Dict[str, Any]]:
    """Fetch KG context for a chunk from Neo4j (with Redis cache)."""
    if redis_client:
        cache_key = f"kg_ctx:{document_id}:{chunk_id}"
        cached = redis_client.get(cache_key)
        if cached:
            import json
            return json.loads(cached)

    if not neo4j_store:
        return None

    try:
        results = neo4j_store.run_query(
            "MATCH (d:Document {doc_id: $doc_id})-[:MENTIONS]->(e:Entity) "
            "RETURN d.doc_name AS doc_name, collect({name: e.name, type: labels(e)[0]}) AS entities "
            "LIMIT 1",
            {"doc_id": document_id},
        )
        if not results:
            return None

        row = results[0]
        entities = {e["name"]: e["type"] for e in row.get("entities", [])[:10]}

        context = {"document": row.get("doc_name", document_id), "entities": entities}

        if redis_client:
            import json
            redis_client.setex(cache_key, 3600, json.dumps(context))

        return context

    except Exception as exc:
        logger.warning("Failed to fetch KG context for %s: %s", document_id, exc)
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/embedding/test_kg_enrichment.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedding/kg_enrichment.py tests/unit/embedding/test_kg_enrichment.py
git commit -m "feat: add KG-enriched embedding text with Neo4j context prepending"
```

---

## Task 23: Three-Signal Retrieval Fusion

**Files:**
- Create: `src/retrieval/fusion.py`
- Test: `tests/unit/retrieval/test_fusion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/retrieval/test_fusion.py
import pytest


def test_rrf_fusion_basic():
    from src.retrieval.fusion import reciprocal_rank_fusion
    rankings = {
        "bge": ["doc1", "doc2", "doc3"],
        "splade": ["doc2", "doc1", "doc4"],
        "v2": ["doc3", "doc2", "doc1"],
    }
    fused = reciprocal_rank_fusion(rankings, k=60)
    # doc2 appears in all 3 rankings — should be top
    assert fused[0] == "doc2"


def test_rrf_with_weights():
    from src.retrieval.fusion import reciprocal_rank_fusion
    rankings = {
        "bge": ["doc1"],
        "splade": ["doc2"],
    }
    weights = {"bge": 0.8, "splade": 0.2}
    fused = reciprocal_rank_fusion(rankings, k=60, weights=weights)
    # doc1 should rank higher due to heavier bge weight
    assert fused[0] == "doc1"


def test_fusion_retriever_combines_signals():
    from src.retrieval.fusion import FusionRetriever
    retriever = FusionRetriever(
        default_weights={"bge": 0.4, "splade": 0.3, "v2": 0.3}
    )
    assert retriever._weights["bge"] == 0.4
    assert sum(retriever._weights.values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/retrieval/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/retrieval/fusion.py
"""Three-signal retrieval fusion using Reciprocal Rank Fusion."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = {"bge": 0.4, "splade": 0.3, "v2": 0.3}


def reciprocal_rank_fusion(
    rankings: Dict[str, List[str]],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Fuse multiple ranked lists using RRF with optional per-signal weights."""
    scores: Dict[str, float] = defaultdict(float)

    for signal_name, ranked_ids in rankings.items():
        weight = (weights or {}).get(signal_name, 1.0)
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] += weight / (k + rank + 1)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_ids


class FusionRetriever:
    def __init__(
        self,
        default_weights: Optional[Dict[str, float]] = None,
        rrf_k: int = 60,
    ):
        self._weights = default_weights or dict(_DEFAULT_WEIGHTS)
        self._rrf_k = rrf_k

    def fuse(
        self,
        bge_results: List[str],
        splade_results: List[str],
        v2_results: List[str],
        *,
        query_type: Optional[str] = None,
    ) -> List[str]:
        weights = self._adjust_weights_for_query(query_type)

        rankings = {}
        if bge_results:
            rankings["bge"] = bge_results
        if splade_results:
            rankings["splade"] = splade_results
        if v2_results:
            rankings["v2"] = v2_results

        return reciprocal_rank_fusion(rankings, k=self._rrf_k, weights=weights)

    def _adjust_weights_for_query(self, query_type: Optional[str]) -> Dict[str, float]:
        if not query_type:
            return dict(self._weights)

        weights = dict(self._weights)
        if query_type in ("exact_lookup", "id_search"):
            weights["splade"] = 0.6
            weights["bge"] = 0.2
            weights["v2"] = 0.2
        elif query_type in ("conceptual", "summary"):
            weights["v2"] = 0.5
            weights["bge"] = 0.35
            weights["splade"] = 0.15

        return weights
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/retrieval/test_fusion.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/fusion.py tests/unit/retrieval/test_fusion.py
git commit -m "feat: add three-signal RRF retrieval fusion"
```

---

## Task 24: Embedding Feedback Loop

**Files:**
- Create: `src/embedding/feedback.py`
- Test: `tests/unit/embedding/test_feedback.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/embedding/test_feedback.py
import pytest


def test_record_retrieval_outcome():
    from src.embedding.feedback import RetrievalFeedbackTracker
    tracker = RetrievalFeedbackTracker()
    tracker.record_outcome(
        query="what is the total?",
        retrieved_ids=["chunk1", "chunk2", "chunk3"],
        relevant_ids=["chunk1", "chunk3"],
    )
    metrics = tracker.get_metrics()
    assert metrics["total_queries"] == 1
    assert metrics["precision_at_k"] == pytest.approx(2 / 3, abs=0.01)
    assert metrics["recall_at_k"] == pytest.approx(1.0)


def test_hard_negatives():
    from src.embedding.feedback import RetrievalFeedbackTracker
    tracker = RetrievalFeedbackTracker()
    tracker.record_outcome(
        query="what is the total?",
        retrieved_ids=["chunk1", "chunk2", "chunk3"],
        relevant_ids=["chunk1"],
    )
    negatives = tracker.get_hard_negatives()
    assert "chunk2" in negatives
    assert "chunk3" in negatives
    assert "chunk1" not in negatives


def test_mrr_calculation():
    from src.embedding.feedback import RetrievalFeedbackTracker
    tracker = RetrievalFeedbackTracker()
    # Relevant doc at position 2
    tracker.record_outcome(query="q1", retrieved_ids=["a", "b", "c"], relevant_ids=["b"])
    metrics = tracker.get_metrics()
    assert metrics["mrr"] == pytest.approx(0.5)  # 1/rank = 1/2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/embedding/test_feedback.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/embedding/feedback.py
"""Retrieval quality tracking and hard negative mining."""
from __future__ import annotations

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class RetrievalFeedbackTracker:
    def __init__(self):
        self._outcomes: List[Dict] = []
        self._hard_negatives: Set[str] = set()

    def record_outcome(
        self,
        query: str,
        retrieved_ids: List[str],
        relevant_ids: List[str],
    ) -> None:
        relevant_set = set(relevant_ids)
        retrieved_set = set(retrieved_ids)

        # Precision and recall
        true_positives = relevant_set & retrieved_set
        precision = len(true_positives) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(true_positives) / len(relevant_ids) if relevant_ids else 0.0

        # MRR
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                mrr = 1.0 / rank
                break

        # Hard negatives: retrieved but not relevant
        negatives = retrieved_set - relevant_set
        self._hard_negatives.update(negatives)

        self._outcomes.append({
            "query": query,
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "hard_negatives": list(negatives),
        })

    def get_metrics(self) -> Dict[str, float]:
        if not self._outcomes:
            return {"total_queries": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}

        n = len(self._outcomes)
        return {
            "total_queries": n,
            "precision_at_k": sum(o["precision"] for o in self._outcomes) / n,
            "recall_at_k": sum(o["recall"] for o in self._outcomes) / n,
            "mrr": sum(o["mrr"] for o in self._outcomes) / n,
        }

    def get_hard_negatives(self) -> List[str]:
        return sorted(self._hard_negatives)

    def clear(self) -> None:
        self._outcomes.clear()
        self._hard_negatives.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/embedding/test_feedback.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedding/feedback.py tests/unit/embedding/test_feedback.py
git commit -m "feat: add retrieval feedback tracking with hard negative mining"
```

---

## Task 25: Post-Extraction Validator

**Files:**
- Create: `src/extraction/validator.py`
- Test: `tests/unit/extraction/test_validator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/extraction/test_validator.py
import pytest


def test_self_consistency_passes():
    from src.extraction.validator import ExtractionValidator
    from src.extraction.models import ExtractionResult
    validator = ExtractionValidator()
    result = ExtractionResult(
        document_id="doc1", subscription_id="sub1", profile_id="prof1",
        clean_text="Invoice total: $1500",
        entities=[{"text": "$1,500", "type": "AMOUNT"}],
    )
    validation = validator.validate(result)
    assert validation.passed is True


def test_self_consistency_fails_on_date_issue():
    from src.extraction.validator import ExtractionValidator
    from src.extraction.models import ExtractionResult
    validator = ExtractionValidator()
    result = ExtractionResult(
        document_id="doc1", subscription_id="sub1", profile_id="prof1",
        clean_text="Contract from 2025-01-01 to 2024-12-31",
        metadata={"start_date": "2025-01-01", "end_date": "2024-12-31"},
    )
    validation = validator.validate(result)
    assert validation.passed is False
    assert "date_chronology" in validation.failed_checks


def test_low_confidence_fields_flagged():
    from src.extraction.validator import ExtractionValidator
    from src.extraction.models import ExtractionResult
    validator = ExtractionValidator(confidence_threshold=0.7)
    result = ExtractionResult(
        document_id="doc1", subscription_id="sub1", profile_id="prof1",
        clean_text="test",
        metadata={"extraction_confidence": 0.5},
    )
    validation = validator.validate(result)
    assert validation.passed is False
    assert "low_confidence" in validation.failed_checks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_validator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/extraction/validator.py
"""Post-extraction validation with self-consistency checks."""
from __future__ import annotations

import logging
import re
from typing import Optional

from src.extraction.models import ExtractionResult, ValidationResult

logger = logging.getLogger(__name__)


class ExtractionValidator:
    def __init__(self, confidence_threshold: float = 0.7, max_retries: int = 2):
        self._confidence_threshold = confidence_threshold
        self._max_retries = max_retries

    def validate(self, result: ExtractionResult) -> ValidationResult:
        failed_checks = []
        field_confidences = {}

        # 1. Confidence check
        overall_confidence = result.metadata.get("extraction_confidence", 1.0) if result.metadata else 1.0
        field_confidences["overall"] = overall_confidence
        if overall_confidence < self._confidence_threshold:
            failed_checks.append("low_confidence")

        # 2. Date chronology check
        metadata = result.metadata or {}
        start_date = metadata.get("start_date")
        end_date = metadata.get("end_date")
        if start_date and end_date and start_date > end_date:
            failed_checks.append("date_chronology")

        # 3. Entity consistency — check for conflicting entity types
        if result.entities:
            entity_types = {}
            for e in result.entities:
                name = e.text.lower() if hasattr(e, "text") else str(e.get("text", "")).lower()
                etype = e.type if hasattr(e, "type") else e.get("type", "")
                if name in entity_types and entity_types[name] != etype:
                    failed_checks.append(f"entity_type_conflict:{name}")
                entity_types[name] = etype

        passed = len(failed_checks) == 0
        retry_recommended = not passed and overall_confidence < self._confidence_threshold

        return ValidationResult(
            passed=passed,
            failed_checks=failed_checks,
            field_confidences=field_confidences,
            retry_recommended=retry_recommended,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_validator.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/extraction/validator.py tests/unit/extraction/test_validator.py
git commit -m "feat: add post-extraction validator with self-consistency checks"
```

---

## Task 26: Insight Engine

**Files:**
- Create: `src/visualization/insights.py`
- Test: `tests/unit/visualization/test_insights.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/visualization/test_insights.py
import pytest


def test_categorize_insight():
    from src.visualization.insights import categorize_insight
    result = categorize_insight("Across 12 invoices, delivery is consistently 5-7 days late")
    assert result["category"] == "pattern_recognition"


def test_categorize_anomaly():
    from src.visualization.insights import categorize_insight
    result = categorize_insight("This contract's liability cap is 10x lower than comparable contracts")
    assert result["category"] == "anomaly_detection"


def test_insight_to_viz_mapping():
    from src.visualization.insights import insight_to_visualization
    viz = insight_to_visualization("trend_analysis", [10, 20, 30, 40])
    assert viz["chart_type"] == "line"


def test_insight_severity():
    from src.visualization.insights import classify_severity
    assert classify_severity("anomaly_detection", 0.95) == "critical"
    assert classify_severity("pattern_recognition", 0.6) == "info"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/visualization/test_insights.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/visualization/insights.py
"""Insight categorization and visualization routing."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CATEGORY_PATTERNS = {
    "pattern_recognition": [
        r"consistently", r"pattern", r"repeatedly", r"across\s+\d+",
        r"always", r"every\s+time", r"regularly",
    ],
    "anomaly_detection": [
        r"\d+x\s+(lower|higher|more|less)", r"anomal", r"unusual",
        r"outlier", r"significantly\s+(lower|higher|different)",
        r"deviates?", r"unexpected",
    ],
    "trend_analysis": [
        r"increas", r"decreas", r"trend", r"growth",
        r"over\s+(?:the\s+)?(?:last|past|next)", r"quarter-over-quarter",
        r"year-over-year", r"rising", r"falling",
    ],
    "comparative_analysis": [
        r"compar", r"differ", r"version\s+\d+", r"versus",
        r"unlike", r"in\s+contrast", r"removes?\s+the",
    ],
    "gap_analysis": [
        r"missing", r"gap", r"covers?\s+\d+\s+of\s+\d+",
        r"incomplete", r"absent", r"lacks?",
    ],
}

_INSIGHT_VIZ_MAP = {
    "pattern_recognition": "bar",
    "anomaly_detection": "bar",
    "trend_analysis": "line",
    "comparative_analysis": "bar",
    "gap_analysis": "pie",
}


def categorize_insight(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    best_category = "pattern_recognition"
    best_score = 0

    for category, patterns in _CATEGORY_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score > best_score:
            best_score = score
            best_category = category

    return {"category": best_category, "score": best_score, "text": text}


def insight_to_visualization(
    category: str,
    data: Any,
    title: str = "",
) -> Dict[str, Any]:
    chart_type = _INSIGHT_VIZ_MAP.get(category, "bar")
    return {
        "chart_type": chart_type,
        "data": data,
        "title": title or f"{category.replace('_', ' ').title()}",
    }


def classify_severity(category: str, confidence: float) -> str:
    if category == "anomaly_detection" and confidence >= 0.8:
        return "critical"
    if category == "anomaly_detection":
        return "warning"
    if confidence >= 0.8:
        return "warning"
    return "info"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/visualization/test_insights.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualization/insights.py tests/unit/visualization/test_insights.py
git commit -m "feat: add insight categorization and visualization routing"
```

---

## Task 27: Multi-Document Dashboard

**Files:**
- Create: `src/visualization/dashboard.py`
- Test: `tests/unit/visualization/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/visualization/test_dashboard.py
import pytest


def test_compose_dashboard_from_multiple_docs():
    from src.visualization.dashboard import compose_dashboard
    data = {
        "documents": ["Invoice_1.pdf", "Invoice_2.pdf", "Invoice_3.pdf"],
        "values": [1500, 2300, 1800],
        "dates": ["2025-01", "2025-02", "2025-03"],
    }
    dashboard = compose_dashboard(data, query="Summarize all invoices")
    assert "sections" in dashboard
    assert len(dashboard["sections"]) >= 1
    # Should have at least a data table and a chart
    section_types = {s["type"] for s in dashboard["sections"]}
    assert "table" in section_types or "chart" in section_types


def test_dashboard_empty_data():
    from src.visualization.dashboard import compose_dashboard
    dashboard = compose_dashboard({}, query="test")
    assert dashboard["sections"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/visualization/test_dashboard.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/visualization/dashboard.py
"""Multi-document dashboard response composition."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compose_dashboard(
    data: Dict[str, Any],
    query: str,
    *,
    max_sections: int = 5,
) -> Dict[str, Any]:
    if not data:
        return {"sections": [], "query": query}

    sections: List[Dict[str, Any]] = []

    documents = data.get("documents", [])
    values = data.get("values", [])
    dates = data.get("dates", [])

    # 1. Summary table
    if documents and values:
        rows = []
        for i, doc in enumerate(documents):
            row = {"document": doc}
            if i < len(values):
                row["value"] = values[i]
            if i < len(dates):
                row["date"] = dates[i]
            rows.append(row)

        sections.append({
            "type": "table",
            "title": "Document Summary",
            "data": rows,
        })

    # 2. Value chart (if numeric data exists)
    if values and len(values) >= 2:
        labels = documents if documents else [f"Doc {i+1}" for i in range(len(values))]
        sections.append({
            "type": "chart",
            "chart_type": "bar",
            "title": "Values by Document",
            "x": labels[:len(values)],
            "y": values,
        })

    # 3. Timeline (if dates exist)
    if dates and values and len(dates) >= 2:
        sections.append({
            "type": "chart",
            "chart_type": "line",
            "title": "Timeline",
            "x": dates,
            "y": values[:len(dates)],
        })

    return {
        "sections": sections[:max_sections],
        "query": query,
        "document_count": len(documents),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/visualization/test_dashboard.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualization/dashboard.py tests/unit/visualization/test_dashboard.py
git commit -m "feat: add multi-document dashboard response composition"
```

---

## Task 28: Retarget Daily Fine-Tune Loop

**Files:**
- Modify: `src/finetune/agentic_orchestrator.py`
- Test: `tests/unit/finetune/test_agentic_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/finetune/test_agentic_orchestrator.py
import pytest
from unittest.mock import patch


def test_target_model_is_v2():
    from src.finetune.agentic_orchestrator import get_target_model
    model = get_target_model()
    assert model == "docwain:v2"


def test_data_policy_rejects_document_content():
    from src.finetune.agentic_orchestrator import enforce_data_policy
    # Training pair with raw document content should be rejected
    pair = {"question": "What is in this doc?", "answer": "The document contains: [full text of page 1]...",
            "source": "document_content"}
    assert enforce_data_policy(pair) is False


def test_data_policy_allows_qa_pairs():
    from src.finetune.agentic_orchestrator import enforce_data_policy
    pair = {"question": "What is the total?", "answer": "$1,500",
            "source": "user_feedback"}
    assert enforce_data_policy(pair) is True


def test_data_policy_allows_metadata():
    from src.finetune.agentic_orchestrator import enforce_data_policy
    pair = {"question": "What type of document is this?", "answer": "Invoice",
            "source": "metadata"}
    assert enforce_data_policy(pair) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/test_agentic_orchestrator.py -v`
Expected: FAIL — `get_target_model` and `enforce_data_policy` don't exist

- [ ] **Step 3: Add functions to agentic_orchestrator.py**

Add to `src/finetune/agentic_orchestrator.py`:

```python
_TARGET_MODEL = "docwain:v2"
_BLOCKED_SOURCES = {"document_content", "raw_text", "embedding_vector"}


def get_target_model() -> str:
    """Return the target model for daily fine-tuning."""
    return _TARGET_MODEL


def enforce_data_policy(pair: dict) -> bool:
    """Enforce training data policy — reject raw document content."""
    source = pair.get("source", "")
    if source in _BLOCKED_SOURCES:
        return False

    # Check for suspiciously long answers that might be document dumps
    answer = pair.get("answer", "")
    if len(answer) > 2000:
        return False

    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/finetune/test_agentic_orchestrator.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/finetune/agentic_orchestrator.py tests/unit/finetune/test_agentic_orchestrator.py
git commit -m "feat: retarget daily fine-tune to V2 14B with data policy enforcement"
```

---

## Task 29: Update Visualization Enhancer

**Files:**
- Modify: `src/visualization/enhancer.py`
- Test: `tests/unit/visualization/test_enhancer_upgrade.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/visualization/test_enhancer_upgrade.py
import pytest


def test_parse_viz_directive():
    from src.visualization.enhancer import parse_viz_directives
    text = 'Here is the data. <viz type="bar" title="Revenue" x="[Q1,Q2,Q3]" y="[100,200,150]" /> And more text.'
    directives, clean_text = parse_viz_directives(text)
    assert len(directives) == 1
    assert directives[0]["type"] == "bar"
    assert directives[0]["title"] == "Revenue"
    assert "<viz" not in clean_text
    assert "And more text." in clean_text


def test_parse_no_directives():
    from src.visualization.enhancer import parse_viz_directives
    text = "Just a plain response with no charts."
    directives, clean_text = parse_viz_directives(text)
    assert len(directives) == 0
    assert clean_text == text


def test_parse_multiple_directives():
    from src.visualization.enhancer import parse_viz_directives
    text = '<viz type="bar" title="A" /> text <viz type="line" title="B" />'
    directives, clean_text = parse_viz_directives(text)
    assert len(directives) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/visualization/test_enhancer_upgrade.py -v`
Expected: FAIL — `parse_viz_directives` doesn't exist

- [ ] **Step 3: Add viz directive parser to enhancer.py**

Add to `src/visualization/enhancer.py`:

```python
import re as _re

_VIZ_RE = _re.compile(r'<viz\s+([^>]*?)\s*/>', _re.DOTALL)
_ATTR_RE = _re.compile(r'(\w+)="([^"]*)"')


def parse_viz_directives(text: str) -> tuple[list[dict], str]:
    """Parse <viz> directives from model output. Returns (directives, clean_text)."""
    directives = []
    for match in _VIZ_RE.finditer(text):
        attrs = dict(_ATTR_RE.findall(match.group(1)))
        directives.append(attrs)

    clean_text = _VIZ_RE.sub("", text).strip()
    # Clean up double spaces/newlines
    clean_text = _re.sub(r"\n{3,}", "\n\n", clean_text)
    clean_text = _re.sub(r"  +", " ", clean_text)

    return directives, clean_text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/visualization/test_enhancer_upgrade.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualization/enhancer.py tests/unit/visualization/test_enhancer_upgrade.py
git commit -m "feat: add viz directive parser for V2 model-native visualization"
```

---

## Task 30: Integration — Update Extraction Engine

**Files:**
- Modify: `src/extraction/engine.py`
- Test: `tests/unit/extraction/test_engine_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/extraction/test_engine_integration.py
import pytest
from unittest.mock import patch, MagicMock


def test_engine_uses_triage():
    from src.extraction.engine import ExtractionEngine
    engine = ExtractionEngine()
    with patch.object(engine, "triager") as mock_triage, \
         patch.object(engine, "structural") as mock_s, \
         patch.object(engine, "semantic") as mock_se, \
         patch.object(engine, "vision") as mock_v, \
         patch.object(engine, "v2_extractor") as mock_v2, \
         patch.object(engine, "merger") as mock_merger:
        from src.extraction.models import TriageResult
        mock_triage.triage.return_value = TriageResult(
            document_type="clean_digital",
            engine_weights={"structural": 0.9, "semantic": 0.8, "vision": 0.3, "v2": 0.7},
            preprocessing_directives=[],
            page_types={},
        )
        mock_s.extract.return_value = {}
        mock_se.extract.return_value = {}
        mock_v.extract.return_value = {}
        mock_v2.extract.return_value = {}
        mock_merger.merge.return_value = MagicMock()

        engine.extract("doc1", "sub1", "prof1", b"bytes", "pdf")
        mock_triage.triage.assert_called_once()
        mock_merger.merge.assert_called_once()


def test_engine_has_v2_extractor():
    from src.extraction.engine import ExtractionEngine
    engine = ExtractionEngine()
    assert hasattr(engine, "v2_extractor")
    assert hasattr(engine, "triager")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_engine_integration.py -v`
Expected: FAIL — engine doesn't have `v2_extractor` or `triager`

- [ ] **Step 3: Update engine.py**

Update `src/extraction/engine.py` `__init__` to add:
```python
from src.extraction.v2_extractor import V2Extractor
from src.extraction.triage import DocumentTriager

self.v2_extractor = V2Extractor(ollama_host=ollama_host)
self.triager = DocumentTriager()
```

Update `extract()` to:
1. Run triage first
2. Run all 4 extractors in parallel (ThreadPoolExecutor with 4 workers)
3. Pass triage and v2 results to merger

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/extraction/test_engine_integration.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/extraction/engine.py tests/unit/extraction/test_engine_integration.py
git commit -m "feat: integrate V2 extractor and triage into extraction engine"
```

---

## Final: Run All Tests

- [ ] **Step 1: Run full test suite**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/unit/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Final commit with all integration**

```bash
git add -A
git commit -m "feat: DocWain intelligence upgrade — V2 training pipeline + document processing"
```
