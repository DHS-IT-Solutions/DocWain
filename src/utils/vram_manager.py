"""
VRAM Memory Manager for DocWain GPU workloads.

Manages dynamic loading and eviction of models on the A100-SXM4-80GB (80GB VRAM)
based on execution mode, model priority, and available budget.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Operational modes that determine which models should be resident in VRAM."""

    DOCUMENT_PROCESSING = "document_processing"
    QUERY_ANSWERING = "query_answering"
    TRAINING = "training"


# Models required per execution mode (order does not imply priority).
_MODE_MODELS: Dict[ExecutionMode, List[str]] = {
    ExecutionMode.QUERY_ANSWERING: ["v2", "bge", "splade", "reranker"],
    ExecutionMode.DOCUMENT_PROCESSING: ["v2", "extraction", "bge"],
    ExecutionMode.TRAINING: ["v2"],
}


@dataclass
class _ModelEntry:
    """Internal record for a registered model."""

    name: str
    estimated_vram_mb: int
    priority: int  # lower number = higher priority (1 = inference, 2 = embedding, 3 = training)
    load_fn: Optional[Callable[[], None]] = None
    unload_fn: Optional[Callable[[], None]] = None
    loaded: bool = False


class VRAMManager:
    """
    Thread-safe VRAM budget manager.

    Tracks which models are loaded, enforces a total VRAM budget, and evicts
    lower-priority models when a higher-priority model needs to be loaded.

    Priority convention: lower number = higher priority.
      1 – inference (e.g. V2 generation)
      2 – embedding (e.g. BGE, SPLADE)
      3 – training / background
    """

    def __init__(
        self,
        total_vram_mb: int = 81_920,
        max_utilization: float = 0.9,
    ) -> None:
        if not (0.0 < max_utilization <= 1.0):
            raise ValueError("max_utilization must be in (0, 1]")
        self._budget_mb: int = int(total_vram_mb * max_utilization)
        self._total_vram_mb: int = total_vram_mb
        self._max_utilization: float = max_utilization
        self._models: Dict[str, _ModelEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_model(
        self,
        name: str,
        estimated_vram_mb: int,
        priority: int,
        load_fn: Optional[Callable[[], None]] = None,
        unload_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a model with the manager.

        Args:
            name: Unique model identifier.
            estimated_vram_mb: Estimated VRAM footprint in megabytes.
            priority: Scheduling priority (lower number = higher priority).
            load_fn: Optional callable invoked when the model is loaded.
            unload_fn: Optional callable invoked when the model is unloaded.
        """
        with self._lock:
            if name in self._models:
                raise ValueError(f"Model '{name}' is already registered.")
            self._models[name] = _ModelEntry(
                name=name,
                estimated_vram_mb=estimated_vram_mb,
                priority=priority,
                load_fn=load_fn,
                unload_fn=unload_fn,
            )
            logger.debug("Registered model '%s' (%d MB, priority %d)", name, estimated_vram_mb, priority)

    def request_load(self, name: str) -> bool:
        """Load a model into VRAM, evicting lower-priority models if necessary.

        Returns True if the model is loaded successfully, False if it cannot
        fit even after evicting all lower-priority models.
        """
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                raise KeyError(f"Model '{name}' is not registered.")
            if entry.loaded:
                return True

            # Fast path: fits without eviction.
            if self._used_vram_mb() + entry.estimated_vram_mb <= self._budget_mb:
                self._load(entry)
                return True

            # Attempt to free space by evicting lower-priority (higher number) models.
            freed = self._try_evict_for(entry)
            if freed:
                self._load(entry)
                return True

            logger.warning(
                "Cannot load '%s': insufficient VRAM even after eviction attempts.", name
            )
            return False

    def request_unload(self, name: str) -> bool:
        """Unload a model from VRAM.

        Returns True if the model was unloaded, False if it was not loaded.
        """
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                raise KeyError(f"Model '{name}' is not registered.")
            if not entry.loaded:
                return False
            self._unload(entry)
            return True

    def get_mode_plan(self, mode: ExecutionMode) -> Dict[str, List[str]]:
        """Return a plan describing what would change when switching to *mode*.

        Returns a dict with keys:
          "load"   – models that need to be loaded (not currently loaded).
          "unload" – models that are loaded but not needed by this mode.
          "keep"   – models that are already loaded and also needed.
        """
        required: Set[str] = set(_MODE_MODELS.get(mode, []))
        with self._lock:
            currently_loaded: Set[str] = {n for n, e in self._models.items() if e.loaded}
        load = sorted(required - currently_loaded)
        unload = sorted(currently_loaded - required)
        keep = sorted(required & currently_loaded)
        return {"load": load, "unload": unload, "keep": keep}

    def switch_mode(self, mode: ExecutionMode) -> Dict[str, List[str]]:
        """Switch to *mode* by executing the mode plan and returning it.

        Models not needed by the target mode are unloaded first (freeing space),
        then required models are loaded in priority order.
        """
        plan = self.get_mode_plan(mode)

        # Unload unneeded models first to free VRAM.
        for name in plan["unload"]:
            self.request_unload(name)

        # Load required models sorted by priority (lowest number first).
        required_names = plan["load"]
        with self._lock:
            entries_to_load = [
                self._models[n] for n in required_names if n in self._models
            ]
        entries_to_load.sort(key=lambda e: e.priority)

        for entry in entries_to_load:
            success = self.request_load(entry.name)
            if not success:
                logger.warning("switch_mode(%s): could not load '%s'", mode.value, entry.name)

        return plan

    @property
    def available_vram_mb(self) -> int:
        """Remaining VRAM budget in megabytes."""
        with self._lock:
            return self._budget_mb - self._used_vram_mb()

    # ------------------------------------------------------------------
    # Internal helpers (must be called under self._lock where applicable)
    # ------------------------------------------------------------------

    def _used_vram_mb(self) -> int:
        return sum(e.estimated_vram_mb for e in self._models.values() if e.loaded)

    def _load(self, entry: _ModelEntry) -> None:
        """Invoke load callback and mark model as loaded (lock already held)."""
        if entry.load_fn is not None:
            try:
                entry.load_fn()
            except Exception:
                logger.exception("load_fn for model '%s' raised an exception.", entry.name)
                raise
        entry.loaded = True
        logger.info("Loaded model '%s' (%d MB).", entry.name, entry.estimated_vram_mb)

    def _unload(self, entry: _ModelEntry) -> None:
        """Invoke unload callback and mark model as unloaded (lock already held)."""
        if entry.unload_fn is not None:
            try:
                entry.unload_fn()
            except Exception:
                logger.exception("unload_fn for model '%s' raised an exception.", entry.name)
                raise
        entry.loaded = False
        logger.info("Unloaded model '%s' (freed %d MB).", entry.name, entry.estimated_vram_mb)

    def _try_evict_for(self, target: _ModelEntry) -> bool:
        """Evict lower-priority models until *target* fits.

        Returns True if, after eviction, *target* would fit in the budget.
        Must be called with self._lock held.
        """
        # Candidates: loaded models with strictly lower priority (higher number).
        candidates = [
            e
            for e in self._models.values()
            if e.loaded and e.priority > target.priority
        ]
        # Evict lowest-priority first (highest number first).
        candidates.sort(key=lambda e: e.priority, reverse=True)

        for candidate in candidates:
            self._unload(candidate)
            if self._used_vram_mb() + target.estimated_vram_mb <= self._budget_mb:
                return True

        # Check once more in case the loop exhausted all candidates exactly.
        return self._used_vram_mb() + target.estimated_vram_mb <= self._budget_mb
