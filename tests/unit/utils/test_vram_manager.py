"""
Unit tests for src/utils/vram_manager.py
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, call

import pytest

from src.utils.vram_manager import ExecutionMode, VRAMManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(total_mb: int = 10_000, max_util: float = 1.0) -> VRAMManager:
    """Return a VRAMManager with a clean, known budget."""
    return VRAMManager(total_vram_mb=total_mb, max_utilization=max_util)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_budget(self):
        mgr = VRAMManager()
        assert mgr.available_vram_mb == int(81_920 * 0.9)

    def test_custom_budget(self):
        mgr = _make_manager(total_mb=10_000, max_util=0.8)
        assert mgr.available_vram_mb == 8_000

    def test_invalid_utilization_raises(self):
        with pytest.raises(ValueError):
            VRAMManager(max_utilization=0.0)
        with pytest.raises(ValueError):
            VRAMManager(max_utilization=1.1)


# ---------------------------------------------------------------------------
# register_model
# ---------------------------------------------------------------------------

class TestRegisterModel:
    def test_register_adds_model(self):
        mgr = _make_manager()
        mgr.register_model("bge", 2_000, priority=2)
        # Model registered but not loaded — VRAM is still fully available.
        assert mgr.available_vram_mb == 10_000

    def test_duplicate_register_raises(self):
        mgr = _make_manager()
        mgr.register_model("bge", 2_000, priority=2)
        with pytest.raises(ValueError, match="already registered"):
            mgr.register_model("bge", 2_000, priority=2)

    def test_register_with_callbacks(self):
        load_fn = MagicMock()
        unload_fn = MagicMock()
        mgr = _make_manager()
        mgr.register_model("v2", 3_000, priority=1, load_fn=load_fn, unload_fn=unload_fn)
        # Callbacks not invoked on registration.
        load_fn.assert_not_called()
        unload_fn.assert_not_called()


# ---------------------------------------------------------------------------
# request_load
# ---------------------------------------------------------------------------

class TestRequestLoad:
    def test_load_model_succeeds(self):
        mgr = _make_manager(total_mb=10_000)
        mgr.register_model("v2", 3_000, priority=1)
        result = mgr.request_load("v2")
        assert result is True
        assert mgr.available_vram_mb == 7_000

    def test_load_invokes_load_fn(self):
        load_fn = MagicMock()
        mgr = _make_manager()
        mgr.register_model("v2", 3_000, priority=1, load_fn=load_fn)
        mgr.request_load("v2")
        load_fn.assert_called_once()

    def test_load_already_loaded_is_noop(self):
        load_fn = MagicMock()
        mgr = _make_manager()
        mgr.register_model("v2", 3_000, priority=1, load_fn=load_fn)
        mgr.request_load("v2")
        mgr.request_load("v2")  # second call
        load_fn.assert_called_once()  # still only once
        assert mgr.available_vram_mb == 7_000

    def test_load_unknown_model_raises(self):
        mgr = _make_manager()
        with pytest.raises(KeyError):
            mgr.request_load("ghost")

    def test_available_vram_decreases_on_load(self):
        mgr = _make_manager(total_mb=10_000)
        mgr.register_model("bge", 2_000, priority=2)
        mgr.register_model("splade", 1_500, priority=2)
        mgr.request_load("bge")
        mgr.request_load("splade")
        assert mgr.available_vram_mb == 6_500


# ---------------------------------------------------------------------------
# Reject over-budget load
# ---------------------------------------------------------------------------

class TestRejectOverBudget:
    def test_single_model_exceeds_budget(self):
        mgr = _make_manager(total_mb=1_000)
        mgr.register_model("huge", 2_000, priority=1)
        result = mgr.request_load("huge")
        assert result is False
        assert mgr.available_vram_mb == 1_000

    def test_second_model_pushes_over_budget(self):
        mgr = _make_manager(total_mb=5_000)
        mgr.register_model("a", 3_000, priority=1)
        mgr.register_model("b", 3_000, priority=1)
        mgr.request_load("a")
        result = mgr.request_load("b")
        # Same priority – no eviction should happen; 'b' cannot fit.
        assert result is False
        assert mgr.available_vram_mb == 2_000  # only 'a' loaded

    def test_utilization_cap_enforced(self):
        # Budget = 10 000 * 0.8 = 8 000 MB.
        mgr = _make_manager(total_mb=10_000, max_util=0.8)
        mgr.register_model("a", 8_001, priority=1)
        assert mgr.request_load("a") is False


# ---------------------------------------------------------------------------
# Evict lower-priority model
# ---------------------------------------------------------------------------

class TestEvictLowPriority:
    def test_evicts_lower_priority_to_fit(self):
        # Budget: 5 000 MB.
        # 'low_prio' (priority=3, 3 000 MB) is loaded first.
        # 'high_prio' (priority=1, 4 000 MB) is requested → should evict 'low_prio'.
        mgr = _make_manager(total_mb=5_000)
        unload_fn = MagicMock()
        mgr.register_model("low_prio", 3_000, priority=3, unload_fn=unload_fn)
        mgr.register_model("high_prio", 4_000, priority=1)
        mgr.request_load("low_prio")
        assert mgr.available_vram_mb == 2_000

        result = mgr.request_load("high_prio")
        assert result is True
        unload_fn.assert_called_once()
        # low_prio freed, high_prio loaded: 5000 - 4000 = 1000 available.
        assert mgr.available_vram_mb == 1_000

    def test_does_not_evict_equal_priority(self):
        mgr = _make_manager(total_mb=5_000)
        mgr.register_model("a", 3_000, priority=2)
        mgr.register_model("b", 3_000, priority=2)
        mgr.request_load("a")
        result = mgr.request_load("b")
        # Same priority → no eviction; 'b' must not load.
        assert result is False
        assert mgr.available_vram_mb == 2_000

    def test_does_not_evict_higher_priority(self):
        mgr = _make_manager(total_mb=5_000)
        mgr.register_model("high", 3_000, priority=1)
        mgr.register_model("low", 3_000, priority=3)
        mgr.request_load("high")
        result = mgr.request_load("low")
        # 'low' cannot evict 'high'; must fail.
        assert result is False

    def test_evicts_multiple_models_if_needed(self):
        # Budget: 6 000 MB.
        # Two low-priority models each 2 000 MB loaded.
        # High-priority model 5 000 MB requested → must evict both.
        mgr = _make_manager(total_mb=6_000)
        mgr.register_model("low1", 2_000, priority=3)
        mgr.register_model("low2", 2_000, priority=3)
        mgr.register_model("high", 5_000, priority=1)
        mgr.request_load("low1")
        mgr.request_load("low2")
        assert mgr.available_vram_mb == 2_000

        result = mgr.request_load("high")
        assert result is True
        assert mgr.available_vram_mb == 1_000


# ---------------------------------------------------------------------------
# request_unload
# ---------------------------------------------------------------------------

class TestRequestUnload:
    def test_unload_loaded_model(self):
        mgr = _make_manager(total_mb=10_000)
        mgr.register_model("bge", 2_000, priority=2)
        mgr.request_load("bge")
        assert mgr.available_vram_mb == 8_000
        result = mgr.request_unload("bge")
        assert result is True
        assert mgr.available_vram_mb == 10_000

    def test_unload_not_loaded_returns_false(self):
        mgr = _make_manager()
        mgr.register_model("bge", 2_000, priority=2)
        result = mgr.request_unload("bge")
        assert result is False

    def test_unload_invokes_unload_fn(self):
        unload_fn = MagicMock()
        mgr = _make_manager()
        mgr.register_model("bge", 2_000, priority=2, unload_fn=unload_fn)
        mgr.request_load("bge")
        mgr.request_unload("bge")
        unload_fn.assert_called_once()

    def test_unload_unknown_model_raises(self):
        mgr = _make_manager()
        with pytest.raises(KeyError):
            mgr.request_unload("ghost")


# ---------------------------------------------------------------------------
# get_mode_plan
# ---------------------------------------------------------------------------

class TestGetModePlan:
    def _registered_manager(self) -> VRAMManager:
        mgr = _make_manager(total_mb=81_920)
        for name, mb, pri in [
            ("v2", 28_000, 1),
            ("bge", 2_000, 2),
            ("splade", 1_500, 2),
            ("reranker", 1_000, 2),
            ("extraction", 5_000, 2),
        ]:
            mgr.register_model(name, mb, priority=pri)
        return mgr

    def test_plan_all_unloaded(self):
        mgr = self._registered_manager()
        plan = mgr.get_mode_plan(ExecutionMode.QUERY_ANSWERING)
        assert set(plan["load"]) == {"v2", "bge", "splade", "reranker"}
        assert plan["unload"] == []
        assert plan["keep"] == []

    def test_plan_partial_already_loaded(self):
        mgr = self._registered_manager()
        mgr.request_load("v2")
        mgr.request_load("bge")
        plan = mgr.get_mode_plan(ExecutionMode.QUERY_ANSWERING)
        assert set(plan["load"]) == {"splade", "reranker"}
        assert set(plan["keep"]) == {"v2", "bge"}
        assert plan["unload"] == []

    def test_plan_unload_unneeded(self):
        mgr = self._registered_manager()
        mgr.request_load("splade")   # not needed by DOCUMENT_PROCESSING
        mgr.request_load("reranker") # not needed by DOCUMENT_PROCESSING
        plan = mgr.get_mode_plan(ExecutionMode.DOCUMENT_PROCESSING)
        assert set(plan["unload"]) == {"splade", "reranker"}
        assert set(plan["load"]) == {"v2", "bge", "extraction"}

    def test_plan_training_mode(self):
        mgr = self._registered_manager()
        mgr.request_load("bge")
        plan = mgr.get_mode_plan(ExecutionMode.TRAINING)
        assert set(plan["load"]) == {"v2"}
        assert set(plan["unload"]) == {"bge"}
        assert plan["keep"] == []

    def test_plan_returns_correct_keys(self):
        mgr = self._registered_manager()
        plan = mgr.get_mode_plan(ExecutionMode.DOCUMENT_PROCESSING)
        assert set(plan.keys()) == {"load", "unload", "keep"}


# ---------------------------------------------------------------------------
# switch_mode
# ---------------------------------------------------------------------------

class TestSwitchMode:
    def _full_manager(self) -> VRAMManager:
        mgr = VRAMManager(total_vram_mb=81_920, max_utilization=0.9)
        specs = [
            ("v2", 28_000, 1),
            ("bge", 2_000, 2),
            ("splade", 1_500, 2),
            ("reranker", 1_000, 2),
            ("extraction", 5_000, 2),
        ]
        for name, mb, pri in specs:
            mgr.register_model(name, mb, priority=pri)
        return mgr

    def test_switch_to_query_answering(self):
        mgr = self._full_manager()
        plan = mgr.switch_mode(ExecutionMode.QUERY_ANSWERING)
        assert set(plan["load"]) == {"v2", "bge", "splade", "reranker"}
        # All required models should now be loaded.
        for name in ["v2", "bge", "splade", "reranker"]:
            assert mgr._models[name].loaded

    def test_switch_to_document_processing(self):
        mgr = self._full_manager()
        mgr.switch_mode(ExecutionMode.QUERY_ANSWERING)
        plan = mgr.switch_mode(ExecutionMode.DOCUMENT_PROCESSING)
        # splade and reranker should be unloaded; extraction loaded.
        assert not mgr._models["splade"].loaded
        assert not mgr._models["reranker"].loaded
        assert mgr._models["extraction"].loaded
        assert mgr._models["v2"].loaded
        assert mgr._models["bge"].loaded

    def test_switch_to_training(self):
        mgr = self._full_manager()
        mgr.switch_mode(ExecutionMode.QUERY_ANSWERING)
        plan = mgr.switch_mode(ExecutionMode.TRAINING)
        assert mgr._models["v2"].loaded
        for name in ["bge", "splade", "reranker"]:
            assert not mgr._models[name].loaded

    def test_switch_mode_returns_plan(self):
        mgr = self._full_manager()
        plan = mgr.switch_mode(ExecutionMode.TRAINING)
        assert set(plan.keys()) == {"load", "unload", "keep"}


# ---------------------------------------------------------------------------
# available_vram_mb property
# ---------------------------------------------------------------------------

class TestAvailableVram:
    def test_full_budget_when_empty(self):
        mgr = _make_manager(total_mb=10_000, max_util=1.0)
        assert mgr.available_vram_mb == 10_000

    def test_budget_tracked_across_loads_and_unloads(self):
        mgr = _make_manager(total_mb=10_000)
        mgr.register_model("a", 3_000, priority=1)
        mgr.register_model("b", 2_000, priority=2)
        mgr.request_load("a")
        assert mgr.available_vram_mb == 7_000
        mgr.request_load("b")
        assert mgr.available_vram_mb == 5_000
        mgr.request_unload("a")
        assert mgr.available_vram_mb == 8_000

    def test_budget_never_negative(self):
        mgr = _make_manager(total_mb=10_000)
        mgr.register_model("a", 10_000, priority=1)
        mgr.request_load("a")
        assert mgr.available_vram_mb == 0


# ---------------------------------------------------------------------------
# Thread safety smoke test
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_loads_do_not_corrupt_state(self):
        mgr = VRAMManager(total_vram_mb=100_000, max_utilization=1.0)
        for i in range(20):
            mgr.register_model(f"model_{i}", 1_000, priority=2)

        errors: list[Exception] = []

        def load_model(name: str):
            try:
                mgr.request_load(name)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=load_model, args=(f"model_{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Total used must not exceed budget.
        assert mgr.available_vram_mb >= 0
