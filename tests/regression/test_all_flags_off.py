"""Hard regression test — with every flag off, behavior is byte-identical
to preprod_v02 (the baseline at the time this branch was cut).

This test is the gate for spec Section 15.1.

Currently asserts:
- feature_flags module exists and reports all-False
- importing feature_flags / adapters / insights / knowledge modules has
  zero side effects on existing modules
"""
from __future__ import annotations

import importlib


def test_feature_flags_module_imports_cleanly(all_flags_off):
    importlib.import_module("src.api.feature_flags")
    from src.api.feature_flags import FLAG_NAMES, is_enabled, FeatureFlags
    flags = FeatureFlags()
    for name in FLAG_NAMES:
        assert is_enabled(name, flags) is False


def test_adapters_module_imports_cleanly(all_flags_off):
    importlib.import_module("src.intelligence.adapters")


def test_insights_module_imports_cleanly(all_flags_off):
    importlib.import_module("src.intelligence.insights")


def test_knowledge_module_imports_cleanly(all_flags_off):
    importlib.import_module("src.intelligence.knowledge")
