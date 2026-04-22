"""Lean consumer-side tests for the ``enable_rich_mode`` flag.

ERRATA §4 makes ``src/config/feature_flags.py`` canonical — Phase 4 is a
consumer only. These tests pin the consumer semantics the Phase 4 wiring
relies on: default off, dependent on master, per-subscription override.
"""
from unittest.mock import MagicMock

from src.config.feature_flags import (
    ENABLE_RICH_MODE,
    FlagStore,
    SME_REDESIGN_ENABLED,
    SMEFeatureFlags,
)


def _flags(overrides: dict[str, bool]) -> SMEFeatureFlags:
    store = MagicMock(spec=FlagStore)
    store.get_subscription_overrides.return_value = overrides
    return SMEFeatureFlags(store=store)


def test_defaults_to_false_when_nothing_configured():
    assert _flags({}).is_enabled("sub_any", ENABLE_RICH_MODE) is False


def test_master_flag_off_forces_false_even_with_override():
    assert (
        _flags({ENABLE_RICH_MODE: True}).is_enabled("sub_a", ENABLE_RICH_MODE)
        is False
    )


def test_master_plus_override_returns_true():
    assert (
        _flags(
            {
                SME_REDESIGN_ENABLED: True,
                ENABLE_RICH_MODE: True,
            }
        ).is_enabled("sub_a", ENABLE_RICH_MODE)
        is True
    )


def test_master_off_beats_per_sub_on():
    assert (
        _flags(
            {
                SME_REDESIGN_ENABLED: False,
                ENABLE_RICH_MODE: True,
            }
        ).is_enabled("sub_a", ENABLE_RICH_MODE)
        is False
    )
