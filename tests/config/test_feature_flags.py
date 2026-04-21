"""Tests for :class:`SMEFeatureFlags` resolution."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config.feature_flags import (
    ENABLE_CROSS_ENCODER_RERANK,
    ENABLE_HYBRID_RETRIEVAL,
    ENABLE_KG_SYNTHESIZED_EDGES,
    ENABLE_RICH_MODE,
    ENABLE_SME_RETRIEVAL,
    ENABLE_SME_SYNTHESIS,
    ENABLE_URL_AS_PROMPT,
    SME_REDESIGN_ENABLED,
    FlagStore,
    SMEFeatureFlags,
    get_flag_resolver,
    init_flag_resolver,
)


ALL_FLAGS = (
    SME_REDESIGN_ENABLED,
    ENABLE_SME_SYNTHESIS,
    ENABLE_SME_RETRIEVAL,
    ENABLE_KG_SYNTHESIZED_EDGES,
    ENABLE_RICH_MODE,
    ENABLE_URL_AS_PROMPT,
    ENABLE_HYBRID_RETRIEVAL,
    ENABLE_CROSS_ENCODER_RERANK,
)


@pytest.fixture
def store() -> MagicMock:
    s = MagicMock(spec=FlagStore)
    s.get_subscription_overrides.return_value = {}
    return s


def test_exports_exactly_eight_flag_constants() -> None:
    # Guard against drift: the eight canonical names must be stable.
    assert SME_REDESIGN_ENABLED == "sme_redesign_enabled"
    assert ENABLE_SME_SYNTHESIS == "enable_sme_synthesis"
    assert ENABLE_SME_RETRIEVAL == "enable_sme_retrieval"
    assert ENABLE_KG_SYNTHESIZED_EDGES == "enable_kg_synthesized_edges"
    assert ENABLE_RICH_MODE == "enable_rich_mode"
    assert ENABLE_URL_AS_PROMPT == "enable_url_as_prompt"
    assert ENABLE_HYBRID_RETRIEVAL == "enable_hybrid_retrieval"
    assert ENABLE_CROSS_ENCODER_RERANK == "enable_cross_encoder_rerank"


def test_all_default_off(store: MagicMock) -> None:
    f = SMEFeatureFlags(store=store)
    for flag in ALL_FLAGS:
        assert f.is_enabled("sub_a", flag) is False


def test_master_gates_dependents_off(store: MagicMock) -> None:
    # Master off → dependent off even if override turns it on.
    store.get_subscription_overrides.return_value = {
        ENABLE_SME_SYNTHESIS: True,
        ENABLE_SME_RETRIEVAL: True,
        ENABLE_RICH_MODE: True,
        ENABLE_KG_SYNTHESIZED_EDGES: True,
        ENABLE_URL_AS_PROMPT: True,
    }
    f = SMEFeatureFlags(store=store)
    assert f.is_enabled("sub_a", ENABLE_SME_SYNTHESIS) is False
    assert f.is_enabled("sub_a", ENABLE_SME_RETRIEVAL) is False
    assert f.is_enabled("sub_a", ENABLE_RICH_MODE) is False
    assert f.is_enabled("sub_a", ENABLE_KG_SYNTHESIZED_EDGES) is False
    assert f.is_enabled("sub_a", ENABLE_URL_AS_PROMPT) is False


def test_master_on_unlocks_dependents(store: MagicMock) -> None:
    store.get_subscription_overrides.return_value = {
        SME_REDESIGN_ENABLED: True,
        ENABLE_SME_SYNTHESIS: True,
    }
    f = SMEFeatureFlags(store=store)
    assert f.is_enabled("sub_a", SME_REDESIGN_ENABLED) is True
    assert f.is_enabled("sub_a", ENABLE_SME_SYNTHESIS) is True
    # Dependents not overridden remain default-off.
    assert f.is_enabled("sub_a", ENABLE_SME_RETRIEVAL) is False


def test_infrastructure_flags_bypass_master(store: MagicMock) -> None:
    # Spec §13.5: hybrid + cross-encoder survive master rollback.
    store.get_subscription_overrides.return_value = {
        ENABLE_HYBRID_RETRIEVAL: True,
        ENABLE_CROSS_ENCODER_RERANK: True,
    }
    f = SMEFeatureFlags(store=store)
    assert f.is_enabled("sub_a", ENABLE_HYBRID_RETRIEVAL) is True
    assert f.is_enabled("sub_a", ENABLE_CROSS_ENCODER_RERANK) is True


def test_unknown_flag_raises(store: MagicMock) -> None:
    f = SMEFeatureFlags(store=store)
    with pytest.raises(KeyError, match="unknown"):
        f.is_enabled("sub_a", "totally_made_up")


def test_subscription_scope(store: MagicMock) -> None:
    def overrides(sub_id: str) -> dict[str, bool]:
        if sub_id == "with_sme":
            return {
                SME_REDESIGN_ENABLED: True,
                ENABLE_SME_SYNTHESIS: True,
            }
        return {}

    store.get_subscription_overrides.side_effect = overrides
    f = SMEFeatureFlags(store=store)
    assert f.is_enabled("with_sme", ENABLE_SME_SYNTHESIS) is True
    assert f.is_enabled("other", ENABLE_SME_SYNTHESIS) is False


def test_get_flag_resolver_before_init_raises() -> None:
    # Reset the module-level singleton to cover the uninitialized path.
    import src.config.feature_flags as ff_mod

    ff_mod._flag_resolver_singleton = None
    with pytest.raises(RuntimeError, match="init_flag_resolver"):
        get_flag_resolver()


def test_init_and_get_flag_resolver(store: MagicMock) -> None:
    resolver = init_flag_resolver(store=store)
    assert get_flag_resolver() is resolver


def test_init_flag_resolver_replaces_singleton(store: MagicMock) -> None:
    first = init_flag_resolver(store=store)
    second_store = MagicMock(spec=FlagStore)
    second_store.get_subscription_overrides.return_value = {}
    second = init_flag_resolver(store=second_store)
    assert first is not second
    assert get_flag_resolver() is second
