"""Phase 1 skeleton test for :class:`RecommendationBankBuilder`."""
from __future__ import annotations

from unittest.mock import MagicMock

from src.intelligence.sme.builders.recommendation_bank import (
    RecommendationBankBuilder,
)


def test_skeleton_returns_empty_list() -> None:
    builder = RecommendationBankBuilder(ctx=MagicMock())
    assert (
        builder.build(
            subscription_id="s", profile_id="p", adapter=MagicMock(), version=1
        )
        == []
    )
    assert builder.artifact_type == "recommendation"
