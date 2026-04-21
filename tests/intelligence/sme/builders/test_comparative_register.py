"""Phase 1 skeleton test for :class:`ComparativeRegisterBuilder`."""
from __future__ import annotations

from unittest.mock import MagicMock

from src.intelligence.sme.builders.comparative_register import (
    ComparativeRegisterBuilder,
)


def test_skeleton_returns_empty_list() -> None:
    builder = ComparativeRegisterBuilder(ctx=MagicMock())
    assert (
        builder.build(
            subscription_id="s", profile_id="p", adapter=MagicMock(), version=1
        )
        == []
    )
    assert builder.artifact_type == "comparison"
