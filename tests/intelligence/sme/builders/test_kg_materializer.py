"""Phase 1 skeleton test for :class:`KGMultiHopMaterializer`."""
from __future__ import annotations

from unittest.mock import MagicMock

from src.intelligence.sme.builders.kg_materializer import KGMultiHopMaterializer


def test_skeleton_returns_empty_list() -> None:
    builder = KGMultiHopMaterializer(ctx=MagicMock())
    assert (
        builder.build(
            subscription_id="s", profile_id="p", adapter=MagicMock(), version=1
        )
        == []
    )
    assert builder.artifact_type == "kg_edge"
