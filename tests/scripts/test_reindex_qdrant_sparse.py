"""Tests for the ``scripts.reindex_qdrant_sparse`` migration script."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts.reindex_qdrant_sparse import (
    ReindexDeps,
    SparseReindexer,
    _parse_args,
)


@pytest.fixture
def deps() -> ReindexDeps:
    return ReindexDeps(qdrant=MagicMock(), sparse_encoder=MagicMock())


def test_reindex_iterates_one_subscription(deps: ReindexDeps) -> None:
    deps.qdrant.scroll_points.return_value = iter(
        [
            [
                {"id": "p1", "payload": {"text": "hi"}, "vector": [0.0]},
                {"id": "p2", "payload": {"text": "foo"}, "vector": [0.1]},
            ]
        ]
    )
    deps.sparse_encoder.encode_batch.return_value = [
        {"indices": [1], "values": [0.3]},
        {"indices": [2], "values": [0.9]},
    ]
    updated = SparseReindexer(deps).reindex_subscription("sub_a")
    assert updated == 2
    deps.qdrant.update_points_sparse.assert_called_once()
    kwargs = deps.qdrant.update_points_sparse.call_args.kwargs
    assert kwargs["collection"] == "sub_a"
    assert [u["id"] for u in kwargs["point_updates"]] == ["p1", "p2"]
    assert all(
        u["payload_patch"] == {"has_sparse": True} for u in kwargs["point_updates"]
    )


def test_empty_collection_is_zero(deps: ReindexDeps) -> None:
    deps.qdrant.scroll_points.return_value = iter([])
    assert SparseReindexer(deps).reindex_subscription("sub_a") == 0
    deps.qdrant.update_points_sparse.assert_not_called()


def test_skips_points_with_has_sparse_flag(deps: ReindexDeps) -> None:
    deps.qdrant.scroll_points.return_value = iter(
        [
            [
                {
                    "id": "p1",
                    "payload": {"text": "t", "has_sparse": True},
                    "vector": [0.0],
                }
            ]
        ]
    )
    assert SparseReindexer(deps).reindex_subscription("sub_a") == 0
    deps.qdrant.update_points_sparse.assert_not_called()
    deps.sparse_encoder.encode_batch.assert_not_called()


def test_dry_run_reports_count_without_writes(deps: ReindexDeps) -> None:
    deps.qdrant.scroll_points.return_value = iter(
        [
            [
                {"id": "p1", "payload": {"text": "a"}, "vector": [0.0]},
                {"id": "p2", "payload": {"text": "b"}, "vector": [0.1]},
            ]
        ]
    )
    updated = SparseReindexer(deps).reindex_subscription("sub_a", dry_run=True)
    assert updated == 2
    deps.qdrant.update_points_sparse.assert_not_called()
    deps.sparse_encoder.encode_batch.assert_not_called()


def test_partial_batch_mixes_skipped_and_updated(deps: ReindexDeps) -> None:
    deps.qdrant.scroll_points.return_value = iter(
        [
            [
                {
                    "id": "p1",
                    "payload": {"text": "done", "has_sparse": True},
                    "vector": [0.0],
                },
                {"id": "p2", "payload": {"text": "todo"}, "vector": [0.1]},
            ]
        ]
    )
    deps.sparse_encoder.encode_batch.return_value = [
        {"indices": [5], "values": [0.5]}
    ]
    updated = SparseReindexer(deps).reindex_subscription("sub_a")
    assert updated == 1
    kwargs = deps.qdrant.update_points_sparse.call_args.kwargs
    assert [u["id"] for u in kwargs["point_updates"]] == ["p2"]


def test_multiple_batches(deps: ReindexDeps) -> None:
    deps.qdrant.scroll_points.return_value = iter(
        [
            [{"id": "a", "payload": {"text": "x"}, "vector": [0.0]}],
            [{"id": "b", "payload": {"text": "y"}, "vector": [0.1]}],
        ]
    )
    deps.sparse_encoder.encode_batch.side_effect = [
        [{"indices": [1], "values": [0.5]}],
        [{"indices": [2], "values": [0.5]}],
    ]
    assert SparseReindexer(deps).reindex_subscription("sub_a") == 2
    assert deps.qdrant.update_points_sparse.call_count == 2


def test_cli_parses_subscription_and_dry_run() -> None:
    ns = _parse_args(["--subscription", "sub_a", "--dry-run"])
    assert ns.subscription == ["sub_a"]
    assert ns.dry_run is True
    assert ns.all is False


def test_cli_supports_all_flag() -> None:
    ns = _parse_args(["--all"])
    assert ns.all is True
    assert ns.subscription == []


def test_cli_supports_multiple_subscriptions() -> None:
    ns = _parse_args(["--subscription", "a", "--subscription", "b"])
    assert ns.subscription == ["a", "b"]
