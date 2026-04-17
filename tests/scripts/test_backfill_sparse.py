"""Lock the behaviour of scripts/backfill_sparse_vectors.py."""
from unittest.mock import MagicMock

from scripts import backfill_sparse_vectors as bf


def _make_collection(name: str):
    col = MagicMock()
    col.name = name
    return col


def test_inventory_mode_lists_collections_and_counts():
    fake_client = MagicMock()
    fake_client.get_collections.return_value.collections = [
        _make_collection("sub_aaa"),
        _make_collection("sub_bbb"),
    ]
    fake_client.count.side_effect = [
        MagicMock(count=1000),
        MagicMock(count=42),
    ]

    inventory = bf.inventory_collections(fake_client)

    assert inventory == [
        {"collection": "sub_aaa", "point_count": 1000},
        {"collection": "sub_bbb", "point_count": 42},
    ]


def test_inventory_mode_filters_by_subscription():
    fake_client = MagicMock()
    fake_client.get_collections.return_value.collections = [
        _make_collection("sub_aaa"),
        _make_collection("sub_bbb"),
    ]
    fake_client.count.return_value = MagicMock(count=1000)

    inventory = bf.inventory_collections(fake_client, subscription_id="aaa")

    assert len(inventory) == 1
    assert inventory[0]["collection"] == "sub_aaa"


def test_inventory_handles_count_failure_gracefully():
    """A bad collection shouldn't abort the whole inventory."""
    fake_client = MagicMock()
    fake_client.get_collections.return_value.collections = [
        _make_collection("sub_aaa"),
        _make_collection("sub_bbb"),
    ]
    fake_client.count.side_effect = [
        Exception("Qdrant 503 for aaa"),
        MagicMock(count=7),
    ]

    inventory = bf.inventory_collections(fake_client)

    assert len(inventory) == 2
    assert inventory[0]["point_count"] == -1
    assert "error" in inventory[0]
    assert inventory[1]["point_count"] == 7
