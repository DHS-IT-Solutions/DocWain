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


def test_process_collection_dry_run_encodes_but_does_not_upsert():
    fake_client = MagicMock()
    pt1 = MagicMock()
    pt1.id = "p1"
    pt1.payload = {"canonical_text": "invoice total $1000"}
    pt2 = MagicMock()
    pt2.id = "p2"
    pt2.payload = {"canonical_text": "contract agreement"}
    # Scroll returns (points, next_offset). next_offset=None → stop.
    fake_client.scroll.return_value = ([pt1, pt2], None)

    fake_encoder = MagicMock()
    fake_encoder.encode_batch.return_value = [
        {"indices": [1, 2], "values": [0.5, 0.7]},
        {"indices": [3, 4], "values": [0.8, 0.2]},
    ]

    stats = bf.process_collection(
        fake_client, "sub_aaa", fake_encoder, batch_size=64, dry_run=True
    )

    assert stats["encoded"] == 2
    assert stats["upserted"] == 0
    fake_encoder.encode_batch.assert_called_once()
    fake_client.update_vectors.assert_not_called()
    fake_client.set_payload.assert_not_called()


def test_process_collection_full_run_upserts_and_marks():
    fake_client = MagicMock()
    pt1 = MagicMock()
    pt1.id = "p1"
    pt1.payload = {"canonical_text": "some text"}
    fake_client.scroll.return_value = ([pt1], None)

    fake_encoder = MagicMock()
    fake_encoder.encode_batch.return_value = [
        {"indices": [1, 2], "values": [0.5, 0.7]},
    ]

    stats = bf.process_collection(
        fake_client, "sub_aaa", fake_encoder, batch_size=64, dry_run=False
    )

    assert stats["encoded"] == 1
    assert stats["upserted"] == 1

    # update_vectors should have been called with a list of PointVectors-like objects
    fake_client.update_vectors.assert_called_once()
    call = fake_client.update_vectors.call_args
    assert call.kwargs["collection_name"] == "sub_aaa"
    upsert_points = call.kwargs["points"]
    assert len(upsert_points) == 1
    # PointVectors has .id and .vector attrs
    assert upsert_points[0].id == "p1"
    assert "keywords_vector" in upsert_points[0].vector

    # set_payload should have been called with the marker for p1
    fake_client.set_payload.assert_called_once()
    sp_call = fake_client.set_payload.call_args
    assert "sparse_backfilled_at" in sp_call.kwargs["payload"]
    assert sp_call.kwargs["points"] == ["p1"]


def test_process_collection_skips_points_with_empty_text():
    fake_client = MagicMock()
    pt_good = MagicMock()
    pt_good.id = "p1"
    pt_good.payload = {"canonical_text": "good text"}
    pt_empty = MagicMock()
    pt_empty.id = "p2"
    pt_empty.payload = {"canonical_text": ""}
    fake_client.scroll.return_value = ([pt_good, pt_empty], None)

    fake_encoder = MagicMock()
    fake_encoder.encode_batch.return_value = [
        {"indices": [1], "values": [0.5]},
    ]

    stats = bf.process_collection(
        fake_client, "sub_aaa", fake_encoder, batch_size=64, dry_run=True
    )

    assert stats["encoded"] == 1
    assert stats["skipped"] == 1
    # Encoder got only the good point's text
    fake_encoder.encode_batch.assert_called_once_with(["good text"])


def test_process_collection_continues_on_encode_failure():
    fake_client = MagicMock()
    pt1 = MagicMock()
    pt1.id = "p1"
    pt1.payload = {"canonical_text": "text"}
    fake_client.scroll.return_value = ([pt1], None)

    fake_encoder = MagicMock()
    fake_encoder.encode_batch.side_effect = RuntimeError("splade blew up")

    stats = bf.process_collection(
        fake_client, "sub_aaa", fake_encoder, batch_size=64, dry_run=False
    )

    assert stats["errors"] == 1
    assert stats["upserted"] == 0
    fake_client.update_vectors.assert_not_called()
