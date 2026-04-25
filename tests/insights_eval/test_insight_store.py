import pytest

from src.intelligence.insights.schema import Insight, EvidenceSpan
from src.intelligence.insights.store import InsightStore, MongoIndexBackend
from src.intelligence.insights.validators import CitationViolation


class FakeMongoCollection:
    def __init__(self):
        self.docs = []

    def update_one(self, filter, update, upsert=False):
        match = next(
            (d for d in self.docs if all(d.get(k) == v for k, v in filter.items())),
            None,
        )
        if match is None:
            if upsert:
                d = {**filter, **update.get("$set", {})}
                self.docs.append(d)
                return type("R", (), {"matched_count": 0, "upserted_id": "new"})()
        else:
            match.update(update.get("$set", {}))
            return type("R", (), {"matched_count": 1, "upserted_id": None})()
        return type("R", (), {"matched_count": 0, "upserted_id": None})()

    def find(self, query):
        def matches(d):
            for k, v in query.items():
                if isinstance(v, dict) and "$in" in v:
                    if d.get(k) not in v["$in"]:
                        return False
                elif d.get(k) != v:
                    return False
            return True
        return [d for d in self.docs if matches(d)]


def _insight() -> Insight:
    span = EvidenceSpan(
        document_id="DOC-1", page=1, char_start=0, char_end=22,
        quote="Excludes: flood damage",
    )
    return Insight(
        insight_id="i-1",
        profile_id="p-1",
        subscription_id="s-1",
        document_ids=["DOC-1"],
        domain="insurance",
        insight_type="gap",
        headline="No flood coverage",
        body="The policy excludes flood damage.",
        evidence_doc_spans=[span],
        confidence=0.95,
        severity="warn",
        adapter_version="insurance@1.0",
    )


def test_write_inserts_into_mongo_index():
    coll = FakeMongoCollection()
    backend = MongoIndexBackend(collection=coll)
    store = InsightStore(mongo_index=backend, qdrant=None, neo4j=None)
    store.write(_insight())
    assert len(coll.docs) == 1
    d = coll.docs[0]
    assert d["insight_id"] == "i-1"
    assert d["profile_id"] == "p-1"
    assert d["insight_type"] == "gap"
    assert d["severity"] == "warn"
    assert "body" not in d
    assert "evidence_doc_spans" not in d


def test_write_rejects_zero_evidence():
    coll = FakeMongoCollection()
    backend = MongoIndexBackend(collection=coll)
    store = InsightStore(mongo_index=backend, qdrant=None, neo4j=None)
    insight = _insight()
    insight.evidence_doc_spans = []
    with pytest.raises(CitationViolation):
        store.write(insight)
    assert len(coll.docs) == 0


def test_dedup_key_upsert():
    coll = FakeMongoCollection()
    backend = MongoIndexBackend(collection=coll)
    store = InsightStore(mongo_index=backend, qdrant=None, neo4j=None)
    a = _insight()
    b = _insight()
    b.insight_id = "i-2"
    store.write(a)
    store.write(b)
    assert len(coll.docs) == 1


class FakeQdrant:
    def __init__(self):
        self.points = []
    def upsert_insight(self, *, insight):
        self.points.append(insight.to_dict())


def test_write_persists_to_qdrant():
    coll = FakeMongoCollection()
    qdrant = FakeQdrant()
    store = InsightStore(
        mongo_index=MongoIndexBackend(collection=coll),
        qdrant=qdrant,
        neo4j=None,
    )
    store.write(_insight())
    assert len(qdrant.points) == 1
    assert qdrant.points[0]["headline"] == "No flood coverage"
    assert qdrant.points[0]["evidence_doc_spans"][0]["document_id"] == "DOC-1"


class FakeNeo4j:
    def __init__(self):
        self.calls = []
    def upsert_insight(self, *, insight):
        self.calls.append(insight.to_dict())


def test_write_persists_to_neo4j():
    coll = FakeMongoCollection()
    neo4j = FakeNeo4j()
    store = InsightStore(
        mongo_index=MongoIndexBackend(collection=coll),
        qdrant=None,
        neo4j=neo4j,
    )
    store.write(_insight())
    assert len(neo4j.calls) == 1
    assert neo4j.calls[0]["insight_id"] == "i-1"


def test_list_filters_by_profile():
    coll = FakeMongoCollection()
    backend = MongoIndexBackend(collection=coll)
    store = InsightStore(mongo_index=backend, qdrant=None, neo4j=None)
    a = _insight()
    a.insight_id = "i-A"
    a.profile_id = "P1"
    b = _insight()
    b.insight_id = "i-B"
    b.profile_id = "P2"
    b.headline = "Different"
    store.write(a)
    store.write(b)
    rows = store.list_for_profile(profile_id="P1")
    assert len(rows) == 1
    assert rows[0]["insight_id"] == "i-A"


def test_list_filters_by_insight_type():
    coll = FakeMongoCollection()
    backend = MongoIndexBackend(collection=coll)
    store = InsightStore(mongo_index=backend, qdrant=None, neo4j=None)
    a = _insight()
    a.insight_id = "ia"
    b = _insight()
    b.insight_id = "ib"
    b.insight_type = "anomaly"
    b.headline = "Anomaly headline"
    store.write(a)
    store.write(b)
    rows = store.list_for_profile(profile_id="p-1", insight_types=["anomaly"])
    assert len(rows) == 1
    assert rows[0]["insight_type"] == "anomaly"
