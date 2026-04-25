from src.intelligence.insights.staleness import mark_stale_for_documents


class FakeColl:
    def __init__(self, docs):
        self.docs = docs

    def update_many(self, query, update):
        n = 0
        for d in self.docs:
            ok = True
            for k, v in query.items():
                if isinstance(v, dict) and "$in" in v:
                    field_value = d.get(k)
                    # Mongo $in on array fields: match if ANY element overlaps
                    if isinstance(field_value, list):
                        if not any(item in v["$in"] for item in field_value):
                            ok = False
                            break
                    else:
                        if field_value not in v["$in"]:
                            ok = False
                            break
                elif d.get(k) != v:
                    ok = False
                    break
            if ok:
                d.update(update.get("$set", {}))
                n += 1
        return type("R", (), {"modified_count": n})()


def test_mark_stale_flags_only_affected_insights():
    docs = [
        {"insight_id": "i-A", "profile_id": "P1", "document_ids": ["DOC-X"], "stale": False},
        {"insight_id": "i-B", "profile_id": "P1", "document_ids": ["DOC-Y"], "stale": False},
        {"insight_id": "i-C", "profile_id": "P1", "document_ids": ["DOC-X", "DOC-Z"], "stale": False},
    ]
    coll = FakeColl(docs)
    n = mark_stale_for_documents(
        collection=coll, profile_id="P1", document_ids=["DOC-X"]
    )
    assert n == 2
    assert docs[0]["stale"] is True
    assert docs[1]["stale"] is False
    assert docs[2]["stale"] is True
