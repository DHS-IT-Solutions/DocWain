def test_per_doc_writes_only_to_researcher_v2_field():
    """Per spec Section 7.3 + feedback_mongo_status_stability.md, the v2
    task must write ONLY to researcher_v2.* — never to pipeline_status,
    stages.*, or the v1 researcher.* field.
    """
    written_paths = []

    class FakeColl:
        def update_one(self, filter, update, upsert=False):
            for k in (update.get("$set") or {}):
                written_paths.append(k)

    fake_coll = FakeColl()

    from src.tasks.researcher_v2 import write_doc_status

    write_doc_status(
        collection=fake_coll, document_id="D", status="RESEARCHER_V2_COMPLETED",
        adapter_version="generic@1.0", written_count=3,
    )

    assert written_paths
    for path in written_paths:
        assert path.startswith("researcher_v2."), f"forbidden write to {path!r}"
