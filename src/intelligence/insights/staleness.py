"""Stale-flag updater.

When a document changes, every insight that includes that document_id
in its document_ids list is marked stale. Surface layer renders stale
insights with a "refreshing..." indicator until they re-run.

Per spec Section 9.3 — never silently show outdated data without flagging.
"""
from __future__ import annotations

from typing import Iterable


def mark_stale_for_documents(*, collection, profile_id: str, document_ids: Iterable[str]) -> int:
    """Mark every insight whose document_ids include any of the listed docs."""
    doc_list = list(document_ids)
    if not doc_list:
        return 0
    result = collection.update_many(
        {"profile_id": profile_id, "document_ids": {"$in": doc_list}},
        {"$set": {"stale": True}},
    )
    return getattr(result, "modified_count", 0)
