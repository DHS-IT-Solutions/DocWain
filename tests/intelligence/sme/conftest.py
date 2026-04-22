"""Shared in-memory fakes for SME Phase 1 integration testing.

These fakes implement the structural protocols every SME module takes so a
Phase 1 sandbox integration run can exercise the full seam set (adapter
loader → builders → verifier → storage → trace) without external services.
Phase 2 swaps these for the real Azure Blob / Qdrant / Neo4j adapters.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


class InMemoryBlob:
    """Blob backend covering read / write / delete / append in one class.

    Read + write satisfy :class:`src.intelligence.sme.adapter_loader.BlobReader`
    and :class:`src.intelligence.sme.storage.BlobStore`; append satisfies
    :class:`src.intelligence.sme.trace.TraceBlobAppender`. One instance can
    therefore sit at every Blob seam in the integration test.
    """

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def read_text(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def write_text(self, path: str, content: str) -> None:
        self.files[path] = content

    def delete(self, path: str) -> None:
        self.files.pop(path, None)

    def append(self, path: str, line: str) -> None:
        self.files[path] = self.files.get(path, "") + line


class InMemoryQdrant:
    """In-memory Qdrant bridge covering the Phase 1 write + dummy search API.

    Points are stored per collection. ``delete_by_filter`` clears the whole
    collection for simplicity — Phase 2 tests that need fine-grained filter
    behavior should use a more sophisticated double.
    """

    def __init__(self) -> None:
        self.points: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def upsert_points(
        self, *, collection: str, points: list[dict[str, Any]]
    ) -> None:
        self.points[collection].extend(points)

    def delete_by_filter(
        self, *, collection: str, filter: dict[str, Any]
    ) -> None:
        self.points[collection] = []

    def search_dense(self, **_: Any) -> list[dict[str, Any]]:
        return []

    def search_sparse(self, **_: Any) -> list[dict[str, Any]]:
        return []


class InMemoryNeo4j:
    """In-memory Neo4j bridge collecting inferred-edge writes."""

    def __init__(self) -> None:
        self.edges: list[dict[str, Any]] = []

    def write_inferred_edges(self, edges: list[dict[str, Any]]) -> None:
        self.edges.extend(edges)
