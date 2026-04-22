"""Audit: list every qdrant read-side payload key reference in src/.

Walks every .py file under src/ (excluding the write path), finds
FieldCondition(key="..."), payload.get("..."), and payload["..."]
references, and prints a flat list sorted by frequency. Any key not
present in the writer's output (build_enriched_payload) is a suspect
for mismatch.

One-shot audit used during Batch 0 cleanup. Safe to re-run.
"""
from __future__ import annotations

import pathlib
import re
import sys
from collections import Counter

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WRITE_PATH_DIRS = {
    "src/embedding/",
    "src/embed/",
    "src/extraction/",
    "src/tasks/",
    "src/api/extraction_service.py",
    "src/api/extraction_pipeline_api.py",
    "src/api/embedding_service.py",
    "src/api/dw_document_extractor.py",
    "src/api/qdrant_indexes.py",
    "src/api/qdrant_setup.py",
    "src/api/vector_store.py",
}

# Keys the writer (build_enriched_payload) emits as of b0c7211.
WRITER_FLAT_KEYS = {
    "subscription_id", "profile_id", "document_id",
    "chunk_id", "resolution", "chunk_kind",
    "section_id", "section_kind", "page",
    "source_name", "doc_domain", "embed_pipeline_version",
    # Enrichment (also top-level)
    "entities", "entity_types", "domain_tags", "doc_category",
    "importance_score", "kg_node_ids", "quality_grade", "text",
    # Nested objects (legacy-compatible)
    "chunk", "section", "provenance",
}

PATTERNS = [
    re.compile(r'FieldCondition\s*\(\s*key\s*=\s*["\']([^"\']+)["\']'),
    re.compile(r'payload\.get\s*\(\s*["\']([^"\']+)["\']'),
    re.compile(r'payload\s*\[\s*["\']([^"\']+)["\']\s*\]'),
]


def is_write_path(relpath: str) -> bool:
    return any(relpath.startswith(p) for p in WRITE_PATH_DIRS)


def main():
    references: Counter[str] = Counter()
    per_key: dict[str, list[str]] = {}
    for path in (REPO_ROOT / "src").rglob("*.py"):
        rel = str(path.relative_to(REPO_ROOT))
        if is_write_path(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pat in PATTERNS:
                for m in pat.finditer(line):
                    key = m.group(1)
                    references[key] += 1
                    per_key.setdefault(key, []).append(f"{rel}:{line_no}")

    unknown = sorted(k for k in references if k not in WRITER_FLAT_KEYS)
    known = sorted(k for k in references if k in WRITER_FLAT_KEYS)

    print("=== Read-side payload key audit ===")
    print(f"Scanned src/ (write path excluded). Writer-known keys used:")
    for k in known:
        print(f"  {k:35s} {references[k]:4d}x")
    print()
    print(f"Keys NOT emitted by the current writer (potential mismatch):")
    if not unknown:
        print("  (none)")
        return 0
    for k in unknown:
        print(f"  {k!r} - {references[k]}x")
        for loc in per_key[k][:10]:
            print(f"      {loc}")
        if len(per_key[k]) > 10:
            print(f"      ...and {len(per_key[k]) - 10} more")
    print()
    print(f"ACTION: for each 'unknown' key, either rewrite the caller to use")
    print(f"a writer-known key, or prove the caller is defensive (has a")
    print(f"fallback to a writer-known key).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
