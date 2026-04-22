"""Orchestrator: harvest Qdrant chunks and distil SFT / DPO training data.

Usage::

    PYTHONPATH=. python scripts/distill_from_teacher.py --max-chunks 1000
    PYTHONPATH=. python scripts/distill_from_teacher.py --max-chunks 50 --skip-merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from qdrant_client import QdrantClient

from src.api.config import Config
from src.finetune.distillation.generators import (
    generate_all_categories,
    generate_crossdoc_examples,
    generate_dpo_pairs,
)

log = logging.getLogger(__name__)

TEACHER_DIR = Path("finetune_artifacts/teacher_data")
EXISTING_SFT = Path("finetune_artifacts/weekend_loop/master_sft.jsonl")
EXISTING_DPO = Path("finetune_artifacts/weekend_loop/master_dpo.jsonl")


# ---------------------------------------------------------------------------
# 1. Harvest
# ---------------------------------------------------------------------------

def harvest_qdrant_chunks(max_chunks: int = 1000) -> List[Dict]:
    """Connect to Qdrant, enumerate collections, scroll chunks."""
    client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
    collections = [c.name for c in client.get_collections().collections]
    log.info("Found %d Qdrant collections: %s", len(collections), collections)

    chunks: List[Dict] = []
    for coll in collections:
        if len(chunks) >= max_chunks:
            break
        offset = None
        while len(chunks) < max_chunks:
            result, next_offset = client.scroll(
                collection_name=coll,
                limit=min(100, max_chunks - len(chunks)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not result:
                break
            for point in result:
                p = point.payload or {}
                text = (
                    p.get("canonical_text")
                    or p.get("embedding_text")
                    or p.get("content")
                    or ""
                )
                if len(text.strip()) < 50:
                    continue
                chunks.append({
                    "text": text.strip(),
                    "doc_type": p.get("doc_domain", "generic"),
                    "source_name": p.get("source_name", "unknown"),
                    "collection": coll,
                    "profile_id": p.get("profile_id", ""),
                    "document_id": p.get("document_id", ""),
                })
                if len(chunks) >= max_chunks:
                    break
            offset = next_offset
            if offset is None:
                break

    log.info("Harvested %d chunks from %d collections", len(chunks), len(collections))
    return chunks


# ---------------------------------------------------------------------------
# 2. Generate
# ---------------------------------------------------------------------------

def generate_from_chunks(chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Run distillation generators over harvested chunks."""
    sft_all: List[Dict] = []
    dpo_all: List[Dict] = []

    for i, c in enumerate(chunks):
        log.debug("Generating from chunk %d/%d (%s)", i + 1, len(chunks), c["source_name"])
        sft_all.extend(generate_all_categories(c["text"], c["doc_type"], c["source_name"]))
        dpo_all.extend(generate_dpo_pairs(c["text"], c["doc_type"]))

    # Cross-doc examples grouped by profile
    by_profile: Dict[str, List[Dict]] = defaultdict(list)
    for c in chunks:
        pid = c.get("profile_id")
        if pid:
            by_profile[pid].append(c)

    for pid, group in by_profile.items():
        if len(group) < 2:
            continue
        texts = [g["text"] for g in group]
        types = [g["doc_type"] for g in group]
        names = [g["source_name"] for g in group]
        sft_all.extend(generate_crossdoc_examples(texts, types, names))

    log.info("Generated %d SFT examples, %d DPO pairs", len(sft_all), len(dpo_all))
    return sft_all, dpo_all


# ---------------------------------------------------------------------------
# 3. Deduplicate
# ---------------------------------------------------------------------------

def _hash_example(ex: Dict) -> str:
    """Hash first 500 chars of the most meaningful text field."""
    for key in ("prompt", "text", "query", "instruction"):
        val = ex.get(key, "")
        if val:
            return hashlib.sha256(val[:500].encode()).hexdigest()
    # Fallback: hash the whole serialised dict
    return hashlib.sha256(json.dumps(ex, sort_keys=True)[:500].encode()).hexdigest()


def deduplicate(examples: List[Dict]) -> List[Dict]:
    """Remove duplicates based on first 500 chars of text/prompt."""
    seen: set = set()
    unique: List[Dict] = []
    for ex in examples:
        h = _hash_example(ex)
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    removed = len(examples) - len(unique)
    if removed:
        log.info("Deduplication removed %d items (%.1f%%)", removed, 100 * removed / max(len(examples), 1))
    return unique


# ---------------------------------------------------------------------------
# 4. Merge with existing
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def merge_with_existing(new_examples: List[Dict], existing_path: Path) -> List[Dict]:
    """Load existing JSONL, combine, deduplicate, shuffle deterministically."""
    existing = _load_jsonl(existing_path)
    log.info("Loaded %d existing examples from %s", len(existing), existing_path)
    combined = existing + new_examples
    combined = deduplicate(combined)
    random.seed(42)
    random.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------

def save_jsonl(data: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log.info("Saved %d examples to %s", len(data), path)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Distil training data from Qdrant chunks")
    parser.add_argument("--max-chunks", type=int, default=1000, help="Max chunks to harvest")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merging with existing data")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.time()

    # Harvest
    log.info("=== Harvesting up to %d chunks from Qdrant ===", args.max_chunks)
    chunks = harvest_qdrant_chunks(max_chunks=args.max_chunks)

    # Generate
    log.info("=== Generating training examples ===")
    sft_new, dpo_new = generate_from_chunks(chunks)

    # Deduplicate
    log.info("=== Deduplicating ===")
    sft_new = deduplicate(sft_new)
    dpo_new = deduplicate(dpo_new)

    # Save teacher data
    log.info("=== Saving teacher data ===")
    save_jsonl(sft_new, TEACHER_DIR / "teacher_sft.jsonl")
    save_jsonl(dpo_new, TEACHER_DIR / "teacher_dpo.jsonl")

    # Merge
    if not args.skip_merge:
        log.info("=== Merging with existing data ===")
        merged_sft = merge_with_existing(sft_new, EXISTING_SFT)
        merged_dpo = merge_with_existing(dpo_new, EXISTING_DPO)
        save_jsonl(merged_sft, TEACHER_DIR / "master_v2.jsonl")
        save_jsonl(merged_dpo, TEACHER_DIR / "master_dpo_v2.jsonl")
    else:
        merged_sft = sft_new
        merged_dpo = dpo_new
        log.info("Skipped merge (--skip-merge)")

    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 60)
    print("DISTILLATION SUMMARY")
    print("=" * 60)
    print(f"  Chunks harvested:      {len(chunks)}")
    print(f"  New SFT examples:      {len(sft_new)}")
    print(f"  New DPO pairs:         {len(dpo_new)}")
    if not args.skip_merge:
        print(f"  Merged SFT total:      {len(merged_sft)}")
        print(f"  Merged DPO total:      {len(merged_dpo)}")
    print(f"  Time elapsed:          {elapsed:.1f}s")
    print(f"  Output dir:            {TEACHER_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
