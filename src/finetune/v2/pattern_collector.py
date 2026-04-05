"""Pattern and metadata collector for DocWain training data generation.

Harvests anonymized patterns from document processing metadata to guide
synthetic training data generation. NO customer content is extracted —
only structural patterns, document types, entity schemas, and quality signals.

Usage::

    from src.finetune.v2.pattern_collector import collect_patterns
    patterns = collect_patterns(profile_id="...", subscription_id="...")
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentPatterns:
    """Anonymized patterns harvested from document processing metadata."""

    # Document type distribution
    doc_type_counts: Dict[str, int] = field(default_factory=dict)
    # File format distribution
    file_format_counts: Dict[str, int] = field(default_factory=dict)
    # Extraction quality distribution (binned: low/medium/high)
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    # Entity type frequency across all documents
    entity_type_counts: Dict[str, int] = field(default_factory=dict)
    # Entity role patterns (e.g., "person:buyer", "organization:vendor")
    entity_role_patterns: List[str] = field(default_factory=list)
    # Relationship type patterns (e.g., "employs", "contains", "references")
    relationship_types: Dict[str, int] = field(default_factory=dict)
    # Table structure patterns (avg cols, avg rows per doc type)
    table_patterns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Section hierarchy depth distribution
    section_depth_counts: Dict[int, int] = field(default_factory=dict)
    # Domain distribution
    domain_counts: Dict[str, int] = field(default_factory=dict)
    # Answerable topic patterns (anonymized question types)
    topic_categories: Dict[str, int] = field(default_factory=dict)
    # Quality signal patterns
    low_confidence_rate: float = 0.0
    grounding_rate: float = 0.0
    avg_confidence: float = 0.0
    # Total documents analyzed
    total_documents: int = 0

    def to_brief_context(self) -> str:
        """Convert patterns to context string for training data generation briefs."""
        parts = []

        if self.doc_type_counts:
            top_types = sorted(self.doc_type_counts.items(), key=lambda x: -x[1])[:10]
            parts.append(
                "Document type distribution: "
                + ", ".join(f"{t} ({c})" for t, c in top_types)
            )

        if self.entity_type_counts:
            top_entities = sorted(self.entity_type_counts.items(), key=lambda x: -x[1])[:8]
            parts.append(
                "Common entity types: "
                + ", ".join(f"{t} ({c})" for t, c in top_entities)
            )

        if self.entity_role_patterns:
            parts.append(
                "Entity role patterns: " + ", ".join(self.entity_role_patterns[:15])
            )

        if self.relationship_types:
            top_rels = sorted(self.relationship_types.items(), key=lambda x: -x[1])[:8]
            parts.append(
                "Relationship types: "
                + ", ".join(f"{t} ({c})" for t, c in top_rels)
            )

        if self.domain_counts:
            parts.append(
                "Domain distribution: "
                + ", ".join(f"{d} ({c})" for d, c in sorted(self.domain_counts.items(), key=lambda x: -x[1]))
            )

        if self.table_patterns:
            parts.append(
                "Table patterns: "
                + ", ".join(
                    f"{dt}: avg {p.get('avg_cols', 0):.0f} cols × {p.get('avg_rows', 0):.0f} rows"
                    for dt, p in self.table_patterns.items()
                )
            )

        if self.low_confidence_rate > 0:
            parts.append(
                f"Quality signals: {self.low_confidence_rate:.1%} low-confidence, "
                f"{self.grounding_rate:.1%} grounded, avg confidence {self.avg_confidence:.2f}"
            )

        parts.append(f"Total documents: {self.total_documents}")

        return "\n".join(parts)


def collect_patterns_from_mongodb(
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    max_docs: int = 1000,
) -> DocumentPatterns:
    """Collect anonymized patterns from MongoDB document metadata.

    Only reads metadata fields — never accesses document content, text,
    or any customer data.
    """
    try:
        from src.api.config import Config
        from pymongo import MongoClient

        client = MongoClient(Config.MongoDB.URI)
        db = client[Config.MongoDB.DATABASE]
        collection = db[Config.MongoDB.DOCUMENTS]
    except Exception as exc:
        logger.warning("MongoDB not available for pattern collection: %s", exc)
        return DocumentPatterns()

    query: Dict[str, Any] = {}
    if subscription_id:
        query["subscription_id"] = subscription_id
    if profile_id:
        query["profile_id"] = profile_id

    # Only project metadata fields — no content
    projection = {
        "_id": 0,
        "doc_type": 1,
        "content_type": 1,
        "extraction.summary.doc_type_detected": 1,
        "extraction.summary.extraction_confidence": 1,
        "extraction.summary.page_count": 1,
        "intelligence.document_type": 1,
        "intelligence.entities": 1,
        "intelligence.relationships": 1,
        "intelligence.answerable_topics": 1,
        "knowledge_graph.node_count": 1,
        "knowledge_graph.edge_count": 1,
    }

    patterns = DocumentPatterns()
    doc_type_counter: Counter = Counter()
    format_counter: Counter = Counter()
    quality_counter: Counter = Counter()
    entity_counter: Counter = Counter()
    role_patterns: set = set()
    rel_counter: Counter = Counter()
    domain_counter: Counter = Counter()
    topic_counter: Counter = Counter()
    confidences: List[float] = []

    cursor = collection.find(query, projection=projection).limit(max_docs)

    for doc in cursor:
        patterns.total_documents += 1

        # Document type
        doc_type = (
            doc.get("intelligence", {}).get("document_type")
            or doc.get("doc_type", "unknown")
        )
        doc_type_counter[doc_type] += 1

        # File format
        content_type = doc.get("content_type", "unknown")
        fmt = content_type.split("/")[-1] if "/" in content_type else content_type
        format_counter[fmt] += 1

        # Extraction quality
        ext_summary = doc.get("extraction", {}).get("summary", {})
        confidence = ext_summary.get("extraction_confidence", 0)
        if confidence:
            confidences.append(confidence)
            if confidence >= 0.8:
                quality_counter["high"] += 1
            elif confidence >= 0.5:
                quality_counter["medium"] += 1
            else:
                quality_counter["low"] += 1

        # Entities (type + role only, no values)
        intel = doc.get("intelligence", {})
        for entity in intel.get("entities", []):
            etype = entity.get("type", "unknown")
            entity_counter[etype] += 1
            role = entity.get("role", "")
            if role:
                role_patterns.add(f"{etype}:{role}")

        # Relationships (type only)
        for rel in intel.get("relationships", []):
            rel_type = rel.get("relation_type", "unknown")
            rel_counter[rel_type] += 1

        # Domain classification
        _DOMAIN_MAP = {
            "resume": "hr", "cv": "hr", "job_description": "hr",
            "invoice": "finance", "purchase_order": "finance",
            "receipt": "finance", "financial_statement": "finance",
            "contract": "legal", "agreement": "legal", "nda": "legal",
            "lease": "legal", "medical_record": "medical",
            "prescription": "medical", "lab_report": "medical",
        }
        domain = _DOMAIN_MAP.get(doc_type, "general")
        domain_counter[domain] += 1

        # Answerable topics (category only)
        for topic in intel.get("answerable_topics", []):
            # Categorize without storing actual topic text
            if any(w in str(topic).lower() for w in ["date", "when", "time"]):
                topic_counter["temporal"] += 1
            elif any(w in str(topic).lower() for w in ["amount", "price", "cost", "total"]):
                topic_counter["financial"] += 1
            elif any(w in str(topic).lower() for w in ["who", "name", "person"]):
                topic_counter["entity_lookup"] += 1
            elif any(w in str(topic).lower() for w in ["compare", "difference", "vs"]):
                topic_counter["comparison"] += 1
            else:
                topic_counter["general"] += 1

    patterns.doc_type_counts = dict(doc_type_counter)
    patterns.file_format_counts = dict(format_counter)
    patterns.quality_distribution = dict(quality_counter)
    patterns.entity_type_counts = dict(entity_counter)
    patterns.entity_role_patterns = sorted(role_patterns)[:50]
    patterns.relationship_types = dict(rel_counter)
    patterns.domain_counts = dict(domain_counter)
    patterns.topic_categories = dict(topic_counter)

    if confidences:
        patterns.avg_confidence = sum(confidences) / len(confidences)
        patterns.low_confidence_rate = sum(1 for c in confidences if c < 0.5) / len(confidences)
        patterns.grounding_rate = sum(1 for c in confidences if c >= 0.8) / len(confidences)

    logger.info(
        "Collected patterns from %d documents: %d doc types, %d entity types, %d relationships",
        patterns.total_documents,
        len(patterns.doc_type_counts),
        len(patterns.entity_type_counts),
        len(patterns.relationship_types),
    )
    return patterns


def collect_feedback_signals() -> Dict[str, Any]:
    """Collect quality signals from the learning signals log.

    Reads the JSONL learning signals to identify patterns in
    model failures and successes — without accessing query content.
    """
    signals_dir = Path("outputs/learning_signals")
    if not signals_dir.exists():
        return {"total_signals": 0}

    signal_types: Counter = Counter()
    confidence_bins: Counter = Counter()
    failure_categories: Counter = Counter()

    for jsonl_file in signals_dir.glob("*.jsonl"):
        try:
            with open(jsonl_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        stype = entry.get("signal_type", "unknown")
                        signal_types[stype] += 1

                        confidence = entry.get("confidence", 0)
                        if confidence >= 0.8:
                            confidence_bins["high"] += 1
                        elif confidence >= 0.5:
                            confidence_bins["medium"] += 1
                        else:
                            confidence_bins["low"] += 1

                        if stype == "failure":
                            cat = entry.get("failure_category", "unknown")
                            failure_categories[cat] += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.warning("Error reading signals file %s: %s", jsonl_file, exc)

    return {
        "total_signals": sum(signal_types.values()),
        "signal_types": dict(signal_types),
        "confidence_distribution": dict(confidence_bins),
        "failure_categories": dict(failure_categories),
    }


def save_patterns(patterns: DocumentPatterns, path: Path) -> None:
    """Save collected patterns to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "doc_type_counts": patterns.doc_type_counts,
        "file_format_counts": patterns.file_format_counts,
        "quality_distribution": patterns.quality_distribution,
        "entity_type_counts": patterns.entity_type_counts,
        "entity_role_patterns": patterns.entity_role_patterns,
        "relationship_types": patterns.relationship_types,
        "table_patterns": patterns.table_patterns,
        "section_depth_counts": {str(k): v for k, v in patterns.section_depth_counts.items()},
        "domain_counts": patterns.domain_counts,
        "topic_categories": patterns.topic_categories,
        "low_confidence_rate": patterns.low_confidence_rate,
        "grounding_rate": patterns.grounding_rate,
        "avg_confidence": patterns.avg_confidence,
        "total_documents": patterns.total_documents,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved patterns to %s", path)


def load_patterns(path: Path) -> DocumentPatterns:
    """Load patterns from a JSON file."""
    if not path.exists():
        return DocumentPatterns()
    data = json.loads(path.read_text(encoding="utf-8"))
    return DocumentPatterns(
        doc_type_counts=data.get("doc_type_counts", {}),
        file_format_counts=data.get("file_format_counts", {}),
        quality_distribution=data.get("quality_distribution", {}),
        entity_type_counts=data.get("entity_type_counts", {}),
        entity_role_patterns=data.get("entity_role_patterns", []),
        relationship_types=data.get("relationship_types", {}),
        table_patterns=data.get("table_patterns", {}),
        section_depth_counts={int(k): v for k, v in data.get("section_depth_counts", {}).items()},
        domain_counts=data.get("domain_counts", {}),
        topic_categories=data.get("topic_categories", {}),
        low_confidence_rate=data.get("low_confidence_rate", 0.0),
        grounding_rate=data.get("grounding_rate", 0.0),
        avg_confidence=data.get("avg_confidence", 0.0),
        total_documents=data.get("total_documents", 0),
    )
