"""Pre-computed Document Intelligence.

Runs during the heavy processing phase (after HITL Gate 2 approval) and stores
structured analysis in the MongoDB ``document_intelligence`` collection.

Each helper extracts a specific facet (temporal, numerical, key-facts, summary)
using regex/heuristics first, then LLM for complex reasoning.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_DATE_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DATE_SLASH = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
_DATE_MONTH_YEAR = re.compile(
    r"\b((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{4})\b",
    re.IGNORECASE,
)
_DATE_QUARTER = re.compile(r"\b(Q[1-4]\s*\d{4})\b", re.IGNORECASE)

_CURRENCY = re.compile(
    r"[\$\£\€]\s*[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|M|B|K|thousand))?",
    re.IGNORECASE,
)
_PERCENTAGE = re.compile(r"\b\d+(?:\.\d+)?%")
_LARGE_NUMBER = re.compile(
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"
)

# Simple named-entity heuristic: capitalised multi-word sequences
_PROPER_NOUN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class DocumentIntelligence:
    """Pre-computed intelligence for a document, stored in MongoDB."""

    document_id: str
    profile_id: str
    subscription_id: str

    # Entities
    entities: List[Dict[str, Any]] = field(default_factory=list)

    # Relationships
    relationships: List[Dict[str, Any]] = field(default_factory=list)

    # Temporal analysis
    temporal: Dict[str, Any] = field(default_factory=dict)

    # Numerical analysis
    numerical: Dict[str, Any] = field(default_factory=dict)

    # Cross-document links
    cross_doc: Dict[str, Any] = field(default_factory=dict)

    # Pre-computed content
    summary: str = ""
    key_facts: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    processed_at: str = ""
    processing_duration_seconds: float = 0.0
    model_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB-storable dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

def _get_intel_collection():
    """Return the ``document_intelligence`` MongoDB collection."""
    from src.api.dataHandler import db
    return db["document_intelligence"]


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _normalise_date(raw: str) -> Optional[str]:
    """Best-effort normalise a date string to YYYY-MM-DD."""
    raw = raw.strip()

    # Already ISO
    if _DATE_ISO.fullmatch(raw):
        return raw

    # MM/DD/YYYY or DD/MM/YYYY — assume US format
    m = _DATE_SLASH.fullmatch(raw)
    if m:
        parts = raw.split("/")
        try:
            dt = datetime(int(parts[2]), int(parts[0]), int(parts[1]))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            return None

    # Month Year  ->  YYYY-MM-01
    m = _DATE_MONTH_YEAR.fullmatch(raw)
    if m:
        try:
            dt = datetime.strptime(raw, "%B %Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # Quarter
    m = _DATE_QUARTER.fullmatch(raw)
    if m:
        q_str = raw.upper().replace(" ", "")
        quarter = int(q_str[1])
        year = int(q_str[2:])
        month = (quarter - 1) * 3 + 1
        return f"{year}-{month:02d}-01"

    return None


def _extract_entities_heuristic(text: str) -> List[Dict[str, Any]]:
    """Quick heuristic entity extraction from capitalised sequences."""
    counts: Dict[str, int] = {}
    for match in _PROPER_NOUN.finditer(text):
        name = match.group(1)
        counts[name] = counts.get(name, 0) + 1

    entities = []
    for name, mentions in counts.items():
        entities.append({
            "name": name,
            "type": "ENTITY",
            "mentions": mentions,
            "confidence": min(0.6 + mentions * 0.05, 0.95),
        })
    return sorted(entities, key=lambda e: e["mentions"], reverse=True)


def _extract_temporal_info(
    text: str,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Extract dates, periods, timelines, and gaps from *text*."""
    raw_dates: List[str] = []

    for pattern in (_DATE_ISO, _DATE_SLASH, _DATE_MONTH_YEAR, _DATE_QUARTER):
        raw_dates.extend(pattern.findall(text))

    normalised = [d for raw in raw_dates if (d := _normalise_date(raw)) is not None]
    normalised = sorted(set(normalised))

    result: Dict[str, Any] = {
        "date_range": {},
        "events_timeline": [],
        "gaps": [],
    }

    if normalised:
        result["date_range"] = {
            "earliest": normalised[0],
            "latest": normalised[-1],
        }
        result["events_timeline"] = [{"date": d} for d in normalised]

    # LLM enrichment for gap detection / timeline ordering
    if llm_fn and normalised:
        prompt = (
            "You are a document analyst. Given these dates extracted from a document, "
            "identify any significant temporal gaps (periods > 6 months with no data) "
            "and return strict JSON: {\"gaps\": [{\"from\": \"YYYY-MM-DD\", \"to\": \"YYYY-MM-DD\", "
            "\"description\": \"...\"}], \"events_timeline\": [{\"date\": \"YYYY-MM-DD\", "
            "\"description\": \"...\"}]}\n\n"
            f"Dates: {json.dumps(normalised)}\n\n"
            f"Document excerpt (first 2000 chars):\n{text[:2000]}"
        )
        try:
            raw_resp = llm_fn(prompt)
            parsed = _safe_parse_json(raw_resp)
            if isinstance(parsed, dict):
                if "gaps" in parsed:
                    result["gaps"] = parsed["gaps"]
                if "events_timeline" in parsed:
                    result["events_timeline"] = parsed["events_timeline"]
        except Exception as exc:
            logger.warning("LLM temporal enrichment failed: %s", exc)

    return result


def _extract_numerical_info(
    text: str,
    tables: Optional[List[Dict]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Extract key figures, trends, anomalies from *text* and *tables*."""
    key_figures: List[Dict[str, Any]] = []

    # Currency
    for m in _CURRENCY.finditer(text):
        key_figures.append({
            "value": m.group(0).strip(),
            "type": "currency",
            "context": text[max(0, m.start() - 40): m.end() + 40].strip(),
        })

    # Percentages
    for m in _PERCENTAGE.finditer(text):
        key_figures.append({
            "value": m.group(0),
            "type": "percentage",
            "context": text[max(0, m.start() - 40): m.end() + 40].strip(),
        })

    # Large numbers
    for m in _LARGE_NUMBER.finditer(text):
        key_figures.append({
            "value": m.group(0),
            "type": "number",
            "context": text[max(0, m.start() - 40): m.end() + 40].strip(),
        })

    # Deduplicate by value
    seen = set()
    deduped = []
    for fig in key_figures:
        if fig["value"] not in seen:
            seen.add(fig["value"])
            deduped.append(fig)
    key_figures = deduped

    result: Dict[str, Any] = {
        "key_figures": key_figures,
        "trends": [],
        "anomalies": [],
    }

    # LLM enrichment for trend / anomaly detection
    if llm_fn and key_figures:
        figures_summary = json.dumps(key_figures[:20], default=str)
        table_text = ""
        if tables:
            table_text = f"\n\nTables:\n{json.dumps(tables[:5], default=str)}"

        prompt = (
            "You are a financial/numerical analyst. Given the key figures extracted "
            "from a document, identify trends and anomalies. Return strict JSON:\n"
            "{\"trends\": [{\"description\": \"...\", \"direction\": \"up|down|stable\", "
            "\"confidence\": 0.0-1.0}], \"anomalies\": [{\"description\": \"...\", "
            "\"severity\": \"low|medium|high\"}]}\n\n"
            f"Key figures: {figures_summary}{table_text}\n\n"
            f"Document excerpt (first 2000 chars):\n{text[:2000]}"
        )
        try:
            raw_resp = llm_fn(prompt)
            parsed = _safe_parse_json(raw_resp)
            if isinstance(parsed, dict):
                if "trends" in parsed:
                    result["trends"] = parsed["trends"]
                if "anomalies" in parsed:
                    result["anomalies"] = parsed["anomalies"]
        except Exception as exc:
            logger.warning("LLM numerical enrichment failed: %s", exc)

    return result


def _extract_key_facts(
    text: str,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """Extract key facts with source locations and confidence."""
    if not llm_fn:
        return []

    prompt = (
        "You are a document analyst. Extract the top 10 most important facts from "
        "the following document text. Return strict JSON — an array of objects:\n"
        '[{"fact": "...", "source": "page/section reference", "confidence": 0.0-1.0}]\n\n'
        f"Document text:\n{text[:4000]}"
    )
    try:
        raw_resp = llm_fn(prompt)
        parsed = _safe_parse_json(raw_resp)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "key_facts" in parsed:
            return parsed["key_facts"]
    except Exception as exc:
        logger.warning("Key facts extraction failed: %s", exc)

    return []


def _generate_summary(
    text: str,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Generate a concise document summary (max 200 words)."""
    if not llm_fn:
        # Fallback: first 200 words
        words = text.split()
        return " ".join(words[:200]) + ("..." if len(words) > 200 else "")

    prompt = (
        "Summarise the following document in no more than 200 words. "
        "Focus on the key points, findings, and conclusions.\n\n"
        f"Document text:\n{text[:6000]}"
    )
    try:
        return llm_fn(prompt)
    except Exception as exc:
        logger.warning("Summary generation failed: %s", exc)
        words = text.split()
        return " ".join(words[:200]) + ("..." if len(words) > 200 else "")


def _extract_relationships_llm(
    text: str,
    entities: List[Dict[str, Any]],
    llm_fn: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """Extract relationships between entities via LLM."""
    if not llm_fn or not entities:
        return []

    entity_names = [e["name"] for e in entities[:20]]
    prompt = (
        "You are a document analyst. Given the following entities found in a document, "
        "identify relationships between them. Return strict JSON — an array:\n"
        '[{"subject": "...", "predicate": "...", "object": "...", "evidence": "brief quote"}]\n\n'
        f"Entities: {json.dumps(entity_names)}\n\n"
        f"Document text:\n{text[:4000]}"
    )
    try:
        raw_resp = llm_fn(prompt)
        parsed = _safe_parse_json(raw_resp)
        if isinstance(parsed, list):
            return parsed
    except Exception as exc:
        logger.warning("Relationship extraction failed: %s", exc)

    return []


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _safe_parse_json(raw: str) -> Any:
    """Attempt to parse JSON from LLM output, handling markdown fences."""
    raw = raw.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Main computation entry point
# ---------------------------------------------------------------------------

def compute_document_intelligence(
    document_id: str,
    profile_id: str,
    subscription_id: str,
    extracted_text: str,
    extracted_tables: Optional[List[Dict]] = None,
    existing_entities: Optional[List[Dict]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> DocumentIntelligence:
    """Compute full document intelligence from extracted content.

    This is the heavy computation that runs during the PROCESSING phase.

    Args:
        document_id: Unique document identifier.
        profile_id: Owner profile.
        subscription_id: Tenant scope.
        extracted_text: Full extracted text of the document.
        extracted_tables: Optional list of table dicts from extraction.
        existing_entities: Pre-extracted entities (e.g. from screening).
        llm_fn: ``callable(prompt: str) -> str`` for LLM-based analysis.

    Returns:
        Fully populated :class:`DocumentIntelligence`.
    """
    start = time.time()

    # 1. Entities — merge heuristic + existing
    entities = existing_entities[:] if existing_entities else []
    heuristic_entities = _extract_entities_heuristic(extracted_text)
    existing_names = {e.get("name", "").lower() for e in entities}
    for ent in heuristic_entities:
        if ent["name"].lower() not in existing_names:
            entities.append(ent)

    # 2. Relationships
    relationships = _extract_relationships_llm(extracted_text, entities, llm_fn)

    # 3. Temporal
    temporal = _extract_temporal_info(extracted_text, llm_fn)

    # 4. Numerical
    numerical = _extract_numerical_info(extracted_text, extracted_tables, llm_fn)

    # 5. Key facts
    key_facts = _extract_key_facts(extracted_text, llm_fn)

    # 6. Summary
    summary = _generate_summary(extracted_text, llm_fn)

    elapsed = time.time() - start

    intel = DocumentIntelligence(
        document_id=document_id,
        profile_id=profile_id,
        subscription_id=subscription_id,
        entities=entities,
        relationships=relationships,
        temporal=temporal,
        numerical=numerical,
        cross_doc={
            "continues": [],
            "contradicts": [],
            "supersedes": [],
            "related_by_entity": [],
        },
        summary=summary,
        key_facts=key_facts,
        processed_at=datetime.now(timezone.utc).isoformat(),
        processing_duration_seconds=round(elapsed, 3),
        model_version="v2",
    )

    logger.info(
        "Document intelligence computed for doc=%s: entities=%d, facts=%d, duration=%.1fs",
        document_id,
        len(entities),
        len(key_facts),
        elapsed,
    )
    return intel


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_document_intelligence(intelligence: DocumentIntelligence) -> bool:
    """Store pre-computed intelligence in MongoDB.

    Uses upsert so re-processing overwrites previous results.
    """
    try:
        coll = _get_intel_collection()
        doc = intelligence.to_dict()
        coll.update_one(
            {"document_id": intelligence.document_id},
            {"$set": doc},
            upsert=True,
        )
        logger.info("Stored document intelligence for doc=%s", intelligence.document_id)
        return True
    except Exception as exc:
        logger.error("Failed to store document intelligence for doc=%s: %s", intelligence.document_id, exc)
        return False


def load_document_intelligence(document_id: str) -> Optional[DocumentIntelligence]:
    """Load pre-computed intelligence from MongoDB for query-time enrichment.

    Returns ``None`` if no intelligence record exists for the given document.
    """
    try:
        coll = _get_intel_collection()
        doc = coll.find_one({"document_id": document_id})
        if doc is None:
            return None

        # Remove MongoDB _id field
        doc.pop("_id", None)

        return DocumentIntelligence(**{
            k: v for k, v in doc.items()
            if k in DocumentIntelligence.__dataclass_fields__
        })
    except Exception as exc:
        logger.error("Failed to load document intelligence for doc=%s: %s", document_id, exc)
        return None
