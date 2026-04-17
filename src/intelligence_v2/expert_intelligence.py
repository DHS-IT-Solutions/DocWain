"""Adaptive Expert Intelligence — background analysis pipeline.

Runs after document embedding completes. Produces a profile_expertise
document that makes DocWain embody the expert its documents demand.

Phase 1: Profile Understanding — single LLM call to identify expertise identity,
         knowledge map, proactive insights, advisory capabilities, knowledge gaps.
Phase 2: Deep Expert Analysis — per-document-cluster LLM calls for connections,
         implications, recommendations.

All calls use the smart path (27B) and run at background priority.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 1 prompt
# ---------------------------------------------------------------------------

_PHASE1_SYSTEM = (
    "You are analyzing a collection of documents to build an expert profile. "
    "Your output must be valid JSON — no markdown fences, no commentary."
)

_PHASE1_PROMPT_TEMPLATE = """\
Below are summaries, entities, and key facts from a document collection.

{doc_summaries}

Based on this collection, produce JSON with these fields:

{{
  "expertise_identity": {{
    "role": "<What kind of expert would deeply understand these documents? Be specific, e.g. 'Senior Contract Negotiation Specialist for IT outsourcing agreements'>",
    "mindset": "<How does this expert approach problems? What do they prioritize?>",
    "tone": "<Communication style: e.g. 'Practical, direct, solution-oriented'>"
  }},
  "knowledge_map": [
    {{
      "area": "<knowledge area covered>",
      "depth": "comprehensive|detailed|partial|minimal",
      "document_ids": ["<doc_ids covering this area>"]
    }}
  ],
  "proactive_insights": [
    {{
      "category": "critical|important|informational",
      "insight": "<What would an expert immediately notice or flag?>",
      "recommendation": "<What action should be taken?>",
      "evidence_refs": ["<document_ids>"]
    }}
  ],
  "advisory_capabilities": ["<What kinds of questions can this expert answer authoritatively?>"],
  "knowledge_gaps": ["<What is NOT covered that a user might expect?>"]
}}
"""

# ---------------------------------------------------------------------------
# Phase 2 prompt
# ---------------------------------------------------------------------------

_PHASE2_SYSTEM = (
    "You are a {role}. Analyze the following document cluster deeply. "
    "Your output must be valid JSON — no markdown fences, no commentary."
)

_PHASE2_PROMPT_TEMPLATE = """\
As a {role} with the mindset "{mindset}", analyze this document cluster:

Topic: {cluster_topic}

Documents:
{cluster_docs}

Produce JSON:
{{
  "cluster_topic": "{cluster_topic}",
  "connections": ["<Connections between documents that aren't explicitly stated>"],
  "implications": ["<Implications a user might miss>"],
  "recommendations": ["<Actionable recommendations based on expert analysis>"]
}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_profile_expertise(
    profile_id: str,
    subscription_id: str,
    mongo_client: Any,
    vllm_manager: Any,
) -> Optional[Dict[str, Any]]:
    """Run Phase 1 + Phase 2 expert analysis for a profile.

    Parameters
    ----------
    profile_id : str
        The profile to analyze.
    subscription_id : str
        Tenant scope.
    mongo_client : Any
        MongoDB client or database handle.
    vllm_manager : Any
        VLLMManager instance for smart path queries.

    Returns
    -------
    dict or None
        The profile_expertise document, or None if analysis failed.
    """
    t0 = time.monotonic()
    db = _get_db(mongo_client)

    # Gather existing intelligence
    doc_intel = list(db["documents"].find(
        {"profile_id": profile_id, "subscription_id": subscription_id, "intelligence_ready": True},
        {"document_id": 1, "filename": 1, "intelligence": 1, "_id": 0},
    ))

    if not doc_intel:
        logger.info("[EXPERT] No documents with intelligence for profile=%s", profile_id)
        return None

    doc_ids = [d["document_id"] for d in doc_intel]
    logger.info("[EXPERT] Phase 1: analyzing %d documents for profile=%s", len(doc_intel), profile_id)

    # Build document summaries block for Phase 1
    doc_summaries = _format_doc_summaries(doc_intel)

    # Phase 1: Profile Understanding
    phase1_prompt = _PHASE1_PROMPT_TEMPLATE.format(doc_summaries=doc_summaries)
    try:
        phase1_raw = vllm_manager.query(
            prompt=phase1_prompt,
            system_prompt=_PHASE1_SYSTEM,
            max_tokens=4096,
            temperature=0.3,
        )
        phase1 = _parse_json(phase1_raw)
        if not phase1:
            logger.error("[EXPERT] Phase 1 returned invalid JSON for profile=%s", profile_id)
            return None
    except Exception:
        logger.error("[EXPERT] Phase 1 LLM call failed for profile=%s", profile_id, exc_info=True)
        return None

    logger.info("[EXPERT] Phase 1 complete: role=%s", phase1.get("expertise_identity", {}).get("role", "unknown"))

    # Phase 2: Deep Expert Analysis (per-document cluster)
    clusters = _build_clusters(doc_intel)
    deep_analysis = []
    identity = phase1.get("expertise_identity", {})

    for cluster in clusters:
        try:
            phase2_prompt = _PHASE2_PROMPT_TEMPLATE.format(
                role=identity.get("role", "document analyst"),
                mindset=identity.get("mindset", "thorough and precise"),
                cluster_topic=cluster["topic"],
                cluster_docs=cluster["docs_text"],
            )
            phase2_system = _PHASE2_SYSTEM.format(role=identity.get("role", "document analyst"))
            phase2_raw = vllm_manager.query(
                prompt=phase2_prompt,
                system_prompt=phase2_system,
                max_tokens=2048,
                temperature=0.3,
            )
            phase2 = _parse_json(phase2_raw)
            if phase2:
                deep_analysis.append(phase2)
        except Exception:
            logger.warning("[EXPERT] Phase 2 failed for cluster=%s", cluster["topic"], exc_info=True)
            continue

        # Backpressure: 2s delay between cluster analysis calls
        time.sleep(2)

    # Build final expertise document
    expertise = {
        "profile_id": profile_id,
        "subscription_id": subscription_id,
        **phase1,
        "deep_analysis": deep_analysis,
        "document_ids_analyzed": doc_ids,
        "version": 1,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    # Upsert to MongoDB
    db["profile_expertise"].replace_one(
        {"profile_id": profile_id},
        expertise,
        upsert=True,
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "[EXPERT] Complete: profile=%s docs=%d clusters=%d elapsed=%.1fs",
        profile_id, len(doc_ids), len(deep_analysis), elapsed,
    )

    return expertise


def get_cached_expertise(
    profile_id: str,
    mongo_client: Any,
) -> Optional[Dict[str, Any]]:
    """Load cached profile expertise from MongoDB."""
    db = _get_db(mongo_client)
    return db["profile_expertise"].find_one(
        {"profile_id": profile_id},
        {"_id": 0},
    )


def is_stale(
    profile_id: str,
    subscription_id: str,
    mongo_client: Any,
) -> bool:
    """Check if expertise needs rebuilding (new docs added/removed)."""
    db = _get_db(mongo_client)
    expertise = db["profile_expertise"].find_one(
        {"profile_id": profile_id},
        {"document_ids_analyzed": 1, "_id": 0},
    )
    if not expertise:
        return True

    current_doc_ids = set(
        d["document_id"] for d in db["documents"].find(
            {"profile_id": profile_id, "subscription_id": subscription_id, "intelligence_ready": True},
            {"document_id": 1, "_id": 0},
        )
    )
    analyzed_ids = set(expertise.get("document_ids_analyzed", []))
    return current_doc_ids != analyzed_ids


def filter_insights_for_query(
    expertise: Dict[str, Any],
    query: str,
) -> List[Dict[str, Any]]:
    """Return expert insights relevant to the query (simple keyword matching)."""
    if not expertise:
        return []

    query_lower = query.lower()
    query_words = set(query_lower.split())
    scored = []

    for insight in expertise.get("proactive_insights", []):
        insight_text = (insight.get("insight", "") + " " + insight.get("recommendation", "")).lower()
        insight_words = set(insight_text.split())
        overlap = len(query_words & insight_words)
        if overlap > 0:
            scored.append((overlap, insight))

    # Also include deep analysis recommendations
    for analysis in expertise.get("deep_analysis", []):
        topic = analysis.get("cluster_topic", "").lower()
        if any(w in topic for w in query_words):
            for rec in analysis.get("recommendations", []):
                scored.append((2, {"category": "important", "insight": rec, "recommendation": ""}))
            for imp in analysis.get("implications", []):
                scored.append((1, {"category": "informational", "insight": imp, "recommendation": ""}))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:5]]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _get_db(mongo_client: Any) -> Any:
    """Return the database handle from a client or pass through."""
    # If it has a 'list_collection_names' method, it's already a Database
    if hasattr(mongo_client, "list_collection_names"):
        return mongo_client
    # If it's a MongoClient, get the default database
    if hasattr(mongo_client, "get_default_database"):
        try:
            return mongo_client.get_default_database()
        except Exception:
            pass
    return mongo_client


def _format_doc_summaries(doc_intel: List[Dict]) -> str:
    """Format document intelligence into a text block for the LLM prompt."""
    parts = []
    for doc in doc_intel:
        doc_id = doc.get("document_id", "unknown")
        filename = doc.get("filename", "unknown")
        intel = doc.get("intelligence", {})
        summary = intel.get("summary", intel.get("document_summary", "No summary"))
        entities = intel.get("entities", intel.get("key_entities", []))
        facts = intel.get("facts", intel.get("key_facts", []))

        parts.append(f"Document: {filename} (ID: {doc_id})")
        parts.append(f"  Summary: {summary}")

        if entities:
            entity_strs = []
            for e in entities[:8]:
                if isinstance(e, dict):
                    entity_strs.append(e.get("value", e.get("name", str(e))))
                else:
                    entity_strs.append(str(e))
            parts.append(f"  Entities: {', '.join(entity_strs)}")

        if facts:
            fact_strs = []
            for f in facts[:5]:
                if isinstance(f, dict):
                    fact_strs.append(f.get("claim", str(f)))
                else:
                    fact_strs.append(str(f))
            parts.append(f"  Key facts: {'; '.join(fact_strs)}")

        parts.append("")

    return "\n".join(parts)


def _build_clusters(doc_intel: List[Dict]) -> List[Dict]:
    """Group documents into clusters by document type for Phase 2 analysis."""
    by_type: Dict[str, List[Dict]] = {}
    for doc in doc_intel:
        intel = doc.get("intelligence", {})
        doc_type = intel.get("document_type", "general")
        by_type.setdefault(doc_type, []).append(doc)

    clusters = []
    for doc_type, docs in by_type.items():
        docs_text = _format_doc_summaries(docs)
        clusters.append({
            "topic": doc_type,
            "docs_text": docs_text,
            "doc_ids": [d["document_id"] for d in docs],
        })

    return clusters


def _parse_json(text: str) -> Optional[Dict]:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    import json
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None
