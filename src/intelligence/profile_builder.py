"""Profile Intelligence Builder — pre-computes structured intelligence for a
document profile after embedding.

Orchestrates domain detection, entity aggregation, profile computation, and
collection insight generation.  Results are cached in MongoDB and formatted
for LLM context injection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger
from src.intelligence.collection_insights import generate_insights

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_TTL = timedelta(hours=1)

_DOC_TYPE_TO_DOMAIN = {
    # HR / Recruitment
    "resume": "hr_recruitment",
    "cv": "hr_recruitment",
    "cover_letter": "hr_recruitment",
    "job_application": "hr_recruitment",
    # Finance
    "invoice": "finance",
    "po": "finance",
    "purchase_order": "finance",
    "receipt": "finance",
    "bank_statement": "finance",
    "tax": "finance",
    "expense_report": "finance",
    # Legal
    "contract": "legal",
    "nda": "legal",
    "agreement": "legal",
    "terms_of_service": "legal",
    "legal": "legal",
    "policy": "legal",
    # Logistics
    "inventory": "logistics",
    "shipping": "logistics",
    "warehouse": "logistics",
    "bill_of_lading": "logistics",
    "packing_list": "logistics",
    # Medical
    "medical_record": "medical",
    "prescription": "medical",
    "lab_report": "medical",
    "discharge_summary": "medical",
    "medical": "medical",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProfileIntelligence:
    """Pre-computed intelligence snapshot for a document profile."""

    profile_id: str
    profile_type: str  # hr_recruitment | finance | legal | logistics | medical | generic
    document_count: int
    last_updated: str  # ISO-8601 timestamp
    entities_summary: Dict[str, Any] = field(default_factory=dict)
    # {total: N, by_type: {Person: 10, Skill: 50, ...}}
    computed_profiles: List[Dict[str, Any]] = field(default_factory=list)
    # list of domain-specific entity profile dicts
    collection_insights: Dict[str, Any] = field(default_factory=dict)
    # {distributions: {}, patterns: [], gaps: [], anomalies: []}
    domain_metadata: Dict[str, Any] = field(default_factory=dict)
    # {detected_domain, document_types, analysis_templates}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfileIntelligence":
        return cls(
            profile_id=data.get("profile_id", ""),
            profile_type=data.get("profile_type", "generic"),
            document_count=data.get("document_count", 0),
            last_updated=data.get("last_updated", ""),
            entities_summary=data.get("entities_summary", {}),
            computed_profiles=data.get("computed_profiles", []),
            collection_insights=data.get("collection_insights", {}),
            domain_metadata=data.get("domain_metadata", {}),
        )


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------


def _detect_domain(doc_metadata: List[Dict]) -> str:
    """Detect the dominant domain from document type labels."""
    domain_votes: Dict[str, int] = {}
    for meta in doc_metadata:
        doc_type = str(meta.get("doc_type", meta.get("document_type", ""))).lower().strip()
        domain = _DOC_TYPE_TO_DOMAIN.get(doc_type)
        if domain:
            domain_votes[domain] = domain_votes.get(domain, 0) + 1
    if not domain_votes:
        return "generic"
    return max(domain_votes, key=domain_votes.get)  # type: ignore[arg-type]


def _unique_doc_types(doc_metadata: List[Dict]) -> List[str]:
    types: set = set()
    for meta in doc_metadata:
        dt = meta.get("doc_type", meta.get("document_type"))
        if dt:
            types.add(str(dt).lower().strip())
    return sorted(types)


# ---------------------------------------------------------------------------
# Domain-specific profile builders
# ---------------------------------------------------------------------------


def _build_hr_profiles(entities: List[Dict]) -> List[Dict]:
    """Build candidate profiles from extracted entities."""
    candidates: Dict[str, Dict] = {}
    for ent in entities:
        etype = str(ent.get("type", "")).lower()
        if etype not in ("person", "candidate", "applicant"):
            continue
        name = ent.get("name", ent.get("label", "unknown"))
        if name not in candidates:
            candidates[name] = {
                "name": name,
                "skills": [],
                "certifications": [],
                "experience_years": ent.get("experience_years"),
                "role_fit_score": ent.get("role_fit_score"),
                "primary_skill": None,
            }
        c = candidates[name]
        for k in ("skills", "certifications"):
            val = ent.get(k)
            if isinstance(val, list):
                c[k] = list(set(c[k] + val))
            elif isinstance(val, str) and val:
                c[k] = list(set(c[k] + [val]))
        for k in ("experience_years", "role_fit_score", "email"):
            if ent.get(k) is not None:
                c[k] = ent[k]
    # Set primary skill
    for c in candidates.values():
        if c["skills"]:
            c["primary_skill"] = c["skills"][0]
    return list(candidates.values())


def _build_finance_profiles(entities: List[Dict]) -> List[Dict]:
    """Build vendor profiles from extracted entities."""
    vendors: Dict[str, Dict] = {}
    for ent in entities:
        etype = str(ent.get("type", "")).lower()
        if etype not in ("vendor", "supplier", "company", "organization"):
            continue
        name = ent.get("name", ent.get("label", "unknown"))
        if name not in vendors:
            vendors[name] = {
                "name": name,
                "total_spend": 0.0,
                "invoice_count": 0,
                "avg_payment_terms": None,
            }
        v = vendors[name]
        spend = ent.get("total_spend", ent.get("amount"))
        if spend is not None:
            try:
                v["total_spend"] += float(spend)
            except (TypeError, ValueError):
                pass
        inv = ent.get("invoice_count")
        if inv is not None:
            try:
                v["invoice_count"] += int(inv)
            except (TypeError, ValueError):
                pass
        terms = ent.get("avg_payment_terms", ent.get("payment_terms"))
        if terms is not None:
            v["avg_payment_terms"] = terms
    return list(vendors.values())


def _build_legal_profiles(entities: List[Dict]) -> List[Dict]:
    """Build contract profiles from extracted entities."""
    contracts: List[Dict] = []
    for ent in entities:
        etype = str(ent.get("type", "")).lower()
        if etype not in ("contract", "agreement", "nda", "document"):
            continue
        contracts.append({
            "name": ent.get("name", ent.get("label", "unknown")),
            "parties": ent.get("parties", []),
            "key_dates": ent.get("key_dates", {}),
            "obligations": ent.get("obligations", []),
            "risk_level": ent.get("risk_level", "unknown"),
        })
    return contracts


def _build_logistics_profiles(entities: List[Dict]) -> List[Dict]:
    """Build product / inventory profiles from extracted entities."""
    products: Dict[str, Dict] = {}
    for ent in entities:
        etype = str(ent.get("type", "")).lower()
        if etype not in ("product", "item", "sku", "inventory"):
            continue
        name = ent.get("name", ent.get("label", "unknown"))
        if name not in products:
            products[name] = {
                "name": name,
                "stock_level": None,
                "reorder_point": None,
                "supplier": None,
                "lead_time_days": None,
            }
        p = products[name]
        for k in ("stock_level", "reorder_point", "supplier", "lead_time_days"):
            if ent.get(k) is not None:
                p[k] = ent[k]
    return list(products.values())


def _build_medical_profiles(entities: List[Dict]) -> List[Dict]:
    """Build patient profiles from extracted entities."""
    patients: Dict[str, Dict] = {}
    for ent in entities:
        etype = str(ent.get("type", "")).lower()
        if etype not in ("patient", "person", "individual"):
            continue
        name = ent.get("name", ent.get("label", "unknown"))
        if name not in patients:
            patients[name] = {
                "name": name,
                "diagnoses": [],
                "medications": [],
                "providers": [],
                "primary_diagnosis": None,
                "primary_provider": None,
            }
        pt = patients[name]
        for k in ("diagnoses", "medications", "providers"):
            val = ent.get(k)
            if isinstance(val, list):
                pt[k] = list(set(pt[k] + val))
            elif isinstance(val, str) and val:
                pt[k] = list(set(pt[k] + [val]))
        if pt["diagnoses"]:
            pt["primary_diagnosis"] = pt["diagnoses"][0]
        if pt["providers"]:
            pt["primary_provider"] = pt["providers"][0]
    return list(patients.values())


def _build_generic_profiles(entities: List[Dict]) -> List[Dict]:
    """Pass-through: just tag each entity."""
    return [
        {
            "name": ent.get("name", ent.get("label", "unknown")),
            "type": ent.get("type", "unknown"),
            **{k: v for k, v in ent.items() if k not in ("name", "label", "type")},
        }
        for ent in entities
    ]


_PROFILE_BUILDERS = {
    "hr_recruitment": _build_hr_profiles,
    "finance": _build_finance_profiles,
    "legal": _build_legal_profiles,
    "logistics": _build_logistics_profiles,
    "medical": _build_medical_profiles,
    "generic": _build_generic_profiles,
}

_ANALYSIS_TEMPLATES = {
    "hr_recruitment": ["candidate_comparison", "skills_gap_analysis", "role_fit_ranking"],
    "finance": ["spend_analysis", "vendor_comparison", "payment_trends"],
    "legal": ["risk_assessment", "obligation_tracking", "expiry_monitoring"],
    "logistics": ["inventory_status", "reorder_planning", "supplier_performance"],
    "medical": ["patient_summary", "medication_review", "care_coordination"],
    "generic": ["document_summary", "entity_overview"],
}


# ---------------------------------------------------------------------------
# Entities summary helper
# ---------------------------------------------------------------------------


def _summarise_entities(entities: List[Dict]) -> Dict[str, Any]:
    by_type: Dict[str, int] = {}
    for ent in entities:
        t = str(ent.get("type", "unknown"))
        by_type[t] = by_type.get(t, 0) + 1
    return {"total": len(entities), "by_type": by_type}


# ---------------------------------------------------------------------------
# ProfileBuilder
# ---------------------------------------------------------------------------


class ProfileBuilder:
    """Pre-computes and caches structured intelligence for a document profile."""

    # -- Public API ----------------------------------------------------------

    def build(
        self,
        profile_id: str,
        subscription_id: str,
        mongo_client: Any,
        kg_client: Any = None,
    ) -> ProfileIntelligence:
        """Build (or rebuild) profile intelligence.

        Steps:
            1. Query MongoDB for document metadata in the profile.
            2. Detect domain from document types.
            3. Query KG for entities/relationships if available.
            4. Compute domain-specific profiles.
            5. Generate collection insights.
            6. Store in MongoDB ``computed_profiles`` collection.
            7. Return :class:`ProfileIntelligence`.
        """
        logger.info("Building intelligence for profile=%s sub=%s", profile_id, subscription_id)

        # 1. Document metadata
        doc_metadata = self._fetch_doc_metadata(profile_id, subscription_id, mongo_client)
        logger.info("Found %d documents in profile %s", len(doc_metadata), profile_id)

        # 2. Domain detection
        domain = _detect_domain(doc_metadata)
        doc_types = _unique_doc_types(doc_metadata)
        logger.info("Detected domain=%s for profile %s", domain, profile_id)

        # 3. Entities / relationships
        entities, relationships = self._fetch_entities(
            profile_id, subscription_id, mongo_client, kg_client
        )
        logger.info("Fetched %d entities, %d relationships", len(entities), len(relationships))

        # 4. Domain-specific profiles
        builder_fn = _PROFILE_BUILDERS.get(domain, _build_generic_profiles)
        computed = builder_fn(entities)

        # 5. Collection insights
        insights = generate_insights(domain, computed, relationships, doc_metadata)

        # 6. Assemble
        now_iso = datetime.now(timezone.utc).isoformat()
        intelligence = ProfileIntelligence(
            profile_id=profile_id,
            profile_type=domain,
            document_count=len(doc_metadata),
            last_updated=now_iso,
            entities_summary=_summarise_entities(entities),
            computed_profiles=computed,
            collection_insights=insights,
            domain_metadata={
                "detected_domain": domain,
                "document_types": doc_types,
                "analysis_templates": _ANALYSIS_TEMPLATES.get(domain, []),
            },
        )

        # 7. Persist
        self._store(intelligence, mongo_client)
        return intelligence

    def get_cached(
        self, profile_id: str, mongo_client: Any
    ) -> Optional[ProfileIntelligence]:
        """Return cached :class:`ProfileIntelligence` if it exists and is
        less than 1 hour old.  Returns ``None`` otherwise.
        """
        try:
            db = mongo_client.get_default_database() if hasattr(mongo_client, "get_default_database") else mongo_client.docwain
            doc = db.computed_profiles.find_one({"profile_id": profile_id})
            if doc is None:
                return None
            last = doc.get("last_updated", "")
            if last:
                updated_dt = datetime.fromisoformat(last)
                if datetime.now(timezone.utc) - updated_dt > _CACHE_TTL:
                    logger.debug("Cache stale for profile %s", profile_id)
                    return None
            return ProfileIntelligence.from_dict(doc)
        except Exception:
            logger.exception("Error reading cached intelligence for %s", profile_id)
            return None

    @staticmethod
    def to_context_block(intelligence: ProfileIntelligence) -> str:
        """Format *intelligence* as a ``<profile_context>`` string suitable
        for injection into an LLM system prompt.
        """
        lines = [
            "<profile_context>",
            f"Profile: {intelligence.profile_id}",
            f"Domain: {intelligence.profile_type}",
            f"Documents: {intelligence.document_count}",
            f"Last updated: {intelligence.last_updated}",
        ]

        # Entities summary
        es = intelligence.entities_summary
        if es:
            lines.append(f"Total entities: {es.get('total', 0)}")
            by_type = es.get("by_type", {})
            if by_type:
                parts = [f"{k}: {v}" for k, v in sorted(by_type.items(), key=lambda x: -x[1])]
                lines.append(f"Entity types: {', '.join(parts)}")

        # Key profiles (limit to 10 for prompt size)
        profiles = intelligence.computed_profiles[:10]
        if profiles:
            lines.append(f"Key profiles ({len(profiles)}):")
            for p in profiles:
                name = p.get("name", "?")
                summary_parts = [f"{k}={v}" for k, v in p.items() if k != "name" and v]
                lines.append(f"  - {name}: {', '.join(summary_parts[:6])}")

        # Patterns
        patterns = intelligence.collection_insights.get("patterns", [])
        if patterns:
            lines.append("Patterns:")
            for pat in patterns[:5]:
                lines.append(f"  - {pat}")

        # Anomalies
        anomalies = intelligence.collection_insights.get("anomalies", [])
        if anomalies:
            lines.append("Anomalies:")
            for a in anomalies[:5]:
                lines.append(f"  - {a}")

        lines.append("</profile_context>")
        return "\n".join(lines)

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _fetch_doc_metadata(
        profile_id: str, subscription_id: str, mongo_client: Any
    ) -> List[Dict]:
        """Query MongoDB for document metadata belonging to the profile."""
        try:
            db = mongo_client.get_default_database() if hasattr(mongo_client, "get_default_database") else mongo_client.docwain
            query = {"profile_id": profile_id}
            if subscription_id:
                query["subscription_id"] = subscription_id
            cursor = db.documents.find(query, {
                "doc_type": 1, "document_type": 1, "filename": 1,
                "uploaded_at": 1, "metadata": 1, "_id": 0,
            })
            return list(cursor)
        except Exception:
            logger.exception("Error fetching doc metadata for profile %s", profile_id)
            return []

    @staticmethod
    def _fetch_entities(
        profile_id: str,
        subscription_id: str,
        mongo_client: Any,
        kg_client: Any,
    ) -> tuple:
        """Fetch entities and relationships from KG client or MongoDB fallback."""
        entities: List[Dict] = []
        relationships: List[Dict] = []

        # Try knowledge graph client first
        if kg_client is not None:
            try:
                kg_result = kg_client.get_profile_graph(profile_id)
                if isinstance(kg_result, dict):
                    entities = kg_result.get("entities", [])
                    relationships = kg_result.get("relationships", [])
                    return entities, relationships
            except Exception:
                logger.warning("KG client failed for profile %s, falling back to MongoDB", profile_id)

        # Fallback: MongoDB entities collection
        try:
            db = mongo_client.get_default_database() if hasattr(mongo_client, "get_default_database") else mongo_client.docwain
            query = {"profile_id": profile_id}
            if subscription_id:
                query["subscription_id"] = subscription_id
            entities = list(db.entities.find(query, {"_id": 0}))
            relationships = list(db.relationships.find(query, {"_id": 0}))
        except Exception:
            logger.exception("Error fetching entities from MongoDB for profile %s", profile_id)

        return entities, relationships

    @staticmethod
    def _store(intelligence: ProfileIntelligence, mongo_client: Any) -> None:
        """Upsert into MongoDB ``computed_profiles`` collection."""
        try:
            db = mongo_client.get_default_database() if hasattr(mongo_client, "get_default_database") else mongo_client.docwain
            db.computed_profiles.update_one(
                {"profile_id": intelligence.profile_id},
                {"$set": intelligence.to_dict()},
                upsert=True,
            )
            logger.info("Stored intelligence for profile %s", intelligence.profile_id)
        except Exception:
            logger.exception("Error storing intelligence for profile %s", intelligence.profile_id)
