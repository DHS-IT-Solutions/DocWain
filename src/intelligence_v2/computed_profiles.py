"""Domain-specific profile computation.

Each compute function takes entity/relationship dicts from KG and returns
structured profile dicts tailored to the domain.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def compute_profiles(
    profile_type: str,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Dispatch to domain-specific profile computation."""
    fn = _COMPUTE_DISPATCH.get(profile_type, compute_generic_profiles)
    return fn(entities, relationships)


# ---------------------------------------------------------------------------
# HR / Recruitment
# ---------------------------------------------------------------------------


def compute_hr_profiles(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build candidate profiles with skills, certifications, experience, role-fit."""
    persons = [e for e in entities if e.get("type", "").upper() in ("PERSON", "CANDIDATE")]
    skills = [e for e in entities if e.get("type", "").upper() in ("SKILL", "TECHNOLOGY")]
    certs = [e for e in entities if e.get("type", "").upper() in ("CERTIFICATION", "CERTIFICATE")]
    orgs = [e for e in entities if e.get("type", "").upper() in ("ORGANIZATION", "COMPANY")]

    # Map person -> related entities via relationships
    person_skills: Dict[str, List[str]] = defaultdict(list)
    person_certs: Dict[str, List[str]] = defaultdict(list)
    person_orgs: Dict[str, List[str]] = defaultdict(list)

    skill_names = {s.get("name", "").lower(): s.get("name", "") for s in skills}
    cert_names = {c.get("name", "").lower(): c.get("name", "") for c in certs}
    org_names = {o.get("name", "").lower(): o.get("name", "") for o in orgs}

    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rel_type = (rel.get("relation_type") or rel.get("relation", "")).upper()

        if tgt.lower() in skill_names:
            person_skills[src].append(skill_names[tgt.lower()])
        elif tgt.lower() in cert_names:
            person_certs[src].append(cert_names[tgt.lower()])
        elif tgt.lower() in org_names:
            person_orgs[src].append(org_names[tgt.lower()])

    profiles: List[Dict[str, Any]] = []
    for person in persons:
        name = person.get("name", "Unknown")
        profiles.append({
            "type": "candidate",
            "label": name,
            "name": name,
            "skills": sorted(set(person_skills.get(name, []))),
            "certifications": sorted(set(person_certs.get(name, []))),
            "experience_orgs": sorted(set(person_orgs.get(name, []))),
            "experience_count": len(person_orgs.get(name, [])),
            "role_fit_indicators": _compute_role_fit(
                person_skills.get(name, []),
                person_certs.get(name, []),
                person_orgs.get(name, []),
            ),
        })

    # If no persons found, summarise skills as a pool
    if not profiles and skills:
        profiles.append({
            "type": "skill_pool",
            "label": "Skill Pool",
            "skills": sorted(set(s.get("name", "") for s in skills))[:50],
            "total_skills": len(skills),
        })

    return profiles


def _compute_role_fit(
    skills: List[str],
    certs: List[str],
    orgs: List[str],
) -> Dict[str, Any]:
    """Heuristic role-fit indicators."""
    return {
        "skill_breadth": len(set(skills)),
        "has_certifications": len(certs) > 0,
        "experience_depth": len(set(orgs)),
    }


# ---------------------------------------------------------------------------
# Finance
# ---------------------------------------------------------------------------


def compute_finance_profiles(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build vendor/invoice profiles with amounts, terms, payment status."""
    vendors = [e for e in entities if e.get("type", "").upper() in ("VENDOR", "SUPPLIER", "ORGANIZATION", "COMPANY")]
    amounts = [e for e in entities if e.get("type", "").upper() in ("AMOUNT", "MONETARY_VALUE", "MONEY")]
    dates = [e for e in entities if e.get("type", "").upper() in ("DATE", "DUE_DATE", "PAYMENT_DATE")]

    vendor_amounts: Dict[str, List[str]] = defaultdict(list)
    vendor_terms: Dict[str, List[str]] = defaultdict(list)

    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rel_type = (rel.get("relation_type") or rel.get("relation", "")).upper()

        if rel_type in ("INVOICED", "BILLED", "CHARGED", "PAID"):
            vendor_amounts[src].append(tgt)
        elif rel_type in ("PAYMENT_TERMS", "DUE", "NET"):
            vendor_terms[src].append(tgt)

    profiles: List[Dict[str, Any]] = []
    for vendor in vendors:
        name = vendor.get("name", "Unknown")
        profiles.append({
            "type": "vendor",
            "label": name,
            "name": name,
            "invoice_amounts": vendor_amounts.get(name, []),
            "payment_terms": vendor_terms.get(name, []),
            "transaction_count": len(vendor_amounts.get(name, [])),
        })

    # Summary profile for monetary values
    if amounts:
        profiles.append({
            "type": "financial_summary",
            "label": "Financial Summary",
            "total_amounts_found": len(amounts),
            "amounts": [a.get("name", "") for a in amounts[:30]],
        })

    return profiles


# ---------------------------------------------------------------------------
# Legal
# ---------------------------------------------------------------------------


def compute_legal_profiles(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build contract profiles with parties, obligations, key dates."""
    parties = [e for e in entities if e.get("type", "").upper() in (
        "PARTY", "ORGANIZATION", "COMPANY", "PERSON", "CONTRACTING_PARTY"
    )]
    obligations = [e for e in entities if e.get("type", "").upper() in (
        "OBLIGATION", "CLAUSE", "TERM", "CONDITION"
    )]
    dates = [e for e in entities if e.get("type", "").upper() in (
        "DATE", "EFFECTIVE_DATE", "EXPIRY_DATE", "RENEWAL_DATE", "DEADLINE"
    )]

    party_obligations: Dict[str, List[str]] = defaultdict(list)
    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rel_type = (rel.get("relation_type") or rel.get("relation", "")).upper()

        if rel_type in ("OBLIGATED", "RESPONSIBLE", "SHALL", "MUST"):
            party_obligations[src].append(tgt)

    profiles: List[Dict[str, Any]] = []
    for party in parties:
        name = party.get("name", "Unknown")
        profiles.append({
            "type": "contract_party",
            "label": name,
            "name": name,
            "obligations": party_obligations.get(name, []),
            "obligation_count": len(party_obligations.get(name, [])),
        })

    if dates:
        profiles.append({
            "type": "contract_timeline",
            "label": "Key Dates",
            "dates": [
                {"type": d.get("type", "DATE"), "value": d.get("name", "")}
                for d in dates[:20]
            ],
        })

    if obligations:
        profiles.append({
            "type": "obligations_summary",
            "label": "Obligations",
            "total": len(obligations),
            "items": [o.get("name", "") for o in obligations[:30]],
        })

    return profiles


# ---------------------------------------------------------------------------
# Logistics
# ---------------------------------------------------------------------------


def compute_logistics_profiles(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build product profiles with stock levels, suppliers, lead times."""
    products = [e for e in entities if e.get("type", "").upper() in (
        "PRODUCT", "ITEM", "SKU", "GOODS", "MATERIAL"
    )]
    suppliers = [e for e in entities if e.get("type", "").upper() in (
        "SUPPLIER", "VENDOR", "MANUFACTURER"
    )]
    locations = [e for e in entities if e.get("type", "").upper() in (
        "LOCATION", "WAREHOUSE", "FACILITY", "PORT"
    )]
    quantities = [e for e in entities if e.get("type", "").upper() in (
        "QUANTITY", "STOCK_LEVEL", "COUNT"
    )]

    product_suppliers: Dict[str, List[str]] = defaultdict(list)
    product_quantities: Dict[str, List[str]] = defaultdict(list)

    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rel_type = (rel.get("relation_type") or rel.get("relation", "")).upper()

        if rel_type in ("SUPPLIED_BY", "SOURCED_FROM", "MANUFACTURED_BY"):
            product_suppliers[src].append(tgt)
        elif rel_type in ("HAS_STOCK", "QUANTITY", "INVENTORY"):
            product_quantities[src].append(tgt)

    profiles: List[Dict[str, Any]] = []
    for product in products:
        name = product.get("name", "Unknown")
        profiles.append({
            "type": "product",
            "label": name,
            "name": name,
            "suppliers": sorted(set(product_suppliers.get(name, []))),
            "stock_levels": product_quantities.get(name, []),
            "supplier_count": len(set(product_suppliers.get(name, []))),
        })

    if locations:
        profiles.append({
            "type": "logistics_locations",
            "label": "Locations",
            "locations": [loc.get("name", "") for loc in locations[:20]],
        })

    return profiles


# ---------------------------------------------------------------------------
# Medical
# ---------------------------------------------------------------------------


def compute_medical_profiles(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build patient profiles with diagnoses, medications, procedures."""
    patients = [e for e in entities if e.get("type", "").upper() in ("PATIENT", "PERSON")]
    diagnoses = [e for e in entities if e.get("type", "").upper() in (
        "DIAGNOSIS", "CONDITION", "DISEASE", "DISORDER"
    )]
    medications = [e for e in entities if e.get("type", "").upper() in (
        "MEDICATION", "DRUG", "PRESCRIPTION", "MEDICINE"
    )]
    procedures = [e for e in entities if e.get("type", "").upper() in (
        "PROCEDURE", "TREATMENT", "SURGERY", "TEST"
    )]

    patient_diagnoses: Dict[str, List[str]] = defaultdict(list)
    patient_meds: Dict[str, List[str]] = defaultdict(list)

    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rel_type = (rel.get("relation_type") or rel.get("relation", "")).upper()

        if rel_type in ("DIAGNOSED_WITH", "HAS_CONDITION", "SUFFERS_FROM"):
            patient_diagnoses[src].append(tgt)
        elif rel_type in ("PRESCRIBED", "TAKES", "ADMINISTERED"):
            patient_meds[src].append(tgt)

    profiles: List[Dict[str, Any]] = []
    for patient in patients:
        name = patient.get("name", "Unknown")
        profiles.append({
            "type": "patient",
            "label": name,
            "name": name,
            "diagnoses": sorted(set(patient_diagnoses.get(name, []))),
            "medications": sorted(set(patient_meds.get(name, []))),
        })

    if diagnoses:
        profiles.append({
            "type": "diagnoses_summary",
            "label": "Diagnoses Overview",
            "total": len(diagnoses),
            "conditions": sorted(set(d.get("name", "") for d in diagnoses))[:30],
        })

    if medications:
        profiles.append({
            "type": "medications_summary",
            "label": "Medications Overview",
            "total": len(medications),
            "drugs": sorted(set(m.get("name", "") for m in medications))[:30],
        })

    return profiles


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


def compute_generic_profiles(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build generic entity summaries when no domain is detected."""
    by_type: Dict[str, List[str]] = defaultdict(list)
    for e in entities:
        etype = e.get("type", "OTHER")
        name = e.get("name", "")
        if name:
            by_type[etype].append(name)

    profiles: List[Dict[str, Any]] = []
    for etype, names in sorted(by_type.items()):
        unique = sorted(set(names))
        profiles.append({
            "type": "entity_group",
            "label": etype,
            "entity_type": etype,
            "entities": unique[:30],
            "total": len(unique),
        })

    if relationships:
        profiles.append({
            "type": "relationship_summary",
            "label": "Relationships",
            "total": len(relationships),
            "sample": [
                f"{r.get('source', '?')} -> {r.get('target', '?')}"
                for r in relationships[:15]
            ],
        })

    return profiles


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_COMPUTE_DISPATCH = {
    "hr_recruitment": compute_hr_profiles,
    "finance": compute_finance_profiles,
    "legal": compute_legal_profiles,
    "logistics": compute_logistics_profiles,
    "medical": compute_medical_profiles,
    "generic": compute_generic_profiles,
}
