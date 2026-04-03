"""Alert detection — threshold-based alerts from profile intelligence and model responses.

Scans computed profiles for domain-specific threshold breaches and extracts
structured alerts from LLM responses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

VALID_SEVERITIES = frozenset({"critical", "warning", "info"})


@dataclass
class Alert:
    """A single actionable alert."""

    severity: str  # critical | warning | info
    category: str
    title: str
    detail: str
    action: str
    source: str

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            self.severity = "info"

    def to_dict(self) -> Dict[str, str]:
        return {
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "detail": self.detail,
            "action": self.action,
            "source": self.source,
        }


class AlertGenerator:
    """Generates alerts from profile intelligence and model responses."""

    # -- Response parsing ----------------------------------------------------

    @staticmethod
    def parse_alerts_from_response(response_text: str) -> List[Alert]:
        """Extract ``<alerts>`` JSON block from a model response.

        Expected format in model output::

            <alerts>
            [{"severity": "...", "category": "...", "title": "...",
              "detail": "...", "action": "...", "source": "..."}]
            </alerts>

        Returns an empty list if no alerts block is found or parsing fails.
        """
        pattern = re.compile(r"<alerts>\s*(.*?)\s*</alerts>", re.DOTALL)
        match = pattern.search(response_text)
        if not match:
            return []

        raw = match.group(1).strip()
        try:
            items = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse alerts JSON from response")
            return []

        if not isinstance(items, list):
            items = [items]

        alerts: List[Alert] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                alerts.append(Alert(
                    severity=item.get("severity", "info"),
                    category=item.get("category", "general"),
                    title=item.get("title", ""),
                    detail=item.get("detail", ""),
                    action=item.get("action", ""),
                    source=item.get("source", "model_response"),
                ))
            except Exception:
                continue

        return alerts

    # -- Intelligence-based alerts -------------------------------------------

    @staticmethod
    def generate_alerts_from_intelligence(
        intelligence: Any,  # ProfileIntelligence
    ) -> List[Alert]:
        """Scan computed profiles for threshold breaches.

        Domain-specific rules are applied based on ``intelligence.profile_type``.
        """
        alerts: List[Alert] = []
        profile_type = getattr(intelligence, "profile_type", "generic")
        computed = getattr(intelligence, "computed_profiles", [])
        insights = getattr(intelligence, "collection_insights", {})

        fn = _DOMAIN_ALERT_DISPATCH.get(profile_type)
        if fn:
            alerts.extend(fn(computed, insights))

        # Always check for generic anomalies from insights
        anomalies = insights.get("anomalies", [])
        for anomaly in anomalies:
            alerts.append(Alert(
                severity="warning",
                category="anomaly",
                title="Anomaly detected",
                detail=str(anomaly),
                action="Review the flagged data point for correctness",
                source="collection_insights",
            ))

        # Check gaps
        gaps = insights.get("gaps", [])
        for gap in gaps:
            alerts.append(Alert(
                severity="info",
                category="data_gap",
                title="Data gap identified",
                detail=str(gap),
                action="Consider supplementing the profile with additional documents",
                source="collection_insights",
            ))

        return alerts


# ---------------------------------------------------------------------------
# Domain-specific alert generators
# ---------------------------------------------------------------------------


def _logistics_alerts(
    profiles: List[Dict[str, Any]],
    insights: Dict[str, Any],
) -> List[Alert]:
    """Logistics: stock below reorder, expiry approaching, lead time issues."""
    alerts: List[Alert] = []
    for p in profiles:
        if p.get("type") != "product":
            continue
        name = p.get("name", "Unknown")

        # Stock below reorder point
        stock_levels = p.get("stock_levels", [])
        for level in stock_levels:
            try:
                qty = float(str(level).replace(",", ""))
            except (ValueError, TypeError):
                continue
            if qty < 10:
                alerts.append(Alert(
                    severity="critical",
                    category="low_stock",
                    title=f"Low stock: {name}",
                    detail=f"Current stock level ({qty}) is below reorder threshold",
                    action=f"Initiate reorder for {name} immediately",
                    source="computed_profiles",
                ))
            elif qty < 50:
                alerts.append(Alert(
                    severity="warning",
                    category="low_stock",
                    title=f"Stock declining: {name}",
                    detail=f"Current stock level ({qty}) approaching reorder threshold",
                    action=f"Plan reorder for {name}",
                    source="computed_profiles",
                ))

        # No suppliers
        if p.get("supplier_count", 0) == 0:
            alerts.append(Alert(
                severity="warning",
                category="supply_chain",
                title=f"No supplier for {name}",
                detail=f"Product {name} has no identified suppliers",
                action="Identify and link suppliers for this product",
                source="computed_profiles",
            ))

    return alerts


def _hr_alerts(
    profiles: List[Dict[str, Any]],
    insights: Dict[str, Any],
) -> List[Alert]:
    """HR: missing certifications, experience gaps."""
    alerts: List[Alert] = []
    for p in profiles:
        if p.get("type") != "candidate":
            continue
        name = p.get("name", "Unknown")

        # Missing certifications
        if not p.get("certifications"):
            alerts.append(Alert(
                severity="info",
                category="missing_certification",
                title=f"No certifications: {name}",
                detail=f"Candidate {name} has no certifications on record",
                action="Verify certification status with candidate",
                source="computed_profiles",
            ))

        # Limited experience
        fit = p.get("role_fit_indicators", {})
        if fit.get("experience_depth", 0) == 0:
            alerts.append(Alert(
                severity="warning",
                category="experience_gap",
                title=f"No experience history: {name}",
                detail=f"Candidate {name} has no linked organisations/experience",
                action="Request detailed work history from candidate",
                source="computed_profiles",
            ))

        # Very few skills
        if fit.get("skill_breadth", 0) < 2:
            alerts.append(Alert(
                severity="info",
                category="skill_gap",
                title=f"Limited skills profile: {name}",
                detail=f"Candidate {name} has fewer than 2 skills identified",
                action="Review resume for additional skill extraction",
                source="computed_profiles",
            ))

    return alerts


def _finance_alerts(
    profiles: List[Dict[str, Any]],
    insights: Dict[str, Any],
) -> List[Alert]:
    """Finance: overdue payments, anomalous amounts."""
    alerts: List[Alert] = []
    for p in profiles:
        if p.get("type") == "vendor":
            name = p.get("name", "Unknown")
            # Many transactions from single vendor
            if p.get("transaction_count", 0) > 20:
                alerts.append(Alert(
                    severity="info",
                    category="vendor_concentration",
                    title=f"High transaction volume: {name}",
                    detail=f"Vendor {name} has {p['transaction_count']} transactions",
                    action="Review vendor concentration risk",
                    source="computed_profiles",
                ))

        if p.get("type") == "financial_summary":
            amounts = p.get("amounts", [])
            parsed: List[float] = []
            for a in amounts:
                cleaned = str(a).replace(",", "").replace("$", "").replace("EUR", "").replace("GBP", "").strip()
                try:
                    parsed.append(float(cleaned))
                except (ValueError, TypeError):
                    continue
            if parsed:
                avg = sum(parsed) / len(parsed)
                for val in parsed:
                    if val > avg * 5 and avg > 0:
                        alerts.append(Alert(
                            severity="warning",
                            category="anomalous_amount",
                            title="Anomalous transaction amount",
                            detail=f"Amount {val:,.2f} is >5x the average ({avg:,.2f})",
                            action="Verify this transaction for correctness",
                            source="computed_profiles",
                        ))

    return alerts


def _legal_alerts(
    profiles: List[Dict[str, Any]],
    insights: Dict[str, Any],
) -> List[Alert]:
    """Legal: obligation deadlines, contract renewals approaching."""
    alerts: List[Alert] = []
    now = datetime.utcnow()
    threshold = now + timedelta(days=30)

    for p in profiles:
        if p.get("type") == "contract_timeline":
            dates = p.get("dates", [])
            for d in dates:
                date_type = d.get("type", "").upper()
                date_val = d.get("value", "")
                parsed_date = _try_parse_date(date_val)
                if parsed_date is None:
                    continue

                if date_type in ("DEADLINE", "EXPIRY_DATE", "RENEWAL_DATE"):
                    if parsed_date <= now:
                        alerts.append(Alert(
                            severity="critical",
                            category="deadline_passed",
                            title=f"{date_type.replace('_', ' ').title()} has passed",
                            detail=f"{date_type}: {date_val} is in the past",
                            action="Immediate review required — deadline may have been missed",
                            source="computed_profiles",
                        ))
                    elif parsed_date <= threshold:
                        alerts.append(Alert(
                            severity="warning",
                            category="deadline_approaching",
                            title=f"{date_type.replace('_', ' ').title()} approaching",
                            detail=f"{date_type}: {date_val} is within 30 days",
                            action="Prepare for upcoming deadline and notify stakeholders",
                            source="computed_profiles",
                        ))

        if p.get("type") == "obligations_summary":
            if p.get("total", 0) > 20:
                alerts.append(Alert(
                    severity="info",
                    category="obligation_volume",
                    title="High number of obligations",
                    detail=f"{p['total']} obligations identified across contracts",
                    action="Consider creating a compliance tracking matrix",
                    source="computed_profiles",
                ))

    return alerts


def _medical_alerts(
    profiles: List[Dict[str, Any]],
    insights: Dict[str, Any],
) -> List[Alert]:
    """Medical: medication interactions, guideline deviations, follow-up overdue."""
    alerts: List[Alert] = []

    # Known high-risk medication combinations (simplified)
    _INTERACTION_PAIRS = {
        frozenset({"warfarin", "aspirin"}),
        frozenset({"metformin", "contrast dye"}),
        frozenset({"ssri", "maoi"}),
        frozenset({"ace inhibitor", "potassium"}),
        frozenset({"lithium", "nsaid"}),
    }

    for p in profiles:
        if p.get("type") == "patient":
            name = p.get("name", "Unknown")
            meds = [m.lower() for m in p.get("medications", [])]

            # Check for known interactions
            for pair in _INTERACTION_PAIRS:
                matches = [m for m in meds if any(drug in m for drug in pair)]
                if len(matches) >= 2:
                    alerts.append(Alert(
                        severity="critical",
                        category="medication_interaction",
                        title=f"Potential drug interaction: {name}",
                        detail=f"Medications {', '.join(matches)} may interact",
                        action="Review medication combination with prescribing physician",
                        source="computed_profiles",
                    ))

            # No diagnoses but has medications
            if meds and not p.get("diagnoses"):
                alerts.append(Alert(
                    severity="warning",
                    category="missing_diagnosis",
                    title=f"Medications without diagnosis: {name}",
                    detail=f"Patient {name} has {len(meds)} medication(s) but no linked diagnosis",
                    action="Verify diagnosis records are complete",
                    source="computed_profiles",
                ))

        if p.get("type") == "diagnoses_summary":
            conditions = p.get("conditions", [])
            if len(conditions) > 10:
                alerts.append(Alert(
                    severity="info",
                    category="complex_case",
                    title="Complex patient cohort",
                    detail=f"{len(conditions)} distinct conditions across patient records",
                    action="Consider specialist review for complex cases",
                    source="computed_profiles",
                ))

    return alerts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_parse_date(value: str) -> Optional[datetime]:
    """Attempt to parse a date string in common formats."""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


_DOMAIN_ALERT_DISPATCH = {
    "logistics": _logistics_alerts,
    "hr_recruitment": _hr_alerts,
    "finance": _finance_alerts,
    "legal": _legal_alerts,
    "medical": _medical_alerts,
}
