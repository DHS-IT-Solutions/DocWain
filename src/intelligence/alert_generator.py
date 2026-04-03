"""Alert generation from intelligence data and LLM responses.

Provides two entry-points:
1. ``parse_alerts_from_response`` — extract ``<alerts>`` JSON from model output.
2. ``generate_alerts_from_intelligence`` — rule-based scan of *ProfileIntelligence*
   for threshold breaches and domain-specific conditions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """Single actionable alert."""

    severity: str  # critical | warning | info
    category: str
    title: str
    detail: str
    action: str
    source: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Parse alerts embedded in model responses
# ---------------------------------------------------------------------------

_ALERTS_PATTERN = re.compile(
    r"<alerts>\s*(.*?)\s*</alerts>", re.DOTALL
)


def parse_alerts_from_response(response_text: str) -> List[Alert]:
    """Extract ``<alerts>`` JSON block(s) from a model response string.

    Expected format inside ``<alerts>``::

        [
          {"severity": "warning", "category": "...", "title": "...",
           "detail": "...", "action": "...", "source": "..."},
          ...
        ]

    Returns a list of :class:`Alert` instances.  Malformed blocks are logged
    and skipped.
    """
    alerts: List[Alert] = []
    for match in _ALERTS_PATTERN.finditer(response_text):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                logger.warning("Alerts block is not a list or dict, skipping")
                continue
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                alerts.append(
                    Alert(
                        severity=str(item.get("severity", "info")).lower(),
                        category=str(item.get("category", "general")),
                        title=str(item.get("title", "")),
                        detail=str(item.get("detail", "")),
                        action=str(item.get("action", "")),
                        source=str(item.get("source", "model_response")),
                    )
                )
        except json.JSONDecodeError:
            logger.warning("Failed to parse alerts JSON block: %.120s", raw)
    return alerts


# ---------------------------------------------------------------------------
# Rule-based alert generation from ProfileIntelligence
# ---------------------------------------------------------------------------

# Severity helpers
_CRITICAL = "critical"
_WARNING = "warning"
_INFO = "info"


def _alerts_from_anomalies(anomalies: List[str], source: str) -> List[Alert]:
    """Convert anomaly strings into warning alerts."""
    return [
        Alert(
            severity=_WARNING,
            category="anomaly",
            title="Statistical anomaly detected",
            detail=anomaly,
            action="Review the flagged item and verify data accuracy.",
            source=source,
        )
        for anomaly in anomalies
    ]


def _alerts_from_gaps(gaps: List[str], source: str) -> List[Alert]:
    """Convert gap strings into info alerts."""
    return [
        Alert(
            severity=_INFO,
            category="data_gap",
            title="Missing data detected",
            detail=gap,
            action="Consider enriching records with the missing information.",
            source=source,
        )
        for gap in gaps
    ]


# -- Domain-specific scanners ------------------------------------------------


def _hr_alerts(profiles: List[Dict], source: str) -> List[Alert]:
    alerts: List[Alert] = []
    for p in profiles:
        exp = p.get("experience_years")
        if exp is not None:
            try:
                if float(exp) < 1:
                    alerts.append(
                        Alert(_WARNING, "hr", "Low experience candidate",
                              f"Candidate '{p.get('name', '?')}' has <1 year experience.",
                              "Verify if role accepts entry-level candidates.", source)
                    )
            except (TypeError, ValueError):
                pass
        certs = p.get("certifications")
        if isinstance(certs, list) and len(certs) == 0:
            alerts.append(
                Alert(_INFO, "hr", "No certifications listed",
                      f"Candidate '{p.get('name', '?')}' lists no certifications.",
                      "Consider requesting certification details.", source)
            )
    return alerts


def _finance_alerts(profiles: List[Dict], source: str) -> List[Alert]:
    alerts: List[Alert] = []
    for p in profiles:
        spend = p.get("total_spend")
        if spend is not None:
            try:
                if float(spend) > 100_000:
                    alerts.append(
                        Alert(_WARNING, "finance", "High vendor spend",
                              f"Vendor '{p.get('name', '?')}' total spend ${float(spend):,.2f}.",
                              "Review vendor contract and negotiate terms.", source)
                    )
            except (TypeError, ValueError):
                pass
        terms = p.get("avg_payment_terms")
        if terms is not None:
            try:
                if float(terms) > 60:
                    alerts.append(
                        Alert(_WARNING, "finance", "Extended payment terms",
                              f"Vendor '{p.get('name', '?')}' avg terms {float(terms):.0f} days.",
                              "Assess cash-flow impact of extended terms.", source)
                    )
            except (TypeError, ValueError):
                pass
    return alerts


def _legal_alerts(profiles: List[Dict], source: str) -> List[Alert]:
    alerts: List[Alert] = []
    for p in profiles:
        risk = str(p.get("risk_level", "")).lower()
        if risk == "critical":
            alerts.append(
                Alert(_CRITICAL, "legal", "Critical risk contract",
                      f"Contract '{p.get('name', '?')}' rated CRITICAL risk.",
                      "Escalate to legal counsel immediately.", source)
            )
        elif risk == "high":
            alerts.append(
                Alert(_WARNING, "legal", "High risk contract",
                      f"Contract '{p.get('name', '?')}' rated HIGH risk.",
                      "Schedule legal review before renewal.", source)
            )
        obligations = p.get("obligations")
        if isinstance(obligations, list) and len(obligations) > 10:
            alerts.append(
                Alert(_WARNING, "legal", "Complex obligations",
                      f"Contract '{p.get('name', '?')}' has {len(obligations)} obligations.",
                      "Ensure compliance tracking is in place.", source)
            )
    return alerts


def _logistics_alerts(profiles: List[Dict], source: str) -> List[Alert]:
    alerts: List[Alert] = []
    for p in profiles:
        stock = p.get("stock_level")
        reorder = p.get("reorder_point")
        if stock is not None and reorder is not None:
            try:
                s, r = float(stock), float(reorder)
                if s <= 0:
                    alerts.append(
                        Alert(_CRITICAL, "logistics", "Out of stock",
                              f"Product '{p.get('name', '?')}' stock is {s}.",
                              "Place emergency reorder immediately.", source)
                    )
                elif s <= r:
                    alerts.append(
                        Alert(_WARNING, "logistics", "Stock below reorder point",
                              f"Product '{p.get('name', '?')}' stock {s} <= reorder {r}.",
                              "Initiate purchase order with supplier.", source)
                    )
            except (TypeError, ValueError):
                pass
        lead = p.get("lead_time_days")
        if lead is not None:
            try:
                if float(lead) > 30:
                    alerts.append(
                        Alert(_INFO, "logistics", "Long lead time",
                              f"Product '{p.get('name', '?')}' lead time {float(lead):.0f} days.",
                              "Consider alternative suppliers or buffer stock.", source)
                    )
            except (TypeError, ValueError):
                pass
    return alerts


def _medical_alerts(profiles: List[Dict], source: str) -> List[Alert]:
    alerts: List[Alert] = []
    for p in profiles:
        meds = p.get("medications")
        if isinstance(meds, list) and len(meds) > 10:
            alerts.append(
                Alert(_WARNING, "medical", "Polypharmacy risk",
                      f"Patient '{p.get('name', '?')}' has {len(meds)} medications.",
                      "Review for drug interactions and necessity.", source)
            )
        diagnoses = p.get("diagnoses")
        if isinstance(diagnoses, list) and len(diagnoses) > 5:
            alerts.append(
                Alert(_INFO, "medical", "Multiple diagnoses",
                      f"Patient '{p.get('name', '?')}' has {len(diagnoses)} diagnoses.",
                      "Ensure coordinated care plan is in place.", source)
            )
    return alerts


_DOMAIN_ALERT_FN = {
    "hr_recruitment": _hr_alerts,
    "finance": _finance_alerts,
    "legal": _legal_alerts,
    "logistics": _logistics_alerts,
    "medical": _medical_alerts,
}


def generate_alerts_from_intelligence(intelligence: Any) -> List[Alert]:
    """Scan a :class:`ProfileIntelligence` instance for threshold breaches.

    Applies domain-specific rules based on ``intelligence.profile_type`` and
    generic anomaly/gap checks from ``intelligence.collection_insights``.

    Args:
        intelligence: A ``ProfileIntelligence`` dataclass (from profile_builder).

    Returns:
        List of :class:`Alert` instances.
    """
    alerts: List[Alert] = []
    source = f"profile:{getattr(intelligence, 'profile_id', 'unknown')}"

    # --- Collection-level anomalies / gaps ----------------------------------
    insights = getattr(intelligence, "collection_insights", {}) or {}
    alerts.extend(_alerts_from_anomalies(insights.get("anomalies", []), source))
    alerts.extend(_alerts_from_gaps(insights.get("gaps", []), source))

    # --- Domain-specific profile scanning -----------------------------------
    profile_type = getattr(intelligence, "profile_type", "generic")
    computed = getattr(intelligence, "computed_profiles", []) or []
    fn = _DOMAIN_ALERT_FN.get(profile_type)
    if fn and computed:
        try:
            alerts.extend(fn(computed, source))
        except Exception:
            logger.exception("Error in domain alert generation for %s", profile_type)

    # --- Low document count -------------------------------------------------
    doc_count = getattr(intelligence, "document_count", 0)
    if doc_count == 1:
        alerts.append(
            Alert(_INFO, "general", "Single document profile",
                  "Profile contains only 1 document; cross-document analysis unavailable.",
                  "Upload additional documents for richer insights.", source)
        )

    logger.info("Generated %d alerts for profile %s", len(alerts), source)
    return alerts
