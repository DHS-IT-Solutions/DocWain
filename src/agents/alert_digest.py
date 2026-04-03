"""Alert digest formatting and storage for DocWain V2.

Takes a list of alerts and profile intelligence, produces a structured
digest suitable for delivery via Teams, email, or the UI alert banner.

Usage::

    digest_builder = AlertDigest()
    digest = digest_builder.format_digest(alerts, intelligence)
    digest_builder.store_digest(digest, profile_id, mongo_client)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AlertDigest:
    """Formats and stores alert digests."""

    def format_digest(
        self,
        alerts: List[Any],
        profile_intelligence: Any,
    ) -> Dict[str, Any]:
        """Format alerts into a structured digest.

        Parameters
        ----------
        alerts:
            List of Alert objects or dicts with severity/category/title/detail/action/source.
        profile_intelligence:
            ProfileIntelligence object with profile metadata.

        Returns
        -------
        Structured digest dict.
        """
        # Normalise alerts to dicts
        alert_dicts = []
        for a in alerts:
            if hasattr(a, "severity"):
                alert_dicts.append({
                    "severity": a.severity,
                    "category": a.category,
                    "title": a.title,
                    "detail": a.detail,
                    "action": a.action,
                    "source": a.source,
                })
            elif isinstance(a, dict):
                alert_dicts.append(a)

        # Count by severity
        critical = [a for a in alert_dicts if a.get("severity") == "critical"]
        warning = [a for a in alert_dicts if a.get("severity") == "warning"]
        info = [a for a in alert_dicts if a.get("severity") == "info"]

        # Build summary line
        parts = []
        if critical:
            parts.append(f"{len(critical)} critical")
        if warning:
            parts.append(f"{len(warning)} warning")
        if info:
            parts.append(f"{len(info)} informational")
        summary = ", ".join(parts) if parts else "No alerts"

        # Profile context
        profile_summary = {}
        if profile_intelligence:
            pi = profile_intelligence
            profile_summary = {
                "profile_id": getattr(pi, "profile_id", ""),
                "profile_type": getattr(pi, "profile_type", "generic"),
                "document_count": getattr(pi, "document_count", 0),
                "domain": getattr(pi, "domain_metadata", {}).get(
                    "detected_domain", "generic"
                ) if hasattr(pi, "domain_metadata") else "generic",
            }

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "critical_count": len(critical),
            "warning_count": len(warning),
            "info_count": len(info),
            "total_count": len(alert_dicts),
            "alerts": {
                "critical": critical,
                "warning": warning,
                "info": info,
            },
            "profile_summary": profile_summary,
        }

    def store_digest(
        self,
        digest: Dict[str, Any],
        profile_id: str,
        mongo_client: Any,
    ) -> None:
        """Store the digest in MongoDB for later retrieval.

        Stored in the ``alert_digests`` collection with the profile_id
        and timestamp as the key.
        """
        try:
            db = mongo_client.get_database()
            doc = {
                "profile_id": profile_id,
                **digest,
            }
            db["alert_digests"].insert_one(doc)
            logger.info(
                "Alert digest stored for profile %s (%d alerts)",
                profile_id, digest.get("total_count", 0),
            )
        except Exception as exc:
            logger.error(
                "Failed to store digest for %s: %s", profile_id, exc
            )

    def get_latest_digest(
        self,
        profile_id: str,
        mongo_client: Any,
    ) -> Dict[str, Any]:
        """Retrieve the most recent digest for a profile."""
        try:
            db = mongo_client.get_database()
            doc = db["alert_digests"].find_one(
                {"profile_id": profile_id},
                sort=[("generated_at", -1)],
            )
            if doc:
                doc.pop("_id", None)
                return doc
        except Exception as exc:
            logger.error(
                "Failed to retrieve digest for %s: %s", profile_id, exc
            )
        return {}

    def format_teams_card(self, digest: Dict[str, Any]) -> Dict[str, Any]:
        """Format digest as a Microsoft Teams Adaptive Card.

        Returns an Adaptive Card JSON payload ready for posting to Teams.
        """
        summary = digest.get("summary", "No alerts")
        profile = digest.get("profile_summary", {})

        # Build alert items for the card
        alert_items = []
        for severity in ["critical", "warning", "info"]:
            color = {"critical": "attention", "warning": "warning", "info": "default"}
            for alert in digest.get("alerts", {}).get(severity, []):
                alert_items.append({
                    "type": "TextBlock",
                    "text": f"**[{severity.upper()}]** {alert.get('title', '')}",
                    "color": color.get(severity, "default"),
                    "wrap": True,
                })
                if alert.get("action"):
                    alert_items.append({
                        "type": "TextBlock",
                        "text": f"Action: {alert['action']}",
                        "size": "Small",
                        "isSubtle": True,
                        "wrap": True,
                    })

        card = {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "text": f"DocWain Alert Digest — {profile.get('profile_type', 'Profile')}",
                    "weight": "Bolder",
                    "size": "Medium",
                },
                {
                    "type": "TextBlock",
                    "text": summary,
                    "wrap": True,
                },
                {"type": "TextBlock", "text": "---"},
                *alert_items,
            ],
        }
        return card
