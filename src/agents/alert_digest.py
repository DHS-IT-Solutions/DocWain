"""Alert Digest — aggregation, formatting, and storage of alert batches.

Consumes a list of :class:`Alert` instances together with the corresponding
:class:`ProfileIntelligence` and produces a structured digest suitable for
API responses, email notifications, or dashboard display.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AlertDigest:
    """Formats and persists alert digests for a profile."""

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_digest(
        self,
        alerts: List[Any],
        profile_intelligence: Any,
    ) -> Dict[str, Any]:
        """Build a structured digest from alerts and profile intelligence.

        Args:
            alerts: List of :class:`Alert` dataclass instances (or dicts).
            profile_intelligence: A :class:`ProfileIntelligence` instance.

        Returns:
            A dict with keys: ``summary``, ``critical_count``, ``warning_count``,
            ``info_count``, ``alerts``, ``profile_summary``, ``timestamp``.
        """
        alert_dicts = self._normalise_alerts(alerts)

        critical = [a for a in alert_dicts if a["severity"] == "critical"]
        warnings = [a for a in alert_dicts if a["severity"] == "warning"]
        infos = [a for a in alert_dicts if a["severity"] == "info"]

        # Build human-readable summary
        profile_id = getattr(profile_intelligence, "profile_id", "unknown")
        profile_type = getattr(profile_intelligence, "profile_type", "generic")
        doc_count = getattr(profile_intelligence, "document_count", 0)

        summary_parts: List[str] = []
        if critical:
            summary_parts.append(f"{len(critical)} critical")
        if warnings:
            summary_parts.append(f"{len(warnings)} warning(s)")
        if infos:
            summary_parts.append(f"{len(infos)} informational")
        summary_line = (
            f"Profile '{profile_id}' ({profile_type}, {doc_count} docs): "
            + (", ".join(summary_parts) if summary_parts else "no alerts")
            + "."
        )

        # Profile summary block
        profile_summary = {
            "profile_id": profile_id,
            "profile_type": profile_type,
            "document_count": doc_count,
            "entities_total": (
                getattr(profile_intelligence, "entities_summary", {}).get("total", 0)
                if hasattr(profile_intelligence, "entities_summary")
                else 0
            ),
            "patterns_count": len(
                getattr(profile_intelligence, "collection_insights", {}).get("patterns", [])
                if hasattr(profile_intelligence, "collection_insights")
                else []
            ),
        }

        # Sort alerts: critical first, then warning, then info
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        sorted_alerts = sorted(
            alert_dicts, key=lambda a: severity_order.get(a["severity"], 9)
        )

        return {
            "summary": summary_line,
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "info_count": len(infos),
            "alerts": sorted_alerts,
            "profile_summary": profile_summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_digest(
        self,
        digest: Dict[str, Any],
        profile_id: str,
        mongo_client: Any,
    ) -> None:
        """Persist the digest in MongoDB ``alert_digests`` collection.

        Each digest is inserted as a new document (not upserted) so that
        historical digests are preserved for trend analysis.

        Args:
            digest: The digest dict produced by :meth:`format_digest`.
            profile_id: Profile identifier.
            mongo_client: A pymongo ``MongoClient`` (or compatible).
        """
        try:
            db = (
                mongo_client.get_default_database()
                if hasattr(mongo_client, "get_default_database")
                else mongo_client.docwain
            )
            doc = {
                "profile_id": profile_id,
                **digest,
            }
            db.alert_digests.insert_one(doc)
            logger.info(
                "Stored alert digest for profile=%s (%d alerts)",
                profile_id,
                len(digest.get("alerts", [])),
            )
        except Exception:
            logger.exception("Failed to store alert digest for profile=%s", profile_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_alerts(alerts: List[Any]) -> List[Dict[str, str]]:
        """Convert Alert dataclass instances (or dicts) to plain dicts."""
        result: List[Dict[str, str]] = []
        for a in alerts:
            if isinstance(a, dict):
                result.append(a)
            elif hasattr(a, "to_dict"):
                result.append(a.to_dict())
            else:
                # Fallback: read dataclass fields
                result.append({
                    "severity": getattr(a, "severity", "info"),
                    "category": getattr(a, "category", "general"),
                    "title": getattr(a, "title", ""),
                    "detail": getattr(a, "detail", ""),
                    "action": getattr(a, "action", ""),
                    "source": getattr(a, "source", "unknown"),
                })
        return result
