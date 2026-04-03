"""Scheduled analysis agent for DocWain V2.

Runs per-profile on a configurable schedule (daily/weekly).  Loads
pre-computed ProfileIntelligence, generates threshold-based alerts,
and stores an alert digest in MongoDB for delivery via Teams/email/UI.

Usage::

    agent = ScheduledAnalysisAgent()
    digest = agent.run_analysis(profile_id, subscription_id, mongo_client)
    agent.run_all_profiles(mongo_client)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ScheduledAnalysisAgent:
    """Background agent that analyses profiles and generates alert digests.

    Parameters
    ----------
    query_pipeline:
        Optional reference to ``run_query_pipeline`` for deep analysis.
        If not provided, only threshold-based alerts are generated.
    """

    def __init__(self, query_pipeline=None) -> None:
        self.query_pipeline = query_pipeline

    def run_analysis(
        self,
        profile_id: str,
        subscription_id: str,
        mongo_client: Any,
        *,
        kg_client: Any = None,
    ) -> Dict[str, Any]:
        """Run analysis for a single profile.

        Steps:
        1. Load or build ProfileIntelligence
        2. Generate alerts from computed profiles (threshold checks)
        3. Optionally run a meta-query through the smart pipeline
        4. Combine alerts and store digest in MongoDB
        5. Return the digest
        """
        from src.intelligence_v2.profile_builder import ProfileBuilder
        from src.intelligence_v2.alert_generator import AlertGenerator
        from src.agents.alert_digest import AlertDigest

        builder = ProfileBuilder()
        alert_gen = AlertGenerator()
        digest_builder = AlertDigest()

        # 1. Load or build intelligence
        intelligence = builder.get_cached(profile_id, mongo_client)
        if intelligence is None:
            logger.info("No cached intelligence for %s, building...", profile_id)
            intelligence = builder.build(
                profile_id, subscription_id, mongo_client, kg_client
            )

        # 2. Generate threshold-based alerts
        alerts = alert_gen.generate_alerts_from_intelligence(intelligence)
        logger.info(
            "Profile %s: %d alerts generated (threshold-based)",
            profile_id, len(alerts),
        )

        # 3. Optional: deep analysis via query pipeline
        pipeline_alerts = []
        if self.query_pipeline is not None:
            try:
                meta_query = (
                    "Analyze all documents in this profile for items "
                    "requiring immediate attention. Check for deadlines, "
                    "threshold breaches, gaps, anomalies, and compliance issues."
                )
                result = self.query_pipeline(
                    query=meta_query,
                    profile_id=profile_id,
                    subscription_id=subscription_id,
                    clients=None,
                    profile_intelligence=intelligence,
                )
                if result and hasattr(result, "alerts") and result.alerts:
                    pipeline_alerts = result.alerts
                    logger.info(
                        "Profile %s: %d additional alerts from pipeline analysis",
                        profile_id, len(pipeline_alerts),
                    )
            except Exception as exc:
                logger.warning(
                    "Pipeline analysis failed for %s: %s", profile_id, exc
                )

        # 4. Combine and deduplicate alerts
        all_alerts = alerts + pipeline_alerts
        seen_titles = set()
        unique_alerts = []
        for alert in all_alerts:
            title = alert.title if hasattr(alert, "title") else alert.get("title", "")
            if title not in seen_titles:
                seen_titles.add(title)
                unique_alerts.append(alert)

        # 5. Build and store digest
        digest = digest_builder.format_digest(unique_alerts, intelligence)
        digest_builder.store_digest(digest, profile_id, mongo_client)

        logger.info(
            "Profile %s analysis complete: %d alerts, digest stored",
            profile_id, len(unique_alerts),
        )
        return digest

    def run_all_profiles(
        self,
        mongo_client: Any,
        *,
        kg_client: Any = None,
        subscription_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run analysis for all active profiles.

        Parameters
        ----------
        mongo_client:
            MongoDB client instance.
        kg_client:
            Optional Neo4j client.
        subscription_filter:
            If set, only analyse profiles in this subscription.

        Returns
        -------
        List of digests, one per profile.
        """
        try:
            db = mongo_client.get_database()
            query: Dict[str, Any] = {"status": {"$ne": "deleted"}}
            if subscription_filter:
                query["subscription_id"] = subscription_filter

            profiles = list(
                db["profiles"].find(query, {"_id": 0, "profile_id": 1, "subscription_id": 1})
            )
        except Exception as exc:
            logger.error("Failed to list profiles: %s", exc)
            return []

        logger.info("Running scheduled analysis for %d profiles", len(profiles))

        digests = []
        for profile in profiles:
            pid = profile.get("profile_id", "")
            sid = profile.get("subscription_id", "")
            if not pid or not sid:
                continue
            try:
                digest = self.run_analysis(pid, sid, mongo_client, kg_client=kg_client)
                digests.append(digest)
            except Exception as exc:
                logger.error("Analysis failed for profile %s: %s", pid, exc)

        logger.info(
            "Scheduled analysis complete: %d/%d profiles processed",
            len(digests), len(profiles),
        )
        return digests
