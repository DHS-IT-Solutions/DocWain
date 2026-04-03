"""Scheduled Analysis Agent — runs per-profile intelligence on a schedule.

Loads (or builds) ProfileIntelligence, generates alerts, optionally runs a
meta-query through the ask pipeline, and stores an alert digest in MongoDB.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger
from src.intelligence.profile_builder import ProfileBuilder, ProfileIntelligence
from src.intelligence.alert_generator import generate_alerts_from_intelligence, Alert
from src.agents.alert_digest import AlertDigest

logger = get_logger(__name__)


class ScheduledAnalysisAgent:
    """Background agent that analyses each profile on a schedule."""

    def __init__(self) -> None:
        self._profile_builder = ProfileBuilder()
        self._digest_builder = AlertDigest()

    # ------------------------------------------------------------------
    # Single-profile analysis
    # ------------------------------------------------------------------

    def run_analysis(
        self,
        profile_id: str,
        subscription_id: str,
        mongo_client: Any,
        query_pipeline: Any = None,
    ) -> Dict[str, Any]:
        """Run a full analysis cycle for one profile.

        Steps:
            1. Load (or build) :class:`ProfileIntelligence`.
            2. Generate alerts from the intelligence data.
            3. Optionally run a meta-query through *query_pipeline*.
            4. Format and store an alert digest in MongoDB.
            5. Return the digest dict.

        Args:
            profile_id: The profile to analyse.
            subscription_id: Owning subscription.
            mongo_client: A pymongo ``MongoClient`` (or compatible).
            query_pipeline: Optional callable/object with an ``ask(query, profile_id, subscription_id)``
                method.  When provided, the agent will send a meta-query
                asking the pipeline to surface items requiring attention.

        Returns:
            The stored digest dict (see :class:`AlertDigest`).
        """
        logger.info("Scheduled analysis starting for profile=%s", profile_id)

        # 1. Intelligence
        intelligence = self._profile_builder.get_cached(profile_id, mongo_client)
        if intelligence is None:
            intelligence = self._profile_builder.build(
                profile_id, subscription_id, mongo_client
            )

        # 2. Alerts from rules
        alerts = generate_alerts_from_intelligence(intelligence)

        # 3. Optional meta-query
        if query_pipeline is not None:
            pipeline_alerts = self._run_meta_query(
                query_pipeline, profile_id, subscription_id
            )
            alerts.extend(pipeline_alerts)

        # 4 & 5. Digest
        digest = self._digest_builder.format_digest(alerts, intelligence)
        self._digest_builder.store_digest(digest, profile_id, mongo_client)

        logger.info(
            "Scheduled analysis complete for profile=%s — %d alerts (%d critical)",
            profile_id,
            len(alerts),
            digest.get("critical_count", 0),
        )
        return digest

    # ------------------------------------------------------------------
    # All-profiles sweep
    # ------------------------------------------------------------------

    def run_all_profiles(self, mongo_client: Any) -> List[Dict[str, Any]]:
        """Iterate all active profiles and run analysis for each.

        Returns a list of digest dicts (one per profile).
        """
        profiles = self._list_active_profiles(mongo_client)
        logger.info("Scheduled sweep: %d active profiles found", len(profiles))

        results: List[Dict[str, Any]] = []
        for prof in profiles:
            profile_id = prof.get("profile_id", "")
            subscription_id = prof.get("subscription_id", "")
            if not profile_id:
                continue
            try:
                digest = self.run_analysis(
                    profile_id, subscription_id, mongo_client
                )
                results.append(digest)
            except Exception:
                logger.exception(
                    "Scheduled analysis failed for profile=%s", profile_id
                )
                results.append({
                    "profile_id": profile_id,
                    "error": traceback.format_exc(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _list_active_profiles(mongo_client: Any) -> List[Dict]:
        """Retrieve all active profiles from MongoDB."""
        try:
            db = (
                mongo_client.get_default_database()
                if hasattr(mongo_client, "get_default_database")
                else mongo_client.docwain
            )
            return list(
                db.profiles.find(
                    {"status": {"$ne": "archived"}},
                    {"profile_id": 1, "subscription_id": 1, "_id": 0},
                )
            )
        except Exception:
            logger.exception("Error listing active profiles")
            return []

    @staticmethod
    def _run_meta_query(
        query_pipeline: Any,
        profile_id: str,
        subscription_id: str,
    ) -> List[Alert]:
        """Send a meta-query and extract alerts from the response."""
        from src.intelligence.alert_generator import parse_alerts_from_response

        meta_query = (
            "Analyze all documents in this profile and identify any items "
            "requiring immediate attention, potential risks, anomalies, or "
            "action items.  Wrap findings in <alerts> JSON tags."
        )
        try:
            if hasattr(query_pipeline, "ask"):
                response = query_pipeline.ask(
                    query=meta_query,
                    profile_id=profile_id,
                    subscription_id=subscription_id,
                )
            elif callable(query_pipeline):
                response = query_pipeline(meta_query, profile_id, subscription_id)
            else:
                logger.warning("query_pipeline has no 'ask' method and is not callable")
                return []

            response_text = (
                response if isinstance(response, str)
                else response.get("answer", response.get("response", ""))
                if isinstance(response, dict)
                else str(response)
            )
            return parse_alerts_from_response(response_text)
        except Exception:
            logger.exception("Meta-query failed for profile %s", profile_id)
            return []
