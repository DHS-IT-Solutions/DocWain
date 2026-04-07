"""AAD tenant auto-provisioning from Teams activity context."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from qdrant_client.models import Distance, VectorParams

from teams_app.storage.namespace import qdrant_collection_name

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024


class TenantManager:
    """Auto-provisions tenants and users on first contact."""

    def __init__(self, db: Any, qdrant_client: Any):
        self.db = db
        self.qdrant = qdrant_client

    @staticmethod
    def extract_identity(activity: Dict[str, Any]) -> Tuple[Optional[str], str, str]:
        """Extract tenant_id, user_id, display_name from a Teams activity dict."""
        tenant_id = None
        channel_data = activity.get("channelData") or {}
        tenant_info = channel_data.get("tenant") or {}
        tenant_id = tenant_info.get("id")

        from_info = activity.get("from") or {}
        user_id = from_info.get("id", "unknown")
        display_name = from_info.get("name", "")
        return tenant_id, user_id, display_name

    def ensure_tenant(self, tenant_id: str, display_name: str = "") -> Dict[str, Any]:
        """Get or create a tenant record. Creates Qdrant collection if new."""
        existing = self.db.teams_tenants.find_one({"tenant_id": tenant_id})
        if existing:
            return existing

        collection = qdrant_collection_name(tenant_id)
        record = {
            "tenant_id": tenant_id,
            "display_name": display_name,
            "qdrant_collection": collection,
            "settings": {
                "kg_enabled": True,
                "max_documents": 1000,
                "express_pipeline": True,
            },
            "document_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.db.teams_tenants.insert_one(record)

        try:
            self.qdrant.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection %s for tenant %s", collection, tenant_id)
        except Exception:
            logger.debug("Qdrant collection %s already exists", collection)

        return record

    def ensure_user(self, user_id: str, tenant_id: str, display_name: str = "") -> Dict[str, Any]:
        """Get or create a user record."""
        existing = self.db.teams_users.find_one({"user_id": user_id, "tenant_id": tenant_id})
        if existing:
            return existing

        record = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "display_name": display_name,
            "first_seen": datetime.now(timezone.utc).isoformat(),
        }
        self.db.teams_users.insert_one(record)
        logger.info("Auto-provisioned user %s for tenant %s", user_id, tenant_id)
        return record
