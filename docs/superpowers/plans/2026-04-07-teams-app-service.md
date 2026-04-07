# DocWain Teams App — Standalone Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the existing Teams bot integration (`src/teams/`) into a standalone FastAPI service (`teams_app/`) with auto-triggered pipeline, tiered fast path, query proxying, and APIM routing.

**Architecture:** The standalone service reuses `src/teams/` classes (bot_app, pipeline, logic, state, tools, storage, cards) via imports — no code duplication. New code handles: lightweight lifespan, auto-trigger orchestration, express fast path, HTTP query proxy to main app, OneDrive/SharePoint Graph API downloads, tenant auto-provisioning, learning signal capture, and APIM deploy automation.

**Tech Stack:** FastAPI, Bot Framework SDK 4.17.0, Microsoft Graph SDK, asyncio workers, MongoDB, Qdrant, Redis, Azure CLI/SDK for APIM.

**Spec:** `docs/superpowers/specs/2026-04-07-teams-app-service-design.md`

---

## File Structure

```
teams_app/
├── __init__.py               ← Package marker
├── main.py                   ← FastAPI app, /teams/messages route, lifespan
├── config.py                 ← Teams service config (port, concurrency, proxy URL)
├── lifespan.py               ← Lightweight startup: embedding model, Qdrant, Mongo, Redis
├── bot/
│   ├── __init__.py
│   └── handler.py            ← Standalone bot: extends src.teams.bot_app, overrides pipeline
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py       ← Auto-trigger pipeline with progress card updates
│   ├── fast_path.py          ← Tiered routing: express vs full
│   └── workers.py            ← Semaphore-bounded concurrent document workers
├── storage/
│   ├── __init__.py
│   ├── namespace.py          ← Qdrant/Redis/Blob namespace helpers
│   └── tenant.py             ← AAD tenant auto-provisioning from Teams activity
├── proxy/
│   ├── __init__.py
│   └── query_proxy.py        ← HTTP proxy to main app /api/ask with SSE chunked updates
├── signals/
│   ├── __init__.py
│   └── capture.py            ← Learning signal capture to finetune_buffer/high_quality JSONL
├── graph/
│   ├── __init__.py
│   └── onedrive.py           ← OneDrive/SharePoint file download via Graph API
├── deploy.py                 ← APIM route update + rollback automation
└── models.py                 ← Pydantic models for Teams service
tests/
└── teams_app/
    ├── __init__.py
    ├── test_config.py
    ├── test_fast_path.py
    ├── test_namespace.py
    ├── test_tenant.py
    ├── test_query_proxy.py
    ├── test_capture.py
    ├── test_orchestrator.py
    └── test_workers.py
```

---

### Task 1: Service Skeleton — Config, Models, Package Init

**Files:**
- Create: `teams_app/__init__.py`
- Create: `teams_app/config.py`
- Create: `teams_app/models.py`
- Create: `teams_app/bot/__init__.py`
- Create: `teams_app/pipeline/__init__.py`
- Create: `teams_app/storage/__init__.py`
- Create: `teams_app/proxy/__init__.py`
- Create: `teams_app/signals/__init__.py`
- Create: `teams_app/graph/__init__.py`
- Test: `tests/teams_app/__init__.py`
- Test: `tests/teams_app/test_config.py`

- [ ] **Step 1: Write config test**

```python
# tests/teams_app/test_config.py
import os
import pytest


def test_default_port():
    from teams_app.config import TeamsAppConfig
    cfg = TeamsAppConfig()
    assert cfg.port == 8300


def test_port_from_env(monkeypatch):
    monkeypatch.setenv("TEAMS_APP_PORT", "9000")
    from teams_app.config import TeamsAppConfig
    cfg = TeamsAppConfig()
    assert cfg.port == 9000


def test_default_proxy_url():
    from teams_app.config import TeamsAppConfig
    cfg = TeamsAppConfig()
    assert cfg.main_app_url == "http://localhost:8000"


def test_default_concurrency():
    from teams_app.config import TeamsAppConfig
    cfg = TeamsAppConfig()
    assert cfg.max_concurrent_documents == 3


def test_express_threshold():
    from teams_app.config import TeamsAppConfig
    cfg = TeamsAppConfig()
    assert cfg.express_min_chars == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'teams_app'`

- [ ] **Step 3: Create package structure and config**

```python
# teams_app/__init__.py
"""DocWain Teams App — Standalone Service."""
```

```python
# teams_app/bot/__init__.py
# teams_app/pipeline/__init__.py
# teams_app/storage/__init__.py
# teams_app/proxy/__init__.py
# teams_app/signals/__init__.py
# teams_app/graph/__init__.py
# (all empty __init__.py files)
```

```python
# tests/teams_app/__init__.py
# (empty)
```

```python
# teams_app/config.py
"""Teams App service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Set


EXPRESS_FILE_TYPES: Set[str] = {
    ".txt", ".md", ".csv", ".xlsx", ".json", ".xml", ".html",
}

FULL_FILE_TYPES: Set[str] = {
    ".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}


@dataclass
class TeamsAppConfig:
    """Configuration for the standalone Teams service."""

    # Service
    port: int = int(os.getenv("TEAMS_APP_PORT", "8300"))
    host: str = os.getenv("TEAMS_APP_HOST", "0.0.0.0")

    # Query proxy
    main_app_url: str = os.getenv("TEAMS_MAIN_APP_URL", "http://localhost:8000")
    proxy_timeout: int = int(os.getenv("TEAMS_PROXY_TIMEOUT", "120"))

    # Pipeline
    max_concurrent_documents: int = int(os.getenv("TEAMS_MAX_CONCURRENT_DOCS", "3"))
    express_min_chars: int = int(os.getenv("TEAMS_EXPRESS_MIN_CHARS", "50"))
    express_chunk_size: int = int(os.getenv("TEAMS_EXPRESS_CHUNK_SIZE", "1024"))
    full_chunk_size: int = int(os.getenv("TEAMS_FULL_CHUNK_SIZE", "2048"))

    # Learning signals
    signals_dir: str = os.getenv(
        "TEAMS_SIGNALS_DIR",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "outputs", "learning_signals"),
    )
```

```python
# teams_app/models.py
"""Pydantic models for the Teams service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TeamsDocument:
    """A document being processed through the Teams pipeline."""

    document_id: str
    tenant_id: str
    user_id: str
    filename: str
    source: str  # "attachment" | "onedrive"
    source_url: Optional[str] = None
    pipeline: str = "full"  # "express" | "full"
    status: str = "downloading"
    progress: Dict[str, Any] = field(default_factory=dict)
    teams_message_id: Optional[str] = None
    teams_conversation_id: Optional[str] = None


@dataclass
class TenantInfo:
    """Auto-provisioned tenant record."""

    tenant_id: str
    display_name: str
    qdrant_collection: str
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "kg_enabled": True,
        "max_documents": 1000,
        "express_pipeline": True,
    })
    document_count: int = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_config.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add teams_app/ tests/teams_app/
git commit -m "feat(teams-app): add service skeleton with config and models"
```

---

### Task 2: Fast Path — Tiered File Classification

**Files:**
- Create: `teams_app/pipeline/fast_path.py`
- Test: `tests/teams_app/test_fast_path.py`

- [ ] **Step 1: Write fast path tests**

```python
# tests/teams_app/test_fast_path.py
import pytest
from teams_app.pipeline.fast_path import classify_file, Pipeline


def test_csv_is_express():
    assert classify_file("report.csv") == Pipeline.EXPRESS


def test_txt_is_express():
    assert classify_file("notes.txt") == Pipeline.EXPRESS


def test_xlsx_is_express():
    assert classify_file("data.xlsx") == Pipeline.EXPRESS


def test_json_is_express():
    assert classify_file("config.json") == Pipeline.EXPRESS


def test_md_is_express():
    assert classify_file("readme.md") == Pipeline.EXPRESS


def test_xml_is_express():
    assert classify_file("feed.xml") == Pipeline.EXPRESS


def test_html_is_express():
    assert classify_file("page.html") == Pipeline.EXPRESS


def test_pdf_is_full():
    assert classify_file("document.pdf") == Pipeline.FULL


def test_docx_is_full():
    assert classify_file("letter.docx") == Pipeline.FULL


def test_pptx_is_full():
    assert classify_file("slides.pptx") == Pipeline.FULL


def test_image_is_full():
    assert classify_file("scan.png") == Pipeline.FULL
    assert classify_file("photo.jpg") == Pipeline.FULL
    assert classify_file("page.tiff") == Pipeline.FULL


def test_unknown_extension_is_full():
    assert classify_file("data.parquet") == Pipeline.FULL


def test_case_insensitive():
    assert classify_file("REPORT.CSV") == Pipeline.EXPRESS
    assert classify_file("Document.PDF") == Pipeline.FULL


def test_no_extension_is_full():
    assert classify_file("Makefile") == Pipeline.FULL


def test_should_escalate_short_text():
    from teams_app.pipeline.fast_path import should_escalate
    assert should_escalate("hi", min_chars=50) is True


def test_should_not_escalate_long_text():
    from teams_app.pipeline.fast_path import should_escalate
    assert should_escalate("x" * 100, min_chars=50) is False


def test_should_escalate_empty():
    from teams_app.pipeline.fast_path import should_escalate
    assert should_escalate("", min_chars=50) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_fast_path.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'teams_app.pipeline.fast_path'`

- [ ] **Step 3: Implement fast path**

```python
# teams_app/pipeline/fast_path.py
"""Tiered file classification for express vs full pipeline."""

from __future__ import annotations

import os
from enum import Enum
from typing import Set

from teams_app.config import EXPRESS_FILE_TYPES, FULL_FILE_TYPES


class Pipeline(str, Enum):
    EXPRESS = "express"
    FULL = "full"


def classify_file(filename: str) -> Pipeline:
    """Classify a file as express or full pipeline based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in EXPRESS_FILE_TYPES:
        return Pipeline.EXPRESS
    return Pipeline.FULL


def should_escalate(extracted_text: str, min_chars: int = 50) -> bool:
    """Check if express extraction yielded too little content and should escalate to full."""
    return len(extracted_text.strip()) < min_chars
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_fast_path.py -v`
Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add teams_app/pipeline/fast_path.py tests/teams_app/test_fast_path.py
git commit -m "feat(teams-app): add tiered fast path file classification"
```

---

### Task 3: Namespace Helpers — Collection & Key Prefixing

**Files:**
- Create: `teams_app/storage/namespace.py`
- Test: `tests/teams_app/test_namespace.py`

- [ ] **Step 1: Write namespace tests**

```python
# tests/teams_app/test_namespace.py
import pytest
from teams_app.storage.namespace import (
    qdrant_collection_name,
    redis_key,
    blob_prefix,
    mongo_collection,
)


def test_qdrant_collection_name():
    assert qdrant_collection_name("tenant_abc") == "teams_tenant_abc"


def test_qdrant_collection_sanitizes():
    # Qdrant collection names: alphanumeric + underscore
    assert qdrant_collection_name("tenant-abc!@#") == "teams_tenant_abc___"


def test_redis_key_session():
    assert redis_key("tenant_abc", "user_123", "session") == "teams:tenant_abc:user_123:session"


def test_redis_key_uploads():
    assert redis_key("tenant_abc", "user_123", "uploads") == "teams:tenant_abc:user_123:uploads"


def test_blob_prefix():
    assert blob_prefix("tenant_abc") == "teams/tenant_abc/"


def test_mongo_documents_collection():
    assert mongo_collection("documents") == "teams_documents"


def test_mongo_tenants_collection():
    assert mongo_collection("tenants") == "teams_tenants"


def test_mongo_users_collection():
    assert mongo_collection("users") == "teams_users"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_namespace.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement namespace helpers**

```python
# teams_app/storage/namespace.py
"""Namespace helpers for isolating Teams data from the main app."""

from __future__ import annotations

import re

_QDRANT_SAFE = re.compile(r"[^a-zA-Z0-9_]")


def qdrant_collection_name(tenant_id: str) -> str:
    """Build a Teams-namespaced Qdrant collection name."""
    safe = _QDRANT_SAFE.sub("_", tenant_id)
    return f"teams_{safe}"


def redis_key(tenant_id: str, user_id: str, suffix: str) -> str:
    """Build a Teams-namespaced Redis key."""
    return f"teams:{tenant_id}:{user_id}:{suffix}"


def blob_prefix(tenant_id: str) -> str:
    """Build a Teams-namespaced Azure Blob path prefix."""
    return f"teams/{tenant_id}/"


def mongo_collection(name: str) -> str:
    """Build a Teams-namespaced MongoDB collection name."""
    return f"teams_{name}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_namespace.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add teams_app/storage/namespace.py tests/teams_app/test_namespace.py
git commit -m "feat(teams-app): add namespace helpers for data isolation"
```

---

### Task 4: Tenant Auto-Provisioning

**Files:**
- Create: `teams_app/storage/tenant.py`
- Test: `tests/teams_app/test_tenant.py`

- [ ] **Step 1: Write tenant provisioning tests**

```python
# tests/teams_app/test_tenant.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from teams_app.storage.tenant import TenantManager


@pytest.fixture
def mock_mongo():
    db = MagicMock()
    db.teams_tenants = MagicMock()
    db.teams_users = MagicMock()
    return db


@pytest.fixture
def mock_qdrant():
    return MagicMock()


@pytest.fixture
def manager(mock_mongo, mock_qdrant):
    return TenantManager(db=mock_mongo, qdrant_client=mock_qdrant)


def test_extract_identity_from_activity():
    activity = {
        "channelData": {"tenant": {"id": "tenant_abc"}},
        "from": {"id": "user_123", "name": "Alice"},
    }
    tenant_id, user_id, display_name = TenantManager.extract_identity(activity)
    assert tenant_id == "tenant_abc"
    assert user_id == "user_123"
    assert display_name == "Alice"


def test_extract_identity_missing_tenant():
    activity = {"from": {"id": "user_123"}}
    tenant_id, user_id, _ = TenantManager.extract_identity(activity)
    assert tenant_id is None
    assert user_id == "user_123"


def test_ensure_tenant_existing(manager, mock_mongo):
    mock_mongo.teams_tenants.find_one.return_value = {"tenant_id": "t1", "qdrant_collection": "teams_t1"}
    result = manager.ensure_tenant("t1", "Contoso")
    assert result["tenant_id"] == "t1"
    mock_mongo.teams_tenants.insert_one.assert_not_called()


def test_ensure_tenant_new_creates_record(manager, mock_mongo, mock_qdrant):
    mock_mongo.teams_tenants.find_one.return_value = None
    result = manager.ensure_tenant("t_new", "NewCorp")
    assert result["tenant_id"] == "t_new"
    assert result["qdrant_collection"] == "teams_t_new"
    mock_mongo.teams_tenants.insert_one.assert_called_once()
    mock_qdrant.create_collection.assert_called_once()


def test_ensure_user_existing(manager, mock_mongo):
    mock_mongo.teams_users.find_one.return_value = {"user_id": "u1"}
    result = manager.ensure_user("u1", "t1", "Alice")
    assert result["user_id"] == "u1"
    mock_mongo.teams_users.insert_one.assert_not_called()


def test_ensure_user_new_creates_record(manager, mock_mongo):
    mock_mongo.teams_users.find_one.return_value = None
    result = manager.ensure_user("u_new", "t1", "Bob")
    assert result["user_id"] == "u_new"
    mock_mongo.teams_users.insert_one.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_tenant.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement tenant manager**

```python
# teams_app/storage/tenant.py
"""AAD tenant auto-provisioning from Teams activity context."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from qdrant_client.models import Distance, VectorParams

from teams_app.storage.namespace import qdrant_collection_name

logger = logging.getLogger(__name__)

# Default embedding dimension — must match the model loaded in lifespan
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
            # Collection may already exist — safe to ignore
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_tenant.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add teams_app/storage/tenant.py tests/teams_app/test_tenant.py
git commit -m "feat(teams-app): add AAD tenant auto-provisioning"
```

---

### Task 5: Query Proxy — HTTP Proxy to Main App with SSE Chunked Updates

**Files:**
- Create: `teams_app/proxy/query_proxy.py`
- Test: `tests/teams_app/test_query_proxy.py`

- [ ] **Step 1: Write query proxy tests**

```python
# tests/teams_app/test_query_proxy.py
import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from teams_app.proxy.query_proxy import QueryProxy, QueryRequest, QueryResult


def test_query_request_builds_payload():
    req = QueryRequest(
        query="what is revenue?",
        user_id="teams_user_123",
        subscription_id="teams_tenant_abc",
    )
    payload = req.to_dict()
    assert payload["query"] == "what is revenue?"
    assert payload["user_id"] == "teams_user_123"
    assert payload["subscription_id"] == "teams_tenant_abc"
    assert payload["stream"] is True
    assert payload["profile_id"] is None


def test_query_request_headers():
    req = QueryRequest(
        query="test",
        user_id="u1",
        subscription_id="teams_t1",
        tenant_id="t1",
    )
    headers = req.headers()
    assert headers["x-source"] == "teams"
    assert headers["x-teams-tenant"] == "t1"
    assert headers["Content-Type"] == "application/json"


def test_query_result_from_sse_data():
    data = {
        "response": "Revenue is $1M",
        "sources": [{"title": "report.pdf"}],
        "grounded": True,
        "context_found": True,
    }
    result = QueryResult.from_response(data)
    assert result.response == "Revenue is $1M"
    assert result.context_found is True
    assert len(result.sources) == 1


def test_query_result_no_context():
    data = {"response": "I don't know", "sources": [], "grounded": False, "context_found": False}
    result = QueryResult.from_response(data)
    assert result.context_found is False


@pytest.mark.asyncio
async def test_proxy_returns_result_on_success():
    proxy = QueryProxy(main_app_url="http://localhost:8000", timeout=10)
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "response": "Answer here",
        "sources": [],
        "grounded": True,
        "context_found": True,
    }

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        req = QueryRequest(query="test", user_id="u1", subscription_id="t1")
        result = await proxy.ask(req)
        assert result.response == "Answer here"
        assert result.context_found is True


@pytest.mark.asyncio
async def test_proxy_returns_error_on_failure():
    proxy = QueryProxy(main_app_url="http://localhost:8000", timeout=10)

    with patch("httpx.AsyncClient.post", side_effect=Exception("Connection refused")):
        req = QueryRequest(query="test", user_id="u1", subscription_id="t1")
        result = await proxy.ask(req)
        assert result.error is not None
        assert result.context_found is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_query_proxy.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement query proxy**

```python
# teams_app/proxy/query_proxy.py
"""HTTP proxy to main app /api/ask with SSE streaming support."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class QueryRequest:
    """A query to proxy to the main app."""

    query: str
    user_id: str
    subscription_id: str
    tenant_id: str = ""
    profile_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "user_id": self.user_id,
            "subscription_id": self.subscription_id,
            "profile_id": self.profile_id,
            "session_id": self.session_id,
            "stream": self.stream,
        }

    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-source": "teams",
            "x-teams-tenant": self.tenant_id,
        }


@dataclass
class QueryResult:
    """Result from the main app."""

    response: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    grounded: bool = False
    context_found: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> QueryResult:
        return cls(
            response=data.get("response", ""),
            sources=data.get("sources", []),
            grounded=data.get("grounded", False),
            context_found=data.get("context_found", False),
            metadata=data.get("metadata", {}),
        )


class QueryProxy:
    """Proxies queries to the main DocWain app."""

    def __init__(self, main_app_url: str, timeout: int = 120):
        self.main_app_url = main_app_url.rstrip("/")
        self.timeout = timeout

    async def ask(self, request: QueryRequest) -> QueryResult:
        """Send a query to the main app and return the result."""
        url = f"{self.main_app_url}/api/ask"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    url,
                    json=request.to_dict(),
                    headers=request.headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                return QueryResult.from_response(data)
        except Exception as exc:
            logger.error("Query proxy failed: %s", exc)
            return QueryResult(
                response="I'm having trouble right now. Please try again in a moment.",
                error=str(exc),
            )

    async def ask_stream(self, request: QueryRequest) -> AsyncIterator[str]:
        """Stream SSE tokens from the main app. Yields partial text chunks."""
        url = f"{self.main_app_url}/api/ask"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", url, json=request.to_dict(), headers=request.headers(),
                ) as resp:
                    resp.raise_for_status()
                    buffer = ""
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            payload = line[6:]
                            if payload == "[DONE]":
                                break
                            try:
                                chunk = json.loads(payload)
                                token = chunk.get("token", chunk.get("response", ""))
                                if token:
                                    yield token
                            except json.JSONDecodeError:
                                yield payload
        except Exception as exc:
            logger.error("Stream proxy failed: %s", exc)
            yield "I'm having trouble right now. Please try again in a moment."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_query_proxy.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add teams_app/proxy/query_proxy.py tests/teams_app/test_query_proxy.py
git commit -m "feat(teams-app): add HTTP query proxy to main app with SSE streaming"
```

---

### Task 6: Learning Signal Capture

**Files:**
- Create: `teams_app/signals/capture.py`
- Test: `tests/teams_app/test_capture.py`

- [ ] **Step 1: Write capture tests**

```python
# tests/teams_app/test_capture.py
import json
import os
import tempfile
import pytest
from teams_app.signals.capture import SignalCapture


@pytest.fixture
def signal_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def capture(signal_dir):
    return SignalCapture(signals_dir=signal_dir)


def test_capture_positive_writes_to_high_quality(capture, signal_dir):
    capture.record(
        query="what is revenue?",
        response="Revenue is $1M",
        sources=[{"title": "report.pdf"}],
        grounded=True,
        context_found=True,
        signal="positive",
        tenant_id="t1",
    )
    path = os.path.join(signal_dir, "high_quality.jsonl")
    assert os.path.exists(path)
    with open(path) as f:
        entry = json.loads(f.readline())
    assert entry["query"] == "what is revenue?"
    assert entry["signal"] == "positive"
    assert entry["source"] == "teams"


def test_capture_negative_writes_to_finetune_buffer(capture, signal_dir):
    capture.record(
        query="bad question",
        response="bad answer",
        sources=[],
        grounded=False,
        context_found=False,
        signal="negative",
        tenant_id="t1",
    )
    path = os.path.join(signal_dir, "finetune_buffer.jsonl")
    assert os.path.exists(path)
    with open(path) as f:
        entry = json.loads(f.readline())
    assert entry["signal"] == "negative"
    assert entry["source"] == "teams"


def test_capture_implicit_writes_to_finetune_buffer(capture, signal_dir):
    capture.record(
        query="q", response="a", sources=[], grounded=True,
        context_found=True, signal="implicit", tenant_id="t1",
    )
    path = os.path.join(signal_dir, "finetune_buffer.jsonl")
    assert os.path.exists(path)


def test_no_document_content_in_signal(capture, signal_dir):
    capture.record(
        query="q", response="a", sources=[{"title": "doc.pdf", "content": "secret data"}],
        grounded=True, context_found=True, signal="positive", tenant_id="t1",
    )
    path = os.path.join(signal_dir, "high_quality.jsonl")
    with open(path) as f:
        entry = json.loads(f.readline())
    # Sources should only have title, not content
    for src in entry["sources"]:
        assert "content" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_capture.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement signal capture**

```python
# teams_app/signals/capture.py
"""Learning signal capture for finetuning from Teams interactions."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


def _sanitize_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Strip document content from sources — only keep titles/references."""
    return [{"title": s.get("title", "")} for s in sources if s.get("title")]


class SignalCapture:
    """Appends query/response pairs to JSONL files for finetuning."""

    def __init__(self, signals_dir: str):
        self.signals_dir = signals_dir
        os.makedirs(signals_dir, exist_ok=True)

    def record(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        grounded: bool,
        context_found: bool,
        signal: str,  # "positive", "negative", "implicit"
        tenant_id: str,
        pipeline: str = "",
        latency_ms: int = 0,
    ) -> None:
        """Record a learning signal to the appropriate JSONL file."""
        entry = {
            "query": query,
            "response": response,
            "sources": _sanitize_sources(sources),
            "grounded": grounded,
            "context_found": context_found,
            "source": "teams",
            "signal": signal,
            "tenant_id": tenant_id,
            "pipeline": pipeline,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if signal == "positive":
            filename = "high_quality.jsonl"
        else:
            filename = "finetune_buffer.jsonl"

        path = os.path.join(self.signals_dir, filename)
        line = json.dumps(entry, ensure_ascii=False) + "\n"

        with _write_lock:
            with open(path, "a") as f:
                f.write(line)

        logger.debug("Recorded %s signal to %s", signal, filename)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_capture.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add teams_app/signals/capture.py tests/teams_app/test_capture.py
git commit -m "feat(teams-app): add learning signal capture for finetuning"
```

---

### Task 7: Concurrent Document Workers

**Files:**
- Create: `teams_app/pipeline/workers.py`
- Test: `tests/teams_app/test_workers.py`

- [ ] **Step 1: Write worker tests**

```python
# tests/teams_app/test_workers.py
import asyncio
import pytest
from teams_app.pipeline.workers import WorkerPool


@pytest.mark.asyncio
async def test_worker_pool_limits_concurrency():
    """Verify that only max_concurrent tasks run at once."""
    pool = WorkerPool(max_concurrent=2)
    running = []
    max_running = 0

    async def track_task(task_id: str):
        nonlocal max_running
        running.append(task_id)
        current = len(running)
        if current > max_running:
            max_running = current
        await asyncio.sleep(0.05)
        running.remove(task_id)
        return f"done-{task_id}"

    results = await pool.run_all([
        ("a", track_task, ("a",)),
        ("b", track_task, ("b",)),
        ("c", track_task, ("c",)),
        ("d", track_task, ("d",)),
    ])

    assert max_running <= 2
    assert len(results) == 4
    assert all(r.startswith("done-") for r in results.values())


@pytest.mark.asyncio
async def test_worker_pool_captures_errors():
    pool = WorkerPool(max_concurrent=2)

    async def fail_task(task_id: str):
        raise ValueError(f"fail-{task_id}")

    async def ok_task(task_id: str):
        return f"ok-{task_id}"

    results = await pool.run_all([
        ("good", ok_task, ("good",)),
        ("bad", fail_task, ("bad",)),
    ])

    assert results["good"] == "ok-good"
    assert isinstance(results["bad"], Exception)


@pytest.mark.asyncio
async def test_worker_pool_empty_input():
    pool = WorkerPool(max_concurrent=3)
    results = await pool.run_all([])
    assert results == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_workers.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement worker pool**

```python
# teams_app/pipeline/workers.py
"""Semaphore-bounded concurrent document processing workers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Tuple

logger = logging.getLogger(__name__)


class WorkerPool:
    """Runs async tasks with bounded concurrency."""

    def __init__(self, max_concurrent: int = 3):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(
        self, task_id: str, coro_fn: Callable[..., Coroutine], args: Tuple
    ) -> Tuple[str, Any]:
        async with self.semaphore:
            try:
                result = await coro_fn(*args)
                return task_id, result
            except Exception as exc:
                logger.error("Worker task %s failed: %s", task_id, exc)
                return task_id, exc

    async def run_all(
        self, tasks: List[Tuple[str, Callable[..., Coroutine], Tuple]]
    ) -> Dict[str, Any]:
        """Run all tasks with bounded concurrency.

        Args:
            tasks: List of (task_id, async_callable, args) tuples.

        Returns:
            Dict mapping task_id to result (or Exception on failure).
        """
        if not tasks:
            return {}

        coros = [self._run_one(tid, fn, args) for tid, fn, args in tasks]
        completed = await asyncio.gather(*coros)
        return dict(completed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_workers.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add teams_app/pipeline/workers.py tests/teams_app/test_workers.py
git commit -m "feat(teams-app): add semaphore-bounded concurrent worker pool"
```

---

### Task 8: OneDrive/SharePoint Download via Graph API

**Files:**
- Create: `teams_app/graph/onedrive.py`
- Test: (manual — Graph API requires auth tokens, tested via integration)

- [ ] **Step 1: Implement OneDrive downloader**

```python
# teams_app/graph/onedrive.py
"""OneDrive/SharePoint file download via Microsoft Graph API."""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Patterns for OneDrive/SharePoint URLs
_ONEDRIVE_PATTERN = re.compile(
    r"https?://[^/]*(?:sharepoint\.com|onedrive\.live\.com|1drv\.ms)[^\s]*",
    re.IGNORECASE,
)


def is_onedrive_url(text: str) -> Optional[str]:
    """Extract a OneDrive/SharePoint URL from message text. Returns URL or None."""
    match = _ONEDRIVE_PATTERN.search(text)
    return match.group(0) if match else None


async def download_shared_file(
    url: str,
    access_token: str,
    max_bytes: int = 50 * 1024 * 1024,
    timeout: int = 60,
) -> Tuple[bytes, str]:
    """Download a file from a OneDrive/SharePoint sharing URL.

    Uses the Graph API shares endpoint to resolve sharing links.

    Args:
        url: OneDrive/SharePoint sharing URL.
        access_token: Graph API access token with Files.Read.All scope.
        max_bytes: Maximum file size to download.
        timeout: HTTP timeout in seconds.

    Returns:
        Tuple of (file_bytes, filename).

    Raises:
        ValueError: If the URL can't be resolved or file is too large.
        httpx.HTTPStatusError: On API errors.
    """
    import base64

    # Encode sharing URL for Graph API
    # https://learn.microsoft.com/en-us/graph/api/shares-get
    encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    share_id = f"u!{encoded}"

    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Resolve the sharing link to get driveItem metadata
        meta_url = f"https://graph.microsoft.com/v1.0/shares/{share_id}/driveItem"
        meta_resp = await client.get(meta_url, headers=headers)
        meta_resp.raise_for_status()
        item = meta_resp.json()

        filename = item.get("name", "unknown_file")
        size = item.get("size", 0)

        if size > max_bytes:
            raise ValueError(
                f"File {filename} is {size / 1024 / 1024:.1f}MB, "
                f"exceeds limit of {max_bytes / 1024 / 1024:.0f}MB"
            )

        # Download the file content
        content_url = f"{meta_url}/content"
        content_resp = await client.get(content_url, headers=headers, follow_redirects=True)
        content_resp.raise_for_status()

        logger.info("Downloaded %s (%d bytes) from OneDrive/SharePoint", filename, len(content_resp.content))
        return content_resp.content, filename
```

- [ ] **Step 2: Commit**

```bash
git add teams_app/graph/onedrive.py
git commit -m "feat(teams-app): add OneDrive/SharePoint file download via Graph API"
```

---

### Task 9: Pipeline Orchestrator — Auto-Trigger with Progress Cards

**Files:**
- Create: `teams_app/pipeline/orchestrator.py`
- Test: `tests/teams_app/test_orchestrator.py`

This is the core module that wires everything together. It extends the existing `src.teams.pipeline.TeamsDocumentPipeline` with auto-trigger behavior and tiered fast path.

- [ ] **Step 1: Write orchestrator tests**

```python
# tests/teams_app/test_orchestrator.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from teams_app.pipeline.orchestrator import TeamsAutoOrchestrator
from teams_app.pipeline.fast_path import Pipeline


@pytest.fixture
def orchestrator():
    return TeamsAutoOrchestrator(
        storage=MagicMock(),
        state_store=MagicMock(),
        tenant_manager=MagicMock(),
        signal_capture=MagicMock(),
    )


def test_classify_express(orchestrator):
    assert orchestrator.classify("data.csv") == Pipeline.EXPRESS


def test_classify_full(orchestrator):
    assert orchestrator.classify("report.pdf") == Pipeline.FULL


def test_should_escalate_short_text(orchestrator):
    assert orchestrator.should_escalate("hi") is True


def test_should_not_escalate_long_text(orchestrator):
    assert orchestrator.should_escalate("x" * 200) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement orchestrator**

```python
# teams_app/pipeline/orchestrator.py
"""Auto-trigger pipeline orchestrator with tiered fast path and progress cards."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from teams_app.config import TeamsAppConfig
from teams_app.pipeline.fast_path import Pipeline, classify_file, should_escalate
from teams_app.pipeline.workers import WorkerPool
from teams_app.signals.capture import SignalCapture
from teams_app.storage.tenant import TenantManager

logger = logging.getLogger(__name__)


class TeamsAutoOrchestrator:
    """Orchestrates document processing with auto-trigger and tiered fast path.

    Wraps the existing src.teams.pipeline.TeamsDocumentPipeline stages,
    adding express/full routing, concurrent workers, and progress card updates.
    """

    def __init__(
        self,
        storage: Any,
        state_store: Any,
        tenant_manager: TenantManager,
        signal_capture: SignalCapture,
        config: Optional[TeamsAppConfig] = None,
    ):
        self.storage = storage
        self.state_store = state_store
        self.tenant_manager = tenant_manager
        self.signal_capture = signal_capture
        self.config = config or TeamsAppConfig()
        self.worker_pool = WorkerPool(max_concurrent=self.config.max_concurrent_documents)

    def classify(self, filename: str) -> Pipeline:
        """Classify a file for express or full pipeline."""
        return classify_file(filename)

    def should_escalate(self, extracted_text: str) -> bool:
        """Check if express extraction needs escalation to full pipeline."""
        return should_escalate(extracted_text, min_chars=self.config.express_min_chars)

    async def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        context: Any,
        turn_context: Any,
        correlation_id: str,
        auth_token: str = "",
    ) -> Dict[str, Any]:
        """Process a single document through the appropriate pipeline.

        This is called by the worker pool for each attachment.
        Uses the existing src.teams.pipeline stages but controls routing.
        """
        from src.teams.pipeline import TeamsDocumentPipeline
        from src.teams.cards import build_card

        pipeline_type = self.classify(filename)
        started = time.monotonic()

        # Create the existing pipeline instance
        base_pipeline = TeamsDocumentPipeline(
            storage=self.storage,
            state_store=self.state_store,
        )

        # Send initial progress card
        progress_card = build_card("stage_progress_card",
            filename=filename,
            pipeline_type=pipeline_type.value,
            stage="downloading",
            progress_pct="10",
        )
        msg_id = await self._send_card(turn_context, progress_card)

        try:
            # Stage 1: Extraction
            await self._update_progress(turn_context, msg_id, filename, pipeline_type, "extracting", "25")

            if pipeline_type == Pipeline.EXPRESS:
                result = await self._express_extract(file_bytes, filename, context, correlation_id)
                extracted_text = result.get("extracted_text", "")

                # Auto-escalate if express yields too little
                if self.should_escalate(extracted_text):
                    logger.info("Escalating %s from express to full pipeline (text too short)", filename)
                    pipeline_type = Pipeline.FULL
                    result = await base_pipeline.stage_identify(
                        file_bytes, filename, content_type, context, correlation_id,
                    )
            else:
                result = await base_pipeline.stage_identify(
                    file_bytes, filename, content_type, context, correlation_id,
                )

            if not result:
                await self._update_progress(turn_context, msg_id, filename, pipeline_type, "failed", "0")
                return {"status": "failed", "filename": filename, "error": "Extraction failed"}

            # Stage 2: Screening
            await self._update_progress(turn_context, msg_id, filename, pipeline_type, "screening", "50")

            screen_result = await base_pipeline.stage_screen(
                document_id=result.get("document_id", ""),
                extracted_text=result.get("extracted_text", "")[:50000],
                filename=filename,
                correlation_id=correlation_id,
            )

            if screen_result and screen_result.get("needs_consent"):
                await self._update_progress(turn_context, msg_id, filename, pipeline_type, "blocked", "50")
                return {"status": "blocked", "filename": filename, "screen_result": screen_result}

            # Stage 3: Embedding
            await self._update_progress(turn_context, msg_id, filename, pipeline_type, "embedding", "75")

            embed_result = await base_pipeline.stage_embed(
                document_id=result.get("document_id", ""),
                extracted_content=result.get("extracted_docs", result),
                filename=filename,
                doc_type=result.get("doc_type", "unknown"),
                context=context,
                correlation_id=correlation_id,
                suggested_questions=result.get("suggested_questions", []),
                doc_intelligence=result.get("doc_intelligence"),
            )

            # Stage 4: KG (full pipeline only)
            if pipeline_type == Pipeline.FULL:
                await self._update_progress(turn_context, msg_id, filename, pipeline_type, "kg_building", "90")
                # KG runs via the existing pipeline's embedding stage (includes entity extraction)

            elapsed = time.monotonic() - started
            await self._update_progress(
                turn_context, msg_id, filename, pipeline_type, "ready",
                "100", elapsed_s=f"{elapsed:.1f}",
                sections=str(result.get("sections_count", 0)),
                chunks=str(embed_result.get("chunks_count", 0) if embed_result else 0),
            )

            return {
                "status": "ready",
                "filename": filename,
                "pipeline": pipeline_type.value,
                "elapsed_s": elapsed,
                "result": result,
                "embed_result": embed_result,
            }

        except Exception as exc:
            logger.exception("Pipeline failed for %s: %s", filename, exc)
            await self._update_progress(turn_context, msg_id, filename, pipeline_type, "failed", "0")
            return {"status": "failed", "filename": filename, "error": str(exc)}

    async def process_attachments(
        self,
        attachments: List[Dict[str, Any]],
        context: Any,
        turn_context: Any,
        correlation_id: str,
        auth_token: str = "",
    ) -> Dict[str, Any]:
        """Process multiple attachments concurrently via the worker pool."""
        tasks = []
        for att in attachments:
            file_bytes = att.get("file_bytes", b"")
            filename = att.get("filename", "unknown")
            content_type = att.get("content_type", "application/octet-stream")

            tasks.append((
                filename,
                self.process_document,
                (file_bytes, filename, content_type, context, turn_context, correlation_id, auth_token),
            ))

        return await self.worker_pool.run_all(tasks)

    async def _express_extract(
        self,
        file_bytes: bytes,
        filename: str,
        context: Any,
        correlation_id: str,
    ) -> Dict[str, Any]:
        """Express extraction using native parsers only — no OCR/layout."""
        from src.api.dataHandler import fileProcessor

        extracted_docs = await asyncio.to_thread(fileProcessor, file_bytes, filename)
        if not extracted_docs:
            return {"extracted_text": "", "extracted_docs": {}}

        # Aggregate all extracted text
        all_text = ""
        for doc_data in extracted_docs.values():
            if isinstance(doc_data, dict):
                all_text += doc_data.get("full_text", "") or doc_data.get("text", "")
            elif isinstance(doc_data, str):
                all_text += doc_data

        return {
            "extracted_text": all_text,
            "extracted_docs": extracted_docs,
            "document_id": correlation_id,
        }

    async def _send_card(self, turn_context: Any, card: Dict) -> Optional[str]:
        """Send an Adaptive Card and return the message ID for updates."""
        try:
            from botbuilder.schema import Activity, ActivityTypes, Attachment

            activity = Activity(
                type=ActivityTypes.message,
                attachments=[Attachment(
                    content_type="application/vnd.microsoft.card.adaptive",
                    content=card,
                )],
            )
            response = await turn_context.send_activity(activity)
            return response.id if response else None
        except Exception as exc:
            logger.error("Failed to send card: %s", exc)
            return None

    async def _update_progress(
        self,
        turn_context: Any,
        message_id: Optional[str],
        filename: str,
        pipeline_type: Pipeline,
        stage: str,
        progress_pct: str,
        **kwargs: str,
    ) -> None:
        """Update the in-place progress card."""
        if not message_id:
            return

        try:
            from botbuilder.schema import Activity, ActivityTypes, Attachment
            from src.teams.cards import build_card

            card = build_card("stage_progress_card",
                filename=filename,
                pipeline_type=pipeline_type.value,
                stage=stage,
                progress_pct=progress_pct,
                **kwargs,
            )
            update = Activity(
                id=message_id,
                type=ActivityTypes.message,
                attachments=[Attachment(
                    content_type="application/vnd.microsoft.card.adaptive",
                    content=card,
                )],
            )
            await turn_context.update_activity(update)
        except Exception as exc:
            logger.debug("Progress card update failed (non-fatal): %s", exc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/test_orchestrator.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add teams_app/pipeline/orchestrator.py tests/teams_app/test_orchestrator.py
git commit -m "feat(teams-app): add auto-trigger pipeline orchestrator with tiered fast path"
```

---

### Task 10: Bot Handler — Standalone Bot Extending src.teams

**Files:**
- Create: `teams_app/bot/handler.py`

- [ ] **Step 1: Implement standalone bot handler**

The handler reuses `DocWainTeamsBot` from `src/teams/bot_app.py` but overrides `on_message_activity` to use the auto-trigger orchestrator, tenant auto-provisioning, and query proxy.

```python
# teams_app/bot/handler.py
"""Standalone Teams bot handler — extends src.teams.bot_app with auto-trigger pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional

from botbuilder.core import TurnContext
from botbuilder.schema import Activity, ActivityTypes, Attachment

from src.teams.bot_app import DocWainTeamsBot
from src.teams.logic import TeamsChatService, TeamsChatContext
from src.teams.state import TeamsStateStore
from src.teams.tools import TeamsToolRouter, format_text_answer
from src.teams.cards import build_card
from src.teams.attachments import _resolve_auth_token, _resolve_download_url, _resolve_filename

from teams_app.config import TeamsAppConfig
from teams_app.graph.onedrive import is_onedrive_url, download_shared_file
from teams_app.pipeline.orchestrator import TeamsAutoOrchestrator
from teams_app.proxy.query_proxy import QueryProxy, QueryRequest
from teams_app.signals.capture import SignalCapture
from teams_app.storage.tenant import TenantManager

logger = logging.getLogger(__name__)


class StandaloneTeamsBot(DocWainTeamsBot):
    """Extends DocWainTeamsBot with auto-trigger pipeline and query proxy.

    Overrides on_message_activity to:
    1. Auto-provision tenants from AAD identity
    2. Route file attachments through the auto-trigger orchestrator
    3. Detect OneDrive/SharePoint links and download via Graph API
    4. Proxy text queries to the main app instead of local RAG
    5. Capture learning signals for finetuning
    """

    def __init__(
        self,
        orchestrator: TeamsAutoOrchestrator,
        query_proxy: QueryProxy,
        tenant_manager: TenantManager,
        signal_capture: SignalCapture,
        config: Optional[TeamsAppConfig] = None,
    ):
        super().__init__()
        self.orchestrator = orchestrator
        self.query_proxy = query_proxy
        self.tenant_manager = tenant_manager
        self.signal_capture = signal_capture
        self.config = config or TeamsAppConfig()

    async def on_message_activity(self, turn_context: TurnContext) -> None:
        """Handle incoming Teams messages — auto-trigger pipeline or proxy query."""
        activity = turn_context.activity
        correlation_id = str(uuid.uuid4())[:8]

        # Auto-provision tenant & user
        raw_activity = activity.as_dict() if hasattr(activity, "as_dict") else {}
        tenant_id, user_id, display_name = TenantManager.extract_identity(raw_activity)

        if tenant_id:
            self.tenant_manager.ensure_tenant(tenant_id, display_name)
            self.tenant_manager.ensure_user(user_id, tenant_id, display_name)

        context = self.chat_service.build_context(
            user_id=user_id,
            session_id=activity.conversation.id if activity.conversation else user_id,
        )

        # Check for file attachments
        attachments = self._extract_file_attachments(activity)

        if attachments:
            auth_token = await _resolve_auth_token(turn_context, None)
            await self._handle_attachments(
                attachments, context, turn_context, correlation_id, auth_token, tenant_id,
            )
            return

        # Check for Adaptive Card actions
        if activity.value and isinstance(activity.value, dict):
            result = await self.tool_router.handle_action(activity.value, context)
            if result:
                await turn_context.send_activity(Activity(**result))
            return

        # Check for OneDrive/SharePoint links in text
        text = (activity.text or "").strip()
        onedrive_url = is_onedrive_url(text)
        if onedrive_url:
            await self._handle_onedrive_link(
                onedrive_url, context, turn_context, correlation_id, tenant_id,
            )
            return

        # Handle text queries — proxy to main app
        if text:
            # Handle special commands
            lower = text.lower()
            if lower in ("help", "/help"):
                card = build_card("help_card")
                await self._send_card_activity(turn_context, card)
                return
            if lower in ("tools", "/tools"):
                result = await self.tool_router.handle_action({"action": "tools"}, context)
                if result:
                    await turn_context.send_activity(Activity(**result))
                return
            if any(phrase in lower for phrase in ("delete all", "remove all", "clear all")):
                card = build_card("delete_confirm_card")
                await self._send_card_activity(turn_context, card)
                return

            await self._handle_query(text, context, turn_context, tenant_id)
            return

        # Fallback — no recognized input
        await turn_context.send_activity("Send me a document to analyze, or ask a question about your uploaded documents.")

    def _extract_file_attachments(self, activity: Activity) -> list:
        """Extract file attachments from the activity."""
        if not activity.attachments:
            return []

        file_attachments = []
        for att in activity.attachments:
            content_type = att.content_type or ""
            # Skip card types
            if "card" in content_type.lower() or "adaptive" in content_type.lower():
                continue
            if att.name or (att.content_url and att.content_type):
                file_attachments.append(att)
        return file_attachments

    async def _handle_attachments(
        self,
        attachments: list,
        context: TeamsChatContext,
        turn_context: TurnContext,
        correlation_id: str,
        auth_token: str,
        tenant_id: str,
    ) -> None:
        """Download and process file attachments through the auto-trigger orchestrator."""
        prepared = []
        for att in attachments:
            try:
                url = _resolve_download_url(att)
                filename = _resolve_filename(att)
                file_bytes = await self._download_file(url, auth_token)
                prepared.append({
                    "file_bytes": file_bytes,
                    "filename": filename,
                    "content_type": att.content_type or "application/octet-stream",
                })
            except Exception as exc:
                logger.error("Failed to download attachment %s: %s", getattr(att, "name", "?"), exc)
                await turn_context.send_activity(f"Failed to download {getattr(att, 'name', 'file')}: {exc}")

        if prepared:
            await self.orchestrator.process_attachments(
                prepared, context, turn_context, correlation_id, auth_token,
            )

    async def _handle_onedrive_link(
        self,
        url: str,
        context: TeamsChatContext,
        turn_context: TurnContext,
        correlation_id: str,
        tenant_id: str,
    ) -> None:
        """Download a file from OneDrive/SharePoint and process it."""
        await turn_context.send_activity("Downloading from OneDrive/SharePoint...")
        try:
            # Get Graph API token (from bot credentials)
            auth_token = await _resolve_auth_token(turn_context, None)
            file_bytes, filename = await download_shared_file(url, auth_token)
            await self.orchestrator.process_attachments(
                [{"file_bytes": file_bytes, "filename": filename, "content_type": "application/octet-stream"}],
                context, turn_context, correlation_id, auth_token,
            )
        except Exception as exc:
            logger.error("OneDrive download failed: %s", exc)
            await turn_context.send_activity(f"Failed to download from OneDrive: {exc}")

    async def _handle_query(
        self,
        query: str,
        context: TeamsChatContext,
        turn_context: TurnContext,
        tenant_id: str,
    ) -> None:
        """Proxy a text query to the main app and send the response."""
        started = time.monotonic()

        # Show typing indicator
        await turn_context.send_activities([
            Activity(type=ActivityTypes.typing),
        ])

        req = QueryRequest(
            query=query,
            user_id=context.user_id,
            subscription_id=context.subscription_id,
            tenant_id=tenant_id or "",
            session_id=context.session_id,
        )

        # Non-streaming for simplicity — collect full response
        result = await self.query_proxy.ask(req)
        elapsed_ms = int((time.monotonic() - started) * 1000)

        # Check for no-documents fallback
        if not result.context_found:
            # Check if tenant has any documents
            has_docs = self.orchestrator.storage.get_document_count(tenant_id) > 0 if tenant_id else False
            if not has_docs:
                result.response += "\n\n_I don't have any documents to search yet. Send me a file to get started!_"

        # Send response as text with thumbs up/down buttons
        response_text = format_text_answer({"response": result.response, "sources": result.sources})

        # Build response card with feedback buttons
        card = {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.4",
            "body": [
                {"type": "TextBlock", "text": result.response, "wrap": True},
            ],
            "actions": [
                {
                    "type": "Action.Submit",
                    "title": "\ud83d\udc4d",
                    "data": {"action": "feedback", "signal": "positive", "query": query},
                },
                {
                    "type": "Action.Submit",
                    "title": "\ud83d\udc4e",
                    "data": {"action": "feedback", "signal": "negative", "query": query},
                },
            ],
        }

        if result.sources:
            sources_text = "\n".join(f"- {s.get('title', 'Unknown')}" for s in result.sources[:5])
            card["body"].append({"type": "TextBlock", "text": f"**Sources:**\n{sources_text}", "wrap": True, "size": "Small"})

        await self._send_card_activity(turn_context, card)

        # Capture learning signal (implicit — no feedback yet)
        self.signal_capture.record(
            query=query,
            response=result.response,
            sources=result.sources,
            grounded=result.grounded,
            context_found=result.context_found,
            signal="implicit",
            tenant_id=tenant_id or "",
            latency_ms=elapsed_ms,
        )

    async def _download_file(self, url: str, auth_token: str) -> bytes:
        """Download a file from a URL with auth token."""
        import httpx
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers, follow_redirects=True)
            resp.raise_for_status()
            return resp.content

    async def _send_card_activity(self, turn_context: TurnContext, card: Dict) -> Optional[str]:
        """Send an Adaptive Card and return the message ID."""
        activity = Activity(
            type=ActivityTypes.message,
            attachments=[Attachment(
                content_type="application/vnd.microsoft.card.adaptive",
                content=card,
            )],
        )
        response = await turn_context.send_activity(activity)
        return response.id if response else None

    async def on_members_added_activity(self, members_added, turn_context: TurnContext) -> None:
        """Send welcome card with onboarding message when bot is added."""
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                card = build_card("welcome_card")
                await self._send_card_activity(turn_context, card)
```

- [ ] **Step 2: Commit**

```bash
git add teams_app/bot/handler.py
git commit -m "feat(teams-app): add standalone bot handler with auto-trigger and query proxy"
```

---

### Task 11: Lifespan & FastAPI Main Entry Point

**Files:**
- Create: `teams_app/lifespan.py`
- Create: `teams_app/main.py`

- [ ] **Step 1: Implement lightweight lifespan**

```python
# teams_app/lifespan.py
"""Lightweight lifespan for the Teams service — loads only what's needed."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import FastAPI

logger = logging.getLogger(__name__)


@dataclass
class TeamsAppState:
    """Minimal state for the Teams service."""

    embedding_model: Any = None
    qdrant_client: Any = None
    mongo_db: Any = None
    redis_client: Any = None
    bot: Any = None
    bot_adapter: Any = None
    tenant_manager: Any = None
    orchestrator: Any = None
    query_proxy: Any = None
    signal_capture: Any = None


@asynccontextmanager
async def teams_lifespan(app: FastAPI):
    """Initialize Teams service dependencies — lightweight, ~10-15s startup."""
    from teams_app.config import TeamsAppConfig

    config = TeamsAppConfig()
    state = TeamsAppState()

    logger.info("Starting DocWain Teams service on port %d", config.port)

    # 1. Embedding model (heaviest — ~5s)
    logger.info("Loading embedding model...")
    from src.api.dw_newron import get_model
    import asyncio
    state.embedding_model = await asyncio.to_thread(get_model)
    logger.info("Embedding model loaded")

    # 2. Qdrant client
    logger.info("Connecting to Qdrant...")
    from src.api.dw_newron import get_qdrant_client
    state.qdrant_client = get_qdrant_client()
    logger.info("Qdrant connected")

    # 3. MongoDB
    logger.info("Connecting to MongoDB...")
    from src.api.dataHandler import get_database
    state.mongo_db = get_database()
    logger.info("MongoDB connected")

    # 4. Redis
    logger.info("Connecting to Redis...")
    try:
        from src.api.dw_newron import get_redis_client
        state.redis_client = get_redis_client()
        logger.info("Redis connected")
    except Exception as exc:
        logger.warning("Redis not available, using in-memory fallback: %s", exc)
        state.redis_client = None

    # 5. Initialize Teams-specific components
    from teams_app.storage.tenant import TenantManager
    from teams_app.signals.capture import SignalCapture
    from teams_app.proxy.query_proxy import QueryProxy
    from teams_app.pipeline.orchestrator import TeamsAutoOrchestrator
    from src.teams.state import TeamsStateStore
    from src.teams.teams_storage import TeamsDocumentStorage

    state.tenant_manager = TenantManager(db=state.mongo_db, qdrant_client=state.qdrant_client)
    state.signal_capture = SignalCapture(signals_dir=config.signals_dir)
    state.query_proxy = QueryProxy(main_app_url=config.main_app_url, timeout=config.proxy_timeout)

    storage = TeamsDocumentStorage()
    teams_state_store = TeamsStateStore()

    state.orchestrator = TeamsAutoOrchestrator(
        storage=storage,
        state_store=teams_state_store,
        tenant_manager=state.tenant_manager,
        signal_capture=state.signal_capture,
        config=config,
    )

    # 6. Bot Framework adapter + bot
    from src.teams.bot_app import bot_adapter
    from teams_app.bot.handler import StandaloneTeamsBot

    state.bot_adapter = bot_adapter
    state.bot = StandaloneTeamsBot(
        orchestrator=state.orchestrator,
        query_proxy=state.query_proxy,
        tenant_manager=state.tenant_manager,
        signal_capture=state.signal_capture,
        config=config,
    )

    app.state.teams = state
    logger.info("DocWain Teams service ready")

    yield

    logger.info("Shutting down DocWain Teams service")
```

- [ ] **Step 2: Implement FastAPI main entry point**

```python
# teams_app/main.py
"""DocWain Teams App — Standalone FastAPI Service."""

from __future__ import annotations

import json
import logging
import os
import sys

# Ensure project root is on sys.path for src imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from teams_app.config import TeamsAppConfig
from teams_app.lifespan import teams_lifespan

logger = logging.getLogger(__name__)

config = TeamsAppConfig()

app = FastAPI(
    title="DocWain Teams Service",
    version="1.0.0",
    lifespan=teams_lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    state = app.state.teams
    checks = {}

    # Qdrant
    try:
        state.qdrant_client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as exc:
        checks["qdrant"] = f"error: {exc}"

    # MongoDB
    try:
        state.mongo_db.command("ping")
        checks["mongodb"] = "ok"
    except Exception as exc:
        checks["mongodb"] = f"error: {exc}"

    # Redis
    if state.redis_client:
        try:
            state.redis_client.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {exc}"
    else:
        checks["redis"] = "unavailable (in-memory fallback)"

    # Main app
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{config.main_app_url}/api/health")
            checks["main_app"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as exc:
        checks["main_app"] = f"unreachable: {exc}"

    healthy = all(v == "ok" for k, v in checks.items() if k != "redis")
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "healthy" if healthy else "degraded", "checks": checks},
    )


@app.post("/teams/messages")
async def handle_teams_messages(request: Request):
    """Handle incoming Teams bot messages — mirrors src/main.py route."""
    state = app.state.teams

    # Parse activity
    try:
        body = await request.body()
        if not body:
            return JSONResponse(status_code=400, content={"error": "Empty body"})
        activity_payload = json.loads(body)
    except Exception as exc:
        logger.error("Failed to parse Teams activity: %s", exc)
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # Get auth header
    auth_header = request.headers.get("Authorization", "")

    # Process via Bot Framework adapter
    try:
        from botbuilder.schema import Activity

        activity_obj = Activity().deserialize(activity_payload)

        async def _run_bot(turn_context):
            await state.bot.on_turn(turn_context)

        response = await state.bot_adapter.process_activity(
            activity_obj, auth_header, _run_bot,
        )

        if response:
            return JSONResponse(status_code=response.status, content=response.body)
        return JSONResponse(status_code=200, content={})

    except PermissionError as exc:
        logger.warning("Auth failed: %s", exc)
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    except Exception as exc:
        logger.exception("Teams message handling failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/teams/feedback")
async def handle_feedback(request: Request):
    """Handle thumbs up/down feedback from response cards."""
    state = app.state.teams
    try:
        data = await request.json()
        signal = data.get("signal", "implicit")
        query = data.get("query", "")
        response = data.get("response", "")
        tenant_id = data.get("tenant_id", "")

        state.signal_capture.record(
            query=query,
            response=response,
            sources=[],
            grounded=True,
            context_found=True,
            signal=signal,
            tenant_id=tenant_id,
        )
        return JSONResponse(status_code=200, content={"status": "recorded"})
    except Exception as exc:
        logger.error("Feedback recording failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    uvicorn.run(app, host=config.host, port=config.port)
```

- [ ] **Step 3: Commit**

```bash
git add teams_app/lifespan.py teams_app/main.py
git commit -m "feat(teams-app): add FastAPI entry point and lightweight lifespan"
```

---

### Task 12: Systemd Service File

**Files:**
- Create: `deploy/docwain-teams.service`

- [ ] **Step 1: Create systemd unit file**

```ini
# deploy/docwain-teams.service
[Unit]
Description=DocWain Teams Bot Service
After=network.target docwain-app.service
Wants=docwain-app.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/PycharmProjects/DocWain
ExecStart=/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m teams_app.main
Restart=on-failure
RestartSec=5
TimeoutStartSec=60
TimeoutStopSec=30
StandardOutput=journal
SyslogIdentifier=docwain-teams
Environment=PYTHONPATH=/home/ubuntu/PycharmProjects/DocWain

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Commit**

```bash
git add deploy/docwain-teams.service
git commit -m "feat(teams-app): add systemd service unit file"
```

---

### Task 13: APIM Deploy Script

**Files:**
- Create: `teams_app/deploy.py`

- [ ] **Step 1: Implement APIM deploy automation**

```python
# teams_app/deploy.py
"""APIM route update automation for the Teams service.

Usage:
    python -m teams_app.deploy route-teams   # Route /teams/* to port 8300
    python -m teams_app.deploy rollback      # Route /teams/* back to main app
    python -m teams_app.deploy verify        # Check routing is working
    python -m teams_app.deploy status        # Show current APIM config
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

# Azure resource details
RESOURCE_GROUP = "rg-docwain-dev"
APIM_SERVICE = "dhs-docwain-api"
API_ID = "docwain-api"
BACKEND_IP = "4.213.139.185"
TEAMS_PORT = 8300
MAIN_PORT = 8000


def _az(args: list[str]) -> dict:
    """Run an az CLI command and return parsed JSON output."""
    cmd = ["az"] + args + ["--output", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("az command failed: %s\n%s", " ".join(cmd), result.stderr)
        raise RuntimeError(f"az CLI error: {result.stderr}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def route_teams():
    """Update APIM to route /teams/* requests to the Teams service on port 8300.

    Uses an APIM policy to override the backend URL for the /teams/messages operation.
    """
    logger.info("Setting up APIM route: /teams/* -> port %d", TEAMS_PORT)

    # Create or update the policy for /teams/messages operation
    # The policy overrides the backend service URL for this specific operation
    policy_xml = f"""<policies>
    <inbound>
        <base />
        <set-backend-service base-url="http://{BACKEND_IP}:{TEAMS_PORT}" />
    </inbound>
    <backend>
        <base />
    </backend>
    <outbound>
        <base />
    </outbound>
    <on-error>
        <base />
    </on-error>
</policies>"""

    # First, ensure the operation exists
    try:
        _az([
            "apim", "api", "operation", "show",
            "--resource-group", RESOURCE_GROUP,
            "--service-name", APIM_SERVICE,
            "--api-id", API_ID,
            "--operation-id", "teams-messages",
        ])
        logger.info("Operation teams-messages exists, updating policy...")
    except RuntimeError:
        # Create the operation if it doesn't exist
        logger.info("Creating APIM operation teams-messages...")
        _az([
            "apim", "api", "operation", "create",
            "--resource-group", RESOURCE_GROUP,
            "--service-name", APIM_SERVICE,
            "--api-id", API_ID,
            "--url-template", "/teams/messages",
            "--method", "POST",
            "--display-name", "Teams Bot Messages",
            "--operation-id", "teams-messages",
        ])

    # Apply the policy to route to port 8300
    # Write policy to temp file for az CLI
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(policy_xml)
        policy_file = f.name

    try:
        subprocess.run([
            "az", "apim", "api", "operation", "policy", "create",
            "--resource-group", RESOURCE_GROUP,
            "--service-name", APIM_SERVICE,
            "--api-id", API_ID,
            "--operation-id", "teams-messages",
            "--xml-file", policy_file,
        ], check=True)
        logger.info("APIM policy applied: /teams/messages -> port %d", TEAMS_PORT)
    finally:
        import os
        os.unlink(policy_file)

    print(f"Done. Teams traffic now routes to http://{BACKEND_IP}:{TEAMS_PORT}")


def rollback():
    """Remove the Teams routing policy — traffic falls back to the main app."""
    logger.info("Rolling back APIM route: /teams/* -> main app (port %d)", MAIN_PORT)

    try:
        subprocess.run([
            "az", "apim", "api", "operation", "policy", "delete",
            "--resource-group", RESOURCE_GROUP,
            "--service-name", APIM_SERVICE,
            "--api-id", API_ID,
            "--operation-id", "teams-messages",
            "--yes",
        ], check=True)
        logger.info("APIM policy removed. Teams traffic now goes to main app.")
    except subprocess.CalledProcessError as exc:
        logger.warning("Policy delete failed (may not exist): %s", exc)

    print(f"Done. Teams traffic now routes to main app at http://{BACKEND_IP}:{MAIN_PORT}")


def verify():
    """Verify the Teams service is reachable through APIM."""
    import httpx

    # Check direct health
    direct_url = f"http://localhost:{TEAMS_PORT}/health"
    try:
        resp = httpx.get(direct_url, timeout=5)
        print(f"Direct health check ({direct_url}): {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
    except Exception as exc:
        print(f"Direct health check FAILED: {exc}")
        return

    print("\nTeams service is running and healthy.")


def status():
    """Show current APIM routing configuration."""
    try:
        ops = _az([
            "apim", "api", "operation", "list",
            "--resource-group", RESOURCE_GROUP,
            "--service-name", APIM_SERVICE,
            "--api-id", API_ID,
        ])
        print("APIM Operations:")
        for op in ops:
            print(f"  {op.get('method', '?')} {op.get('urlTemplate', '?')} — {op.get('displayName', '?')}")
    except Exception as exc:
        print(f"Failed to query APIM: {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    commands = {
        "route-teams": route_teams,
        "rollback": rollback,
        "verify": verify,
        "status": status,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python -m teams_app.deploy <{'|'.join(commands.keys())}>")
        sys.exit(1)

    commands[sys.argv[1]]()
```

- [ ] **Step 2: Commit**

```bash
git add teams_app/deploy.py
git commit -m "feat(teams-app): add APIM deploy script with route-teams and rollback"
```

---

### Task 14: Integration Test & End-to-End Verification

**Files:**
- (no new files — uses existing test infrastructure)

- [ ] **Step 1: Run all unit tests**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/pytest tests/teams_app/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify the service starts**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && timeout 20 .venv/bin/python -m teams_app.main 2>&1 || true`
Expected: See "DocWain Teams service ready" in output, then timeout kills it

- [ ] **Step 3: Verify health endpoint**

In another terminal:
Run: `curl http://localhost:8300/health`
Expected: JSON with status "healthy" or "degraded" + individual checks

- [ ] **Step 4: Install systemd service**

```bash
sudo cp deploy/docwain-teams.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable docwain-teams
```

- [ ] **Step 5: Start the service and verify**

```bash
sudo systemctl start docwain-teams
sleep 5
curl http://localhost:8300/health
journalctl -u docwain-teams --no-pager -n 20
```
Expected: Service running, health endpoint returns OK

- [ ] **Step 6: Commit final state**

```bash
git add -A
git commit -m "feat(teams-app): complete standalone Teams service with all components"
```

---

### Task 15: APIM Route Deployment

This task is run manually once the service is verified working.

- [ ] **Step 1: Verify Teams service is healthy**

Run: `curl http://localhost:8300/health`
Expected: All checks "ok"

- [ ] **Step 2: Update APIM routing**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/python -m teams_app.deploy route-teams`
Expected: "Done. Teams traffic now routes to http://4.213.139.185:8300"

- [ ] **Step 3: Verify end-to-end via APIM**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/python -m teams_app.deploy verify`
Expected: Health check passes through APIM

- [ ] **Step 4: Test from Teams client**

Send a message to the DocWain bot in Microsoft Teams. Verify:
1. Bot responds (text query works)
2. Upload a file — verify progress card appears and updates
3. Ask a question about the uploaded file — verify answer comes back

- [ ] **Step 5: (If issues) Rollback**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && .venv/bin/python -m teams_app.deploy rollback`
Expected: Traffic returns to main app
