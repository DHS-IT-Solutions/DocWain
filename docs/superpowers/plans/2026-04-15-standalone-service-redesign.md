# Standalone Service Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully independent standalone FastAPI service (port 8400) that provides document extraction and intelligence endpoints by calling vLLM directly, with zero imports from the main app.

**Architecture:** A thin FastAPI service in `standalone/` at the project root. It accepts file uploads, converts them to text via lightweight local parsing, sends the text to the vLLM-served DocWain model for extraction/intelligence, and returns structured responses. API keys are admin-provisioned and stored in a separate MongoDB database. The service runs as its own systemd unit.

**Tech Stack:** FastAPI, httpx (async HTTP client to vLLM), PyMuPDF (PDF text extraction), python-docx, openpyxl, pymongo, pydantic v2, pytest

---

## File Map

```
standalone/
├── __init__.py
├── __main__.py              # uvicorn entry point
├── app.py                   # FastAPI app, middleware, router registration
├── config.py                # All env-var config
├── auth.py                  # API key hashing, validation dependency, admin auth
├── vllm_client.py           # Async httpx client to vLLM /v1/chat/completions
├── file_reader.py           # File → plain text conversion (PDF, DOCX, Excel, CSV, images, text)
├── output_formatter.py      # LLM text → structured format (JSON, CSV, sections, flatfile, tables)
├── schemas.py               # Pydantic request/response models
├── endpoints/
│   ├── __init__.py
│   ├── extract.py           # POST /api/v1/standalone/extract
│   ├── intelligence.py      # POST /api/v1/standalone/intelligence
│   └── keys.py              # POST/GET/DELETE /admin/keys
tests/standalone/
├── conftest.py              # Path setup, shared fixtures (app, client, mock vllm)
├── test_config.py           # Config loading tests
├── test_auth.py             # Key hashing, validation, admin auth tests
├── test_vllm_client.py      # vLLM client request/response tests
├── test_file_reader.py      # File conversion tests
├── test_output_formatter.py # Output formatting tests
├── test_schemas.py          # Pydantic model validation tests
├── test_extract.py          # Extract endpoint integration tests
├── test_intelligence.py     # Intelligence endpoint integration tests
└── test_keys.py             # Key management endpoint tests
systemd/
└── docwain-standalone.service  # New systemd unit
```

**Main app cleanup:**
- Remove: `src/api/standalone_api.py`, `standalone_processor.py`, `standalone_multi.py`, `standalone_auth.py`, `standalone_templates.py`, `standalone_output.py`, `standalone_webhook.py`, `standalone_schemas.py`
- Modify: `src/main.py` (remove lines 217-221), `src/api/config.py` (remove lines 208-216)

---

### Task 1: Project Scaffold & Config

**Files:**
- Create: `standalone/__init__.py`
- Create: `standalone/config.py`
- Create: `standalone/__main__.py`
- Create: `standalone/app.py`
- Create: `standalone/endpoints/__init__.py`
- Create: `tests/standalone/__init__.py`
- Create: `tests/standalone/conftest.py`
- Create: `tests/standalone/test_config.py`

- [ ] **Step 1: Write config tests**

Create `tests/standalone/test_config.py`:

```python
import os
import pytest


def test_config_defaults():
    """Config loads sensible defaults when no env vars set."""
    # Clear any standalone env vars that might be set
    env_keys = [
        "STANDALONE_PORT", "VLLM_BASE_URL", "VLLM_MODEL_NAME",
        "VLLM_TIMEOUT", "STANDALONE_MONGODB_URI", "STANDALONE_MONGODB_DB",
        "STANDALONE_ADMIN_SECRET", "STANDALONE_MAX_FILE_SIZE_MB", "STANDALONE_LOG_LEVEL",
    ]
    old_vals = {}
    for k in env_keys:
        old_vals[k] = os.environ.pop(k, None)

    try:
        # Re-import to pick up env changes
        import importlib
        import standalone.config as cfg_mod
        importlib.reload(cfg_mod)
        from standalone.config import Config

        assert Config.PORT == 8400
        assert Config.VLLM_BASE_URL == "http://localhost:8100/v1"
        assert Config.VLLM_MODEL_NAME == "docwain-fast"
        assert Config.VLLM_TIMEOUT == 120
        assert Config.MONGODB_URI == "mongodb://localhost:27017"
        assert Config.MONGODB_DB == "docwain_standalone"
        assert Config.MAX_FILE_SIZE_MB == 50
        assert Config.LOG_LEVEL == "INFO"
    finally:
        for k, v in old_vals.items():
            if v is not None:
                os.environ[k] = v


def test_config_respects_env_vars():
    """Config reads values from environment variables."""
    os.environ["STANDALONE_PORT"] = "9999"
    os.environ["VLLM_BASE_URL"] = "http://gpu-server:8200/v1"
    os.environ["VLLM_MODEL_NAME"] = "docwain-smart"

    try:
        import importlib
        import standalone.config as cfg_mod
        importlib.reload(cfg_mod)
        from standalone.config import Config

        assert Config.PORT == 9999
        assert Config.VLLM_BASE_URL == "http://gpu-server:8200/v1"
        assert Config.VLLM_MODEL_NAME == "docwain-smart"
    finally:
        del os.environ["STANDALONE_PORT"]
        del os.environ["VLLM_BASE_URL"]
        del os.environ["VLLM_MODEL_NAME"]


def test_config_admin_secret_from_env():
    """ADMIN_SECRET must come from env."""
    os.environ["STANDALONE_ADMIN_SECRET"] = "my-secret-123"
    try:
        import importlib
        import standalone.config as cfg_mod
        importlib.reload(cfg_mod)
        from standalone.config import Config

        assert Config.ADMIN_SECRET == "my-secret-123"
    finally:
        del os.environ["STANDALONE_ADMIN_SECRET"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_config.py -v`
Expected: ModuleNotFoundError for `standalone.config`

- [ ] **Step 3: Create project scaffold**

Create `standalone/__init__.py`:
```python
```

Create `standalone/endpoints/__init__.py`:
```python
```

Create `tests/standalone/__init__.py`:
```python
```

Create `tests/standalone/conftest.py`:
```python
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
```

Create `standalone/config.py`:
```python
import os


class Config:
    PORT = int(os.getenv("STANDALONE_PORT", "8400"))
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8100/v1")
    VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "docwain-fast")
    VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "120"))
    MONGODB_URI = os.getenv("STANDALONE_MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB = os.getenv("STANDALONE_MONGODB_DB", "docwain_standalone")
    ADMIN_SECRET = os.getenv("STANDALONE_ADMIN_SECRET", "")
    MAX_FILE_SIZE_MB = int(os.getenv("STANDALONE_MAX_FILE_SIZE_MB", "50"))
    LOG_LEVEL = os.getenv("STANDALONE_LOG_LEVEL", "INFO")
```

Create `standalone/app.py`:
```python
from fastapi import FastAPI

app = FastAPI(
    title="DocWain Standalone",
    description="Document extraction and intelligence API",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

Create `standalone/__main__.py`:
```python
import uvicorn
from standalone.config import Config

if __name__ == "__main__":
    uvicorn.run(
        "standalone.app:app",
        host="0.0.0.0",
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower(),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_config.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add standalone/ tests/standalone/
git commit -m "feat(standalone): project scaffold with config and health endpoint"
```

---

### Task 2: Pydantic Schemas

**Files:**
- Create: `standalone/schemas.py`
- Create: `tests/standalone/test_schemas.py`

- [ ] **Step 1: Write schema validation tests**

Create `tests/standalone/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError


def test_extract_request_valid_modes():
    from standalone.schemas import ExtractRequest

    for fmt in ("json", "csv", "sections", "flatfile", "tables"):
        req = ExtractRequest(output_format=fmt)
        assert req.output_format == fmt


def test_extract_request_invalid_mode():
    from standalone.schemas import ExtractRequest

    with pytest.raises(ValidationError):
        ExtractRequest(output_format="xml")


def test_extract_request_optional_prompt():
    from standalone.schemas import ExtractRequest

    req = ExtractRequest(output_format="json")
    assert req.prompt is None

    req2 = ExtractRequest(output_format="json", prompt="focus on tables")
    assert req2.prompt == "focus on tables"


def test_intelligence_request_defaults():
    from standalone.schemas import IntelligenceRequest

    req = IntelligenceRequest()
    assert req.analysis_type == "auto"
    assert req.prompt is None


def test_intelligence_request_valid_types():
    from standalone.schemas import IntelligenceRequest

    for t in ("summary", "key_facts", "risk_assessment", "recommendations", "auto"):
        req = IntelligenceRequest(analysis_type=t)
        assert req.analysis_type == t


def test_intelligence_request_invalid_type():
    from standalone.schemas import IntelligenceRequest

    with pytest.raises(ValidationError):
        IntelligenceRequest(analysis_type="magic")


def test_extract_response_structure():
    from standalone.schemas import ExtractResponse, ResponseMetadata

    resp = ExtractResponse(
        request_id="abc-123",
        document_type="invoice",
        output_format="json",
        content={"items": [{"name": "Widget", "price": 10}]},
        metadata=ResponseMetadata(pages=3, processing_time_ms=1200),
    )
    assert resp.request_id == "abc-123"
    assert resp.metadata.pages == 3


def test_intelligence_response_structure():
    from standalone.schemas import IntelligenceResponse, ResponseMetadata

    resp = IntelligenceResponse(
        request_id="def-456",
        document_type="contract",
        analysis_type="risk_assessment",
        insights={"summary": "High risk", "findings": [], "evidence": []},
        metadata=ResponseMetadata(pages=12, processing_time_ms=3400),
    )
    assert resp.analysis_type == "risk_assessment"
    assert resp.insights["summary"] == "High risk"


def test_key_create_request():
    from standalone.schemas import KeyCreateRequest

    req = KeyCreateRequest(name="production-key")
    assert req.name == "production-key"


def test_key_create_request_empty_name_rejected():
    from standalone.schemas import KeyCreateRequest

    with pytest.raises(ValidationError):
        KeyCreateRequest(name="")


def test_key_response_structure():
    from standalone.schemas import KeyCreateResponse

    resp = KeyCreateResponse(
        key_id="k-123",
        raw_key="dw_sa_abc123",
        key_prefix="dw_sa_abc",
        name="test",
        created_at="2026-04-15T00:00:00Z",
    )
    assert resp.raw_key == "dw_sa_abc123"


def test_key_list_item():
    from standalone.schemas import KeyListItem

    item = KeyListItem(
        key_id="k-123",
        key_prefix="dw_sa_abc",
        name="test",
        created_at="2026-04-15T00:00:00Z",
        total_requests=42,
    )
    assert item.total_requests == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_schemas.py -v`
Expected: ImportError for `standalone.schemas`

- [ ] **Step 3: Implement schemas**

Create `standalone/schemas.py`:

```python
from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    json = "json"
    csv = "csv"
    sections = "sections"
    flatfile = "flatfile"
    tables = "tables"


class AnalysisType(str, Enum):
    summary = "summary"
    key_facts = "key_facts"
    risk_assessment = "risk_assessment"
    recommendations = "recommendations"
    auto = "auto"


class ExtractRequest(BaseModel):
    output_format: OutputFormat
    prompt: Optional[str] = None


class IntelligenceRequest(BaseModel):
    analysis_type: AnalysisType = AnalysisType.auto
    prompt: Optional[str] = None


class ResponseMetadata(BaseModel):
    pages: int
    processing_time_ms: int


class ExtractResponse(BaseModel):
    request_id: str
    document_type: str
    output_format: str
    content: Any
    metadata: ResponseMetadata


class IntelligenceResponse(BaseModel):
    request_id: str
    document_type: str
    analysis_type: str
    insights: dict[str, Any]
    metadata: ResponseMetadata


class KeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)


class KeyCreateResponse(BaseModel):
    key_id: str
    raw_key: str
    key_prefix: str
    name: str
    created_at: str


class KeyListItem(BaseModel):
    key_id: str
    key_prefix: str
    name: str
    created_at: str
    total_requests: int


class ErrorResponse(BaseModel):
    error: str
    request_id: Optional[str] = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_schemas.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add standalone/schemas.py tests/standalone/test_schemas.py
git commit -m "feat(standalone): pydantic request/response schemas"
```

---

### Task 3: Authentication

**Files:**
- Create: `standalone/auth.py`
- Create: `tests/standalone/test_auth.py`

- [ ] **Step 1: Write auth tests**

Create `tests/standalone/test_auth.py`:

```python
import hashlib
import secrets
import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException


def test_hash_api_key_deterministic():
    from standalone.auth import hash_api_key

    key = "dw_sa_" + "a" * 48
    h1 = hash_api_key(key)
    h2 = hash_api_key(key)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest


def test_hash_api_key_different_keys_differ():
    from standalone.auth import hash_api_key

    h1 = hash_api_key("dw_sa_" + "a" * 48)
    h2 = hash_api_key("dw_sa_" + "b" * 48)
    assert h1 != h2


def test_generate_api_key_format():
    from standalone.auth import generate_api_key

    key = generate_api_key()
    assert key.startswith("dw_sa_")
    assert len(key) == 6 + 48  # prefix + 48 hex chars


@pytest.mark.asyncio
async def test_validate_api_key_success():
    from standalone.auth import validate_api_key

    raw_key = "dw_sa_" + "a" * 48
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    mock_collection = MagicMock()
    mock_collection.find_one = MagicMock(return_value={
        "key_hash": key_hash, "active": True, "name": "Test",
    })

    result = await validate_api_key(raw_key, mock_collection)
    assert result["name"] == "Test"
    mock_collection.find_one.assert_called_once_with({"key_hash": key_hash, "active": True})


@pytest.mark.asyncio
async def test_validate_api_key_not_found():
    from standalone.auth import validate_api_key

    mock_collection = MagicMock()
    mock_collection.find_one = MagicMock(return_value=None)

    result = await validate_api_key("dw_sa_bad", mock_collection)
    assert result is None


def test_verify_admin_secret_success():
    from standalone.auth import verify_admin_secret

    assert verify_admin_secret("correct-secret", "correct-secret") is True


def test_verify_admin_secret_failure():
    from standalone.auth import verify_admin_secret

    assert verify_admin_secret("wrong", "correct-secret") is False


def test_verify_admin_secret_empty():
    from standalone.auth import verify_admin_secret

    assert verify_admin_secret("anything", "") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_auth.py -v`
Expected: ImportError for `standalone.auth`

- [ ] **Step 3: Implement auth module**

Create `standalone/auth.py`:

```python
import hashlib
import secrets
from typing import Optional


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key() -> str:
    return "dw_sa_" + secrets.token_hex(24)


async def validate_api_key(raw_key: str, keys_collection) -> Optional[dict]:
    key_hash = hash_api_key(raw_key)
    return keys_collection.find_one({"key_hash": key_hash, "active": True})


def verify_admin_secret(provided: str, expected: str) -> bool:
    if not expected:
        return False
    return secrets.compare_digest(provided, expected)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_auth.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add standalone/auth.py tests/standalone/test_auth.py
git commit -m "feat(standalone): API key auth and admin secret verification"
```

---

### Task 4: vLLM Client

**Files:**
- Create: `standalone/vllm_client.py`
- Create: `tests/standalone/test_vllm_client.py`

- [ ] **Step 1: Write vLLM client tests**

Create `tests/standalone/test_vllm_client.py`:

```python
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_extract_builds_correct_request():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"tables": [["A", "B"], [1, 2]]}'}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    result = await client.extract("Invoice total: $500", output_format="json", prompt=None)

    assert result == '{"tables": [["A", "B"], [1, 2]]}'
    call_args = mock_client.post.call_args
    body = call_args.kwargs["json"]
    assert body["model"] == "docwain-fast"
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert "Invoice total: $500" in body["messages"][1]["content"]


@pytest.mark.asyncio
async def test_extract_includes_user_prompt():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "extracted content"}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    await client.extract("doc text", output_format="tables", prompt="focus on financial data")

    body = mock_client.post.call_args.kwargs["json"]
    user_content = body["messages"][1]["content"]
    assert "focus on financial data" in user_content
    assert "doc text" in user_content


@pytest.mark.asyncio
async def test_analyze_builds_correct_request():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Risk assessment: moderate risk found"}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    result = await client.analyze("contract text", analysis_type="risk_assessment", prompt=None)

    assert "Risk assessment" in result
    body = mock_client.post.call_args.kwargs["json"]
    assert body["messages"][0]["role"] == "system"
    assert "risk_assessment" in body["messages"][0]["content"].lower() or "risk" in body["messages"][0]["content"].lower()


@pytest.mark.asyncio
async def test_analyze_with_custom_prompt():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "analysis result"}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    await client.analyze("doc text", analysis_type="summary", prompt="focus on compliance")

    body = mock_client.post.call_args.kwargs["json"]
    user_content = body["messages"][1]["content"]
    assert "focus on compliance" in user_content


@pytest.mark.asyncio
async def test_strips_think_tags():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "<think>reasoning here</think>The actual answer"}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    result = await client.extract("text", output_format="json", prompt=None)
    assert "<think>" not in result
    assert "The actual answer" in result


@pytest.mark.asyncio
async def test_health_check():
    from standalone.vllm_client import VLLMClient

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    result = await client.health_check()
    assert result is True


@pytest.mark.asyncio
async def test_health_check_failure():
    from standalone.vllm_client import VLLMClient
    import httpx

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    client = VLLMClient(base_url="http://test:8100/v1", model="docwain-fast", timeout=30)
    client._client = mock_client

    result = await client.health_check()
    assert result is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_vllm_client.py -v`
Expected: ImportError for `standalone.vllm_client`

- [ ] **Step 3: Implement vLLM client**

Create `standalone/vllm_client.py`:

```python
import re
import httpx

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

EXTRACT_SYSTEM_PROMPT = """/no_think
You are DocWain, a document extraction engine. Extract the content from the provided document text and return it in the requested structured format.

Output format: {output_format}

Rules:
- For "json": Return a JSON object with keys like "document_type", "entities", "tables", "sections", "key_values" as appropriate.
- For "csv": Return CSV-formatted rows with headers.
- For "sections": Return the document broken into labeled sections with their content.
- For "flatfile": Return a flat key-value representation, one per line.
- For "tables": Return all tabular data as arrays of rows.
- Identify entities, classify the document type, and summarize sections alongside the structural output.
- Be thorough and precise. Include all content from the document."""

ANALYZE_SYSTEM_PROMPT = """/no_think
You are DocWain, a document intelligence engine. Analyze the provided document and produce high-level insights.

Analysis type: {analysis_type}

Rules:
- For "summary": Provide a comprehensive summary of the document's content, purpose, and key points.
- For "key_facts": Extract all important facts, figures, dates, names, and data points.
- For "risk_assessment": Identify risks, concerns, compliance issues, and potential problems.
- For "recommendations": Provide actionable recommendations based on the document content.
- For "auto": Choose the most appropriate analysis based on the document type and content.
- Ground every insight in the actual document content. Cite specific sections or quotes as evidence.
- Structure your response as JSON with keys: "summary", "findings" (array), "evidence" (array)."""


class VLLMClient:
    def __init__(self, base_url: str, model: str, timeout: int):
        self._base_url = base_url
        self._model = model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def extract(self, text: str, output_format: str, prompt: str | None) -> str:
        system = EXTRACT_SYSTEM_PROMPT.format(output_format=output_format)
        user_parts = []
        if prompt:
            user_parts.append(f"Additional instructions: {prompt}\n\n")
        user_parts.append(f"Document content:\n{text}")
        user_content = "".join(user_parts)

        return await self._call(system, user_content)

    async def analyze(self, text: str, analysis_type: str, prompt: str | None) -> str:
        system = ANALYZE_SYSTEM_PROMPT.format(analysis_type=analysis_type)
        user_parts = []
        if prompt:
            user_parts.append(f"Additional instructions: {prompt}\n\n")
        user_parts.append(f"Document content:\n{text}")
        user_content = "".join(user_parts)

        return await self._call(system, user_content)

    async def health_check(self) -> bool:
        try:
            # base_url is like http://localhost:8100/v1, health is at root
            health_url = self._base_url.rsplit("/v1", 1)[0] + "/health"
            resp = await self._client.get(health_url)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def _call(self, system: str, user: str) -> str:
        resp = await self._client.post(
            f"{self._base_url}/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _THINK_RE.sub("", content).strip()

    async def close(self):
        await self._client.aclose()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_vllm_client.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add standalone/vllm_client.py tests/standalone/test_vllm_client.py
git commit -m "feat(standalone): async vLLM client with extract and analyze methods"
```

---

### Task 5: File Reader

**Files:**
- Create: `standalone/file_reader.py`
- Create: `tests/standalone/test_file_reader.py`

- [ ] **Step 1: Write file reader tests**

Create `tests/standalone/test_file_reader.py`:

```python
import io
import pytest


def test_read_plain_text():
    from standalone.file_reader import read_file

    content = read_file("test.txt", b"Hello, this is a test document.")
    assert content == "Hello, this is a test document."


def test_read_json_file():
    from standalone.file_reader import read_file

    data = b'{"name": "DocWain", "version": 2}'
    content = read_file("data.json", data)
    assert "DocWain" in content


def test_read_csv_file():
    from standalone.file_reader import read_file

    csv_data = b"Name,Age,City\nAlice,30,London\nBob,25,Paris\n"
    content = read_file("people.csv", csv_data)
    assert "Alice" in content
    assert "Bob" in content
    assert "London" in content


def test_read_tsv_file():
    from standalone.file_reader import read_file

    tsv_data = b"Name\tAge\nAlice\t30\n"
    content = read_file("people.tsv", tsv_data)
    assert "Alice" in content
    assert "30" in content


def test_unsupported_format_raises():
    from standalone.file_reader import read_file, UnsupportedFileType

    with pytest.raises(UnsupportedFileType):
        read_file("archive.zip", b"PK\x03\x04 some zip data")


def test_detect_type_by_extension():
    from standalone.file_reader import detect_file_type

    assert detect_file_type("report.pdf", b"") == "pdf"
    assert detect_file_type("doc.docx", b"") == "docx"
    assert detect_file_type("sheet.xlsx", b"") == "xlsx"
    assert detect_file_type("data.csv", b"") == "csv"
    assert detect_file_type("notes.txt", b"") == "text"
    assert detect_file_type("photo.png", b"") == "image"
    assert detect_file_type("photo.jpg", b"") == "image"


def test_detect_type_pdf_by_magic_bytes():
    from standalone.file_reader import detect_file_type

    assert detect_file_type("unknown.bin", b"%PDF-1.4 content") == "pdf"


def test_read_file_returns_string():
    from standalone.file_reader import read_file

    result = read_file("test.txt", b"some content")
    assert isinstance(result, str)


def test_read_file_metadata():
    from standalone.file_reader import read_file_with_metadata

    content, meta = read_file_with_metadata("test.csv", b"a,b\n1,2\n3,4\n")
    assert isinstance(content, str)
    assert meta["file_type"] == "csv"
    assert meta["size_bytes"] == len(b"a,b\n1,2\n3,4\n")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_file_reader.py -v`
Expected: ImportError for `standalone.file_reader`

- [ ] **Step 3: Implement file reader**

Create `standalone/file_reader.py`:

```python
import csv
import io
from pathlib import Path


class UnsupportedFileType(Exception):
    pass


_EXT_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "text",
    ".text": "text",
    ".md": "text",
    ".json": "text",
    ".xml": "text",
    ".html": "text",
    ".htm": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".tif": "image",
    ".bmp": "image",
    ".webp": "image",
}

_SUPPORTED_TYPES = {"pdf", "docx", "xlsx", "csv", "tsv", "text", "image"}


def detect_file_type(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        return "docx"  # could be docx or xlsx, default docx
    return "unknown"


def read_file(filename: str, data: bytes) -> str:
    file_type = detect_file_type(filename, data)

    if file_type not in _SUPPORTED_TYPES:
        raise UnsupportedFileType(f"Unsupported file type: {filename}")

    if file_type == "text":
        return data.decode("utf-8", errors="replace")

    if file_type == "csv":
        return _read_csv(data)

    if file_type == "tsv":
        return _read_csv(data, delimiter="\t")

    if file_type == "pdf":
        return _read_pdf(data)

    if file_type == "docx":
        return _read_docx(data)

    if file_type == "xlsx":
        return _read_xlsx(data)

    if file_type == "image":
        return f"[Image file: {filename}]"

    raise UnsupportedFileType(f"Unsupported file type: {filename}")


def read_file_with_metadata(filename: str, data: bytes) -> tuple[str, dict]:
    file_type = detect_file_type(filename, data)
    content = read_file(filename, data)
    meta = {
        "file_type": file_type,
        "filename": filename,
        "size_bytes": len(data),
    }
    return content, meta


def _read_csv(data: bytes, delimiter: str = ",") -> str:
    text = data.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return ""
    lines = []
    for row in rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _read_pdf(data: bytes) -> str:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required for PDF processing. Install with: pip install PyMuPDF")


def _read_docx(data: bytes) -> str:
    try:
        from docx import Document

        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")


def _read_xlsx(data: bytes) -> str:
    try:
        from openpyxl import load_workbook

        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        sheets = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                rows.append(" | ".join(cells))
            if rows:
                sheets.append(f"Sheet: {sheet_name}\n" + "\n".join(rows))
        wb.close()
        return "\n\n".join(sheets)
    except ImportError:
        raise ImportError("openpyxl is required for Excel processing. Install with: pip install openpyxl")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_file_reader.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add standalone/file_reader.py tests/standalone/test_file_reader.py
git commit -m "feat(standalone): file reader with PDF, DOCX, Excel, CSV, text support"
```

---

### Task 6: Output Formatter

**Files:**
- Create: `standalone/output_formatter.py`
- Create: `tests/standalone/test_output_formatter.py`

- [ ] **Step 1: Write output formatter tests**

Create `tests/standalone/test_output_formatter.py`:

```python
import json
import pytest


def test_format_json_passthrough():
    from standalone.output_formatter import format_output

    raw = '{"document_type": "invoice", "total": 500}'
    result = format_output(raw, "json")
    assert isinstance(result, dict)
    assert result["document_type"] == "invoice"


def test_format_json_from_non_json_text():
    from standalone.output_formatter import format_output

    raw = "The document is an invoice with total $500."
    result = format_output(raw, "json")
    assert isinstance(result, dict)
    assert "content" in result


def test_format_csv():
    from standalone.output_formatter import format_output

    raw = '{"tables": [{"headers": ["Name", "Amount"], "rows": [["Alice", "100"], ["Bob", "200"]]}]}'
    result = format_output(raw, "csv")
    assert isinstance(result, str)
    assert "Name" in result
    assert "Alice" in result


def test_format_sections():
    from standalone.output_formatter import format_output

    raw = '{"sections": [{"title": "Introduction", "content": "This is the intro."}]}'
    result = format_output(raw, "sections")
    assert isinstance(result, dict)
    assert "sections" in result


def test_format_flatfile():
    from standalone.output_formatter import format_output

    raw = '{"document_type": "invoice", "vendor": "Acme", "total": "500"}'
    result = format_output(raw, "flatfile")
    assert isinstance(result, str)
    assert "document_type" in result
    assert "invoice" in result


def test_format_tables():
    from standalone.output_formatter import format_output

    raw = '{"tables": [{"headers": ["A", "B"], "rows": [["1", "2"]]}]}'
    result = format_output(raw, "tables")
    assert isinstance(result, (dict, list))


def test_format_json_extracts_from_markdown_fence():
    from standalone.output_formatter import format_output

    raw = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
    result = format_output(raw, "json")
    assert isinstance(result, dict)
    assert result["key"] == "value"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_output_formatter.py -v`
Expected: ImportError for `standalone.output_formatter`

- [ ] **Step 3: Implement output formatter**

Create `standalone/output_formatter.py`:

```python
import csv
import io
import json
import re
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def format_output(raw_llm_response: str, output_format: str) -> Any:
    parsed = _try_parse_json(raw_llm_response)

    if output_format == "json":
        if parsed is not None:
            return parsed
        return {"content": raw_llm_response}

    if output_format == "csv":
        return _to_csv(parsed, raw_llm_response)

    if output_format == "sections":
        if parsed is not None and "sections" in parsed:
            return parsed
        return {"sections": [{"title": "Document", "content": raw_llm_response}]}

    if output_format == "flatfile":
        return _to_flatfile(parsed, raw_llm_response)

    if output_format == "tables":
        if parsed is not None:
            if "tables" in parsed:
                return parsed["tables"]
            return parsed
        return {"content": raw_llm_response}

    return {"content": raw_llm_response}


def _try_parse_json(text: str) -> dict | list | None:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    match = _JSON_FENCE_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _to_csv(parsed: dict | None, raw: str) -> str:
    if parsed and "tables" in parsed:
        output = io.StringIO()
        writer = csv.writer(output)
        for table in parsed["tables"]:
            if "headers" in table:
                writer.writerow(table["headers"])
            if "rows" in table:
                for row in table["rows"]:
                    writer.writerow(row)
        return output.getvalue()
    return raw


def _to_flatfile(parsed: dict | None, raw: str) -> str:
    if parsed and isinstance(parsed, dict):
        lines = []
        for key, value in parsed.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            lines.append(f"{key}={value}")
        return "\n".join(lines)
    return raw
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_output_formatter.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add standalone/output_formatter.py tests/standalone/test_output_formatter.py
git commit -m "feat(standalone): output formatter for JSON, CSV, sections, flatfile, tables"
```

---

### Task 7: Extract Endpoint

**Files:**
- Create: `standalone/endpoints/extract.py`
- Create: `tests/standalone/test_extract.py`

- [ ] **Step 1: Write extract endpoint tests**

Create `tests/standalone/test_extract.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app():
    from standalone.endpoints.extract import router
    from standalone.app import get_db, get_vllm_client

    app = FastAPI()
    app.include_router(router)

    mock_db = MagicMock()
    mock_keys = MagicMock()
    mock_keys.find_one = MagicMock(return_value={
        "key_hash": "abc", "active": True, "name": "test",
    })
    mock_db.__getitem__ = MagicMock(return_value=mock_keys)
    mock_logs = MagicMock()
    mock_logs.insert_one = MagicMock()
    mock_db.request_logs = mock_logs

    mock_vllm = MagicMock()
    mock_vllm.extract = AsyncMock(return_value='{"document_type": "invoice", "total": 500}')

    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_vllm_client] = lambda: mock_vllm

    return app, mock_vllm


def test_extract_success():
    app, mock_vllm = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.extract.validate_api_key", return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"output_format": "json"},
            files={"file": ("test.txt", b"Invoice total: $500", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["output_format"] == "json"
    assert "request_id" in data
    assert "content" in data
    assert "metadata" in data


def test_extract_missing_file():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.extract.validate_api_key", return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"output_format": "json"},
        )

    assert response.status_code == 422


def test_extract_invalid_format():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.extract.validate_api_key", return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"output_format": "xml"},
            files={"file": ("test.txt", b"content", "text/plain")},
        )

    assert response.status_code == 422


def test_extract_no_api_key():
    app, _ = _make_app()
    client = TestClient(app)

    response = client.post(
        "/api/v1/standalone/extract",
        data={"output_format": "json"},
        files={"file": ("test.txt", b"content", "text/plain")},
    )

    assert response.status_code == 401


def test_extract_csv_format():
    app, mock_vllm = _make_app()
    mock_vllm.extract = AsyncMock(
        return_value='{"tables": [{"headers": ["Item", "Price"], "rows": [["Widget", "10"]]}]}'
    )
    client = TestClient(app)

    with patch("standalone.endpoints.extract.validate_api_key", return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"output_format": "csv"},
            files={"file": ("data.txt", b"Item: Widget, Price: 10", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["output_format"] == "csv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_extract.py -v`
Expected: ImportError for `standalone.endpoints.extract`

- [ ] **Step 3: Implement extract endpoint**

First, update `standalone/app.py` to add dependency providers:

```python
from fastapi import FastAPI
from pymongo import MongoClient
from standalone.config import Config
from standalone.vllm_client import VLLMClient

app = FastAPI(
    title="DocWain Standalone",
    description="Document extraction and intelligence API",
    version="1.0.0",
)

_mongo_client: MongoClient | None = None
_vllm_client: VLLMClient | None = None


def get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(Config.MONGODB_URI)
    return _mongo_client[Config.MONGODB_DB]


def get_vllm_client() -> VLLMClient:
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = VLLMClient(
            base_url=Config.VLLM_BASE_URL,
            model=Config.VLLM_MODEL_NAME,
            timeout=Config.VLLM_TIMEOUT,
        )
    return _vllm_client


@app.get("/health")
async def health():
    vllm = get_vllm_client()
    vllm_ok = await vllm.health_check()
    return {"status": "ok" if vllm_ok else "degraded", "vllm": vllm_ok}
```

Create `standalone/endpoints/extract.py`:

```python
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from standalone.app import get_db, get_vllm_client
from standalone.auth import hash_api_key, validate_api_key
from standalone.file_reader import UnsupportedFileType, read_file_with_metadata
from standalone.output_formatter import format_output
from standalone.schemas import ExtractResponse, OutputFormat, ResponseMetadata
from standalone.vllm_client import VLLMClient

router = APIRouter()


@router.post("/api/v1/standalone/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(...),
    output_format: OutputFormat = Form(...),
    prompt: str | None = Form(None),
    x_api_key: str | None = Header(None),
    db=Depends(get_db),
    vllm: VLLMClient = Depends(get_vllm_client),
):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-Api-Key header")

    key_doc = await validate_api_key(x_api_key, db.api_keys)
    if not key_doc:
        raise HTTPException(status_code=401, detail="Invalid API key")

    request_id = str(uuid.uuid4())
    start = time.time()

    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    try:
        text, meta = read_file_with_metadata(file.filename or "unknown", data)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=422, detail=str(e))

    raw_response = await vllm.extract(text, output_format.value, prompt)
    content = format_output(raw_response, output_format.value)

    elapsed_ms = int((time.time() - start) * 1000)

    # Log request (fire-and-forget)
    try:
        db.request_logs.insert_one({
            "request_id": request_id,
            "endpoint": "extract",
            "key_hash": hash_api_key(x_api_key),
            "filename": file.filename,
            "file_type": meta["file_type"],
            "output_format": output_format.value,
            "processing_time_ms": elapsed_ms,
            "timestamp": time.time(),
        })
    except Exception:
        pass  # Don't fail the request if logging fails

    return ExtractResponse(
        request_id=request_id,
        document_type=meta["file_type"],
        output_format=output_format.value,
        content=content,
        metadata=ResponseMetadata(
            pages=meta.get("pages", 1),
            processing_time_ms=elapsed_ms,
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_extract.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add standalone/app.py standalone/endpoints/extract.py tests/standalone/test_extract.py
git commit -m "feat(standalone): extract endpoint with auth, file reading, vLLM call"
```

---

### Task 8: Intelligence Endpoint

**Files:**
- Create: `standalone/endpoints/intelligence.py`
- Create: `tests/standalone/test_intelligence.py`

- [ ] **Step 1: Write intelligence endpoint tests**

Create `tests/standalone/test_intelligence.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app():
    from standalone.endpoints.intelligence import router
    from standalone.app import get_db, get_vllm_client

    app = FastAPI()
    app.include_router(router)

    mock_db = MagicMock()
    mock_keys = MagicMock()
    mock_keys.find_one = MagicMock(return_value={
        "key_hash": "abc", "active": True, "name": "test",
    })
    mock_db.__getitem__ = MagicMock(return_value=mock_keys)
    mock_logs = MagicMock()
    mock_logs.insert_one = MagicMock()
    mock_db.request_logs = mock_logs

    mock_vllm = MagicMock()
    mock_vllm.analyze = AsyncMock(return_value='{"summary": "This is an invoice.", "findings": ["Due date: April 30"], "evidence": ["Page 1: Payment terms"]}')

    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_vllm_client] = lambda: mock_vllm

    return app, mock_vllm


def test_intelligence_success():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.intelligence.validate_api_key", return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_test"},
            files={"file": ("contract.txt", b"This contract is between A and B...", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_type"] == "auto"
    assert "insights" in data
    assert "request_id" in data


def test_intelligence_with_analysis_type():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.intelligence.validate_api_key", return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"analysis_type": "risk_assessment"},
            files={"file": ("contract.txt", b"This contract includes a penalty clause...", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_type"] == "risk_assessment"


def test_intelligence_with_prompt():
    app, mock_vllm = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.intelligence.validate_api_key", return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"prompt": "focus on compliance risks"},
            files={"file": ("policy.txt", b"Company policy document...", "text/plain")},
        )

    assert response.status_code == 200
    call_args = mock_vllm.analyze.call_args
    assert "focus on compliance risks" in call_args.args[2] or "focus on compliance risks" in str(call_args)


def test_intelligence_no_api_key():
    app, _ = _make_app()
    client = TestClient(app)

    response = client.post(
        "/api/v1/standalone/intelligence",
        files={"file": ("test.txt", b"content", "text/plain")},
    )

    assert response.status_code == 401


def test_intelligence_invalid_analysis_type():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.intelligence.validate_api_key", return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"analysis_type": "magic"},
            files={"file": ("test.txt", b"content", "text/plain")},
        )

    assert response.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_intelligence.py -v`
Expected: ImportError for `standalone.endpoints.intelligence`

- [ ] **Step 3: Implement intelligence endpoint**

Create `standalone/endpoints/intelligence.py`:

```python
import json
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from standalone.app import get_db, get_vllm_client
from standalone.auth import hash_api_key, validate_api_key
from standalone.file_reader import UnsupportedFileType, read_file_with_metadata
from standalone.schemas import AnalysisType, IntelligenceResponse, ResponseMetadata
from standalone.vllm_client import VLLMClient

router = APIRouter()


@router.post("/api/v1/standalone/intelligence", response_model=IntelligenceResponse)
async def intelligence(
    file: UploadFile = File(...),
    analysis_type: AnalysisType = Form(AnalysisType.auto),
    prompt: str | None = Form(None),
    x_api_key: str | None = Header(None),
    db=Depends(get_db),
    vllm: VLLMClient = Depends(get_vllm_client),
):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-Api-Key header")

    key_doc = await validate_api_key(x_api_key, db.api_keys)
    if not key_doc:
        raise HTTPException(status_code=401, detail="Invalid API key")

    request_id = str(uuid.uuid4())
    start = time.time()

    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    try:
        text, meta = read_file_with_metadata(file.filename or "unknown", data)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=422, detail=str(e))

    raw_response = await vllm.analyze(text, analysis_type.value, prompt)

    # Parse insights from LLM response
    insights = _parse_insights(raw_response)

    elapsed_ms = int((time.time() - start) * 1000)

    # Log request
    try:
        db.request_logs.insert_one({
            "request_id": request_id,
            "endpoint": "intelligence",
            "key_hash": hash_api_key(x_api_key),
            "filename": file.filename,
            "file_type": meta["file_type"],
            "analysis_type": analysis_type.value,
            "processing_time_ms": elapsed_ms,
            "timestamp": time.time(),
        })
    except Exception:
        pass

    return IntelligenceResponse(
        request_id=request_id,
        document_type=meta["file_type"],
        analysis_type=analysis_type.value,
        insights=insights,
        metadata=ResponseMetadata(
            pages=meta.get("pages", 1),
            processing_time_ms=elapsed_ms,
        ),
    )


def _parse_insights(raw: str) -> dict:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON from markdown fence
    import re
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return {"summary": raw, "findings": [], "evidence": []}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_intelligence.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add standalone/endpoints/intelligence.py tests/standalone/test_intelligence.py
git commit -m "feat(standalone): intelligence endpoint with analysis types and custom prompts"
```

---

### Task 9: Key Management Endpoint

**Files:**
- Create: `standalone/endpoints/keys.py`
- Create: `tests/standalone/test_keys.py`

- [ ] **Step 1: Write key management tests**

Create `tests/standalone/test_keys.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app():
    from standalone.endpoints.keys import router
    from standalone.app import get_db

    app = FastAPI()
    app.include_router(router)

    mock_db = MagicMock()
    mock_keys = MagicMock()
    mock_keys.insert_one = MagicMock()
    mock_keys.find = MagicMock(return_value=[
        {
            "_id": "k-1",
            "key_prefix": "dw_sa_abc",
            "name": "production",
            "created_at": "2026-04-15T00:00:00Z",
            "total_requests": 100,
            "active": True,
        }
    ])
    mock_keys.update_one = MagicMock()
    mock_db.api_keys = mock_keys

    app.dependency_overrides[get_db] = lambda: mock_db

    return app, mock_keys


def test_create_key_success():
    app, mock_keys = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "test-secret"
        response = client.post(
            "/admin/keys",
            headers={"X-Admin-Secret": "test-secret"},
            json={"name": "my-new-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["raw_key"].startswith("dw_sa_")
    assert data["name"] == "my-new-key"
    assert "key_id" in data
    mock_keys.insert_one.assert_called_once()


def test_create_key_no_admin_secret():
    app, _ = _make_app()
    client = TestClient(app)

    response = client.post(
        "/admin/keys",
        json={"name": "my-key"},
    )

    assert response.status_code == 401


def test_create_key_wrong_admin_secret():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "correct-secret"
        response = client.post(
            "/admin/keys",
            headers={"X-Admin-Secret": "wrong-secret"},
            json={"name": "my-key"},
        )

    assert response.status_code == 401


def test_list_keys():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "test-secret"
        response = client.get(
            "/admin/keys",
            headers={"X-Admin-Secret": "test-secret"},
        )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == "production"
    # Raw key should never appear in list
    assert "raw_key" not in data[0]
    assert "key_hash" not in data[0]


def test_delete_key():
    app, mock_keys = _make_app()
    mock_keys.update_one = MagicMock(return_value=MagicMock(modified_count=1))
    client = TestClient(app)

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "test-secret"
        response = client.delete(
            "/admin/keys/k-123",
            headers={"X-Admin-Secret": "test-secret"},
        )

    assert response.status_code == 200
    mock_keys.update_one.assert_called_once()


def test_delete_key_not_found():
    app, mock_keys = _make_app()
    mock_keys.update_one = MagicMock(return_value=MagicMock(modified_count=0))
    client = TestClient(app)

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "test-secret"
        response = client.delete(
            "/admin/keys/nonexistent",
            headers={"X-Admin-Secret": "test-secret"},
        )

    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_keys.py -v`
Expected: ImportError for `standalone.endpoints.keys`

- [ ] **Step 3: Implement key management endpoint**

Create `standalone/endpoints/keys.py`:

```python
import time
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException

from standalone.app import get_db
from standalone.auth import generate_api_key, hash_api_key, verify_admin_secret
from standalone.config import Config
from standalone.schemas import KeyCreateRequest, KeyCreateResponse, KeyListItem

router = APIRouter()


def _require_admin(x_admin_secret: str | None = Header(None)):
    if not x_admin_secret:
        raise HTTPException(status_code=401, detail="Missing X-Admin-Secret header")
    if not verify_admin_secret(x_admin_secret, Config.ADMIN_SECRET):
        raise HTTPException(status_code=401, detail="Invalid admin secret")


@router.post("/admin/keys", status_code=201, response_model=KeyCreateResponse)
def create_key(
    body: KeyCreateRequest,
    db=Depends(get_db),
    _=Depends(_require_admin),
):
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    key_id = str(uuid.uuid4())
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    db.api_keys.insert_one({
        "_id": key_id,
        "key_hash": key_hash,
        "key_prefix": raw_key[:12],
        "name": body.name,
        "active": True,
        "total_requests": 0,
        "created_at": created_at,
    })

    return KeyCreateResponse(
        key_id=key_id,
        raw_key=raw_key,
        key_prefix=raw_key[:12],
        name=body.name,
        created_at=created_at,
    )


@router.get("/admin/keys", response_model=list[KeyListItem])
def list_keys(
    db=Depends(get_db),
    _=Depends(_require_admin),
):
    keys = db.api_keys.find({"active": True})
    return [
        KeyListItem(
            key_id=str(k["_id"]),
            key_prefix=k["key_prefix"],
            name=k["name"],
            created_at=k["created_at"],
            total_requests=k.get("total_requests", 0),
        )
        for k in keys
    ]


@router.delete("/admin/keys/{key_id}")
def delete_key(
    key_id: str,
    db=Depends(get_db),
    _=Depends(_require_admin),
):
    result = db.api_keys.update_one(
        {"_id": key_id, "active": True},
        {"$set": {"active": False}},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"status": "revoked", "key_id": key_id}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/test_keys.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add standalone/endpoints/keys.py tests/standalone/test_keys.py
git commit -m "feat(standalone): admin key management endpoints (create, list, revoke)"
```

---

### Task 10: Wire Up App & Router Registration

**Files:**
- Modify: `standalone/app.py`
- Modify: `standalone/__main__.py`

- [ ] **Step 1: Update app.py to register all routers**

Update `standalone/app.py` to its final form:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pymongo import MongoClient

from standalone.config import Config
from standalone.vllm_client import VLLMClient

_mongo_client: MongoClient | None = None
_vllm_client: VLLMClient | None = None


def get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(Config.MONGODB_URI)
    return _mongo_client[Config.MONGODB_DB]


def get_vllm_client() -> VLLMClient:
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = VLLMClient(
            base_url=Config.VLLM_BASE_URL,
            model=Config.VLLM_MODEL_NAME,
            timeout=Config.VLLM_TIMEOUT,
        )
    return _vllm_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if _vllm_client:
        await _vllm_client.close()
    if _mongo_client:
        _mongo_client.close()


app = FastAPI(
    title="DocWain Standalone",
    description="Document extraction and intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
from standalone.endpoints.extract import router as extract_router
from standalone.endpoints.intelligence import router as intelligence_router
from standalone.endpoints.keys import router as keys_router

app.include_router(extract_router)
app.include_router(intelligence_router)
app.include_router(keys_router)


@app.get("/health")
async def health():
    vllm = get_vllm_client()
    vllm_ok = await vllm.health_check()
    return {"status": "ok" if vllm_ok else "degraded", "vllm": vllm_ok}
```

- [ ] **Step 2: Run all standalone tests**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/ -v`
Expected: All tests PASS (config: 3, schemas: 12, auth: 7, vllm: 7, file_reader: 10, output_formatter: 7, extract: 5, intelligence: 5, keys: 6 = ~62 tests)

- [ ] **Step 3: Commit**

```bash
git add standalone/app.py standalone/__main__.py
git commit -m "feat(standalone): wire up all routers and lifespan management"
```

---

### Task 11: Systemd Service

**Files:**
- Create: `systemd/docwain-standalone.service`

- [ ] **Step 1: Create systemd unit file**

Create `systemd/docwain-standalone.service`:

```ini
[Unit]
Description=DocWain Standalone API
After=network.target mongod.service
Wants=mongod.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/PycharmProjects/DocWain
Environment=STANDALONE_PORT=8400
Environment=VLLM_BASE_URL=http://localhost:8100/v1
Environment=VLLM_MODEL_NAME=docwain-fast
Environment=STANDALONE_MONGODB_URI=mongodb://localhost:27017
Environment=STANDALONE_MONGODB_DB=docwain_standalone
EnvironmentFile=-/home/ubuntu/PycharmProjects/DocWain/.env.standalone
ExecStart=/home/ubuntu/PycharmProjects/DocWain/venv/bin/python -m standalone
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Verify the unit file is syntactically valid**

Run: `systemd-analyze verify systemd/docwain-standalone.service 2>&1 || true`
Expected: No critical errors (warnings about missing user/paths at analysis time are fine)

- [ ] **Step 3: Commit**

```bash
git add systemd/docwain-standalone.service
git commit -m "feat(standalone): systemd service unit for port 8400"
```

---

### Task 12: Remove Old Standalone from Main App

**Files:**
- Delete: `src/api/standalone_api.py`
- Delete: `src/api/standalone_processor.py`
- Delete: `src/api/standalone_multi.py`
- Delete: `src/api/standalone_auth.py`
- Delete: `src/api/standalone_templates.py`
- Delete: `src/api/standalone_output.py`
- Delete: `src/api/standalone_webhook.py`
- Delete: `src/api/standalone_schemas.py`
- Modify: `src/main.py:217-221`
- Modify: `src/api/config.py:208-216`

- [ ] **Step 1: Remove standalone router from main.py**

In `src/main.py`, delete the try-except block at lines 217-221:

```python
# DELETE these lines:
try:
    from src.api.standalone_api import standalone_router
    api_router.include_router(standalone_router, tags=["Standalone API"])
except ImportError:
    pass
```

- [ ] **Step 2: Remove Config.Standalone from config.py**

In `src/api/config.py`, delete the Standalone class at lines 208-216:

```python
# DELETE these lines:
    class Standalone:
        ENABLED = os.getenv("DOCWAIN_STANDALONE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        TEMP_COLLECTION_TTL = int(os.getenv("STANDALONE_TEMP_TTL", "3600"))
        MAX_BATCH_FILES = int(os.getenv("STANDALONE_MAX_BATCH", "10"))
        MAX_FILE_SIZE_MB = int(os.getenv("STANDALONE_MAX_FILE_MB", "50"))
        WEBHOOK_MAX_WORKERS = int(os.getenv("STANDALONE_WEBHOOK_WORKERS", "4"))
        WEBHOOK_MAX_RETRIES = int(os.getenv("STANDALONE_WEBHOOK_RETRIES", "3"))
        API_KEYS_COLLECTION = os.getenv("STANDALONE_KEYS_COLLECTION", "api_keys")
        REQUESTS_COLLECTION = os.getenv("STANDALONE_REQUESTS_COLLECTION", "standalone_requests")
```

- [ ] **Step 3: Delete old standalone files**

Run:
```bash
git rm src/api/standalone_api.py src/api/standalone_processor.py src/api/standalone_multi.py src/api/standalone_auth.py src/api/standalone_templates.py src/api/standalone_output.py src/api/standalone_webhook.py src/api/standalone_schemas.py
```

- [ ] **Step 4: Verify main app still starts**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -c "from src.main import app; print('Main app imports OK')"`
Expected: "Main app imports OK" (no import errors)

- [ ] **Step 5: Commit**

```bash
git add -u src/api/ src/main.py
git commit -m "refactor: remove old standalone router from main app"
```

---

### Task 13: Integration Smoke Test

**Files:**
- Create: `tests/standalone/test_integration.py`

- [ ] **Step 1: Write end-to-end integration test**

Create `tests/standalone/test_integration.py`:

```python
"""
Smoke tests that verify the full standalone app works end-to-end
with mocked vLLM responses.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def full_app():
    """Create the full standalone app with mocked dependencies."""
    # Mock MongoDB
    mock_db = MagicMock()
    mock_keys = MagicMock()
    mock_keys.find_one = MagicMock(return_value={
        "key_hash": "abc", "active": True, "name": "test",
    })
    mock_keys.insert_one = MagicMock()
    mock_keys.find = MagicMock(return_value=[])
    mock_keys.update_one = MagicMock(return_value=MagicMock(modified_count=1))
    mock_db.api_keys = mock_keys

    mock_logs = MagicMock()
    mock_logs.insert_one = MagicMock()
    mock_db.request_logs = mock_logs
    mock_db.__getitem__ = MagicMock(return_value=mock_keys)

    # Mock vLLM
    mock_vllm = MagicMock()
    mock_vllm.extract = AsyncMock(return_value='{"document_type": "report", "sections": [{"title": "Summary", "content": "Test content"}]}')
    mock_vllm.analyze = AsyncMock(return_value='{"summary": "Test report summary", "findings": ["Finding 1"], "evidence": ["Page 1"]}')
    mock_vllm.health_check = AsyncMock(return_value=True)

    from standalone.app import app, get_db, get_vllm_client
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_vllm_client] = lambda: mock_vllm

    yield TestClient(app), mock_vllm, mock_db

    app.dependency_overrides.clear()


def test_health_check(full_app):
    client, _, _ = full_app
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_full_extract_flow(full_app):
    client, _, _ = full_app

    with patch("standalone.endpoints.extract.validate_api_key", return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_testkey123"},
            data={"output_format": "json"},
            files={"file": ("report.txt", b"Quarterly report with revenue data...", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["output_format"] == "json"
    assert data["content"]["document_type"] == "report"


def test_full_intelligence_flow(full_app):
    client, _, _ = full_app

    with patch("standalone.endpoints.intelligence.validate_api_key", return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_testkey123"},
            data={"analysis_type": "summary"},
            files={"file": ("report.txt", b"Quarterly report content...", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_type"] == "summary"
    assert "summary" in data["insights"]


def test_full_key_lifecycle(full_app):
    client, _, mock_db = full_app

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "admin-secret"

        # Create key
        response = client.post(
            "/admin/keys",
            headers={"X-Admin-Secret": "admin-secret"},
            json={"name": "new-key"},
        )
        assert response.status_code == 201
        key_data = response.json()
        assert key_data["raw_key"].startswith("dw_sa_")

        # List keys
        response = client.get(
            "/admin/keys",
            headers={"X-Admin-Secret": "admin-secret"},
        )
        assert response.status_code == 200

        # Delete key
        response = client.delete(
            f"/admin/keys/{key_data['key_id']}",
            headers={"X-Admin-Secret": "admin-secret"},
        )
        assert response.status_code == 200


def test_extract_then_intelligence_same_file(full_app):
    """Verify both endpoints can process the same file independently."""
    client, _, _ = full_app
    file_content = b"Contract between Company A and Company B for $1M services."

    with patch("standalone.endpoints.extract.validate_api_key", return_value={"name": "test"}):
        extract_resp = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_testkey123"},
            data={"output_format": "json"},
            files={"file": ("contract.txt", file_content, "text/plain")},
        )

    with patch("standalone.endpoints.intelligence.validate_api_key", return_value={"name": "test"}):
        intel_resp = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_testkey123"},
            data={"analysis_type": "risk_assessment"},
            files={"file": ("contract.txt", file_content, "text/plain")},
        )

    assert extract_resp.status_code == 200
    assert intel_resp.status_code == 200
    # Both should have different request_ids
    assert extract_resp.json()["request_id"] != intel_resp.json()["request_id"]
```

- [ ] **Step 2: Run full test suite**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/standalone/ -v`
Expected: All tests PASS (~67 tests total)

- [ ] **Step 3: Commit**

```bash
git add tests/standalone/test_integration.py
git commit -m "test(standalone): integration smoke tests for full service"
```
