# vLLM Production Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire vLLM as the primary serving backend for DocWain with dual-instance (14B fast + 27B smart), systemd lifecycle, GPU scheduler, and Ollama Cloud fallback.

**Architecture:** systemd manages vLLM processes. VLLMManager refactored to client-only (no subprocess). App lifecycle initializes VLLMManager and passes it through the query pipeline. GPU scheduler daemon handles training coexistence.

**Tech Stack:** vLLM 0.19.0, systemd, FastAPI, Python 3.12, Ollama (fallback)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/serving/config.py` | Modify | Add Ollama Cloud config, GPU mode file path, env-driven URLs |
| `src/serving/vllm_manager.py` | Rewrite | Client-only: health check, query routing, fallback chain, GPU mode awareness |
| `src/serving/__init__.py` | Modify | Update exports |
| `src/api/config.py` | Modify | Add VLLM_FAST_URL, VLLM_SMART_URL settings |
| `src/api/rag_state.py` | Modify | Add vllm_manager field to AppState |
| `src/api/app_lifespan.py` | Modify | Initialize VLLMManager at startup, store in AppState |
| `src/query/pipeline.py` | Modify | Ensure vllm_manager flows from AppState to clients dict |
| `systemd/docwain-vllm-fast.service` | Create | systemd unit for 14B vLLM instance |
| `systemd/docwain-vllm-smart.service` | Create | systemd unit for 27B vLLM instance |
| `systemd/docwain-gpu-scheduler.service` | Create | systemd unit for GPU scheduler daemon |
| `scripts/gpu_scheduler.py` | Create | GPU scheduler: idle detection, training orchestration, hot-swap |
| `tests/test_vllm_manager.py` | Create | Unit tests for refactored VLLMManager |
| `tests/test_gpu_scheduler.py` | Create | Unit tests for GPU scheduler logic |

---

### Task 1: Update serving config with new settings

**Files:**
- Modify: `src/serving/config.py`
- Modify: `src/api/config.py`

- [ ] **Step 1: Add new config entries to `src/api/config.py`**

Add inside the `Config.VLLM` inner class (after line 350):

```python
class VLLM:
    ENABLED = os.getenv("VLLM_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8001/v1/chat/completions")
    MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-14B-AWQ")
    API_KEY = _secret("VLLM_API_KEY", "")
    TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "30"))
    # New: dual-instance URLs
    FAST_URL = os.getenv("VLLM_FAST_URL", "http://localhost:8100")
    SMART_URL = os.getenv("VLLM_SMART_URL", "http://localhost:8200")
    FAST_MODEL = os.getenv("VLLM_FAST_MODEL", "docwain-fast")
    SMART_MODEL = os.getenv("VLLM_SMART_MODEL", "docwain-smart")
    GPU_MODE_FILE = os.getenv("DOCWAIN_GPU_MODE_FILE", "/tmp/docwain-gpu-mode.json")
```

- [ ] **Step 2: Update `src/serving/config.py` with Ollama Cloud fallback config and env-driven URLs**

Replace the `OllamaFallbackConfig` and add `OllamaCloudConfig`:

```python
@dataclass(frozen=True)
class OllamaFallbackConfig:
    """Local Ollama instance — emergency fallback."""
    host: str = "http://localhost:11434"
    model: str = "qwen3:14b"
    timeout_s: float = 300.0


@dataclass(frozen=True)
class OllamaCloudConfig:
    """Ollama Cloud — primary fallback during training."""
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:397b")
    api_key: str = os.getenv("OLLAMA_API_KEY", "")
    timeout_s: float = 300.0


OLLAMA_FALLBACK_CONFIG = OllamaFallbackConfig()
OLLAMA_CLOUD_CONFIG = OllamaCloudConfig()

GPU_MODE_FILE = os.getenv("DOCWAIN_GPU_MODE_FILE", "/tmp/docwain-gpu-mode.json")
```

- [ ] **Step 3: Commit**

```bash
git add src/api/config.py src/serving/config.py
git commit -m "feat(serving): add dual-instance vLLM and Ollama Cloud config"
```

---

### Task 2: Refactor VLLMManager to client-only

**Files:**
- Rewrite: `src/serving/vllm_manager.py`
- Test: `tests/test_vllm_manager.py`

- [ ] **Step 1: Write failing tests for the client-only VLLMManager**

Create `tests/test_vllm_manager.py`:

```python
"""Tests for VLLMManager client-only refactor."""

import json
import os
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import patch

import pytest


class FakeVLLMHandler(BaseHTTPRequestHandler):
    """Minimal fake vLLM server for testing."""

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if "/v1/chat/completions" in self.path:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            response = {
                "choices": [{
                    "message": {"content": f"Response from {body.get('model', 'unknown')}"}
                }]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # suppress test output


@pytest.fixture()
def fake_vllm_server():
    server = HTTPServer(("127.0.0.1", 0), FakeVLLMHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.fixture()
def gpu_mode_file(tmp_path):
    path = tmp_path / "gpu-mode.json"
    path.write_text(json.dumps({"mode": "serving"}))
    return str(path)


class TestVLLMManagerHealthCheck:
    def test_healthy_instance(self, fake_vllm_server):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager(
            fast_url=f"http://127.0.0.1:{fake_vllm_server}",
            smart_url=f"http://127.0.0.1:{fake_vllm_server}",
        )
        assert mgr.health_check("fast") is True
        assert mgr.health_check("smart") is True

    def test_unhealthy_instance(self):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager(
            fast_url="http://127.0.0.1:19999",
            smart_url="http://127.0.0.1:19998",
        )
        assert mgr.health_check("fast") is False
        assert mgr.health_check("smart") is False

    def test_unknown_instance(self, fake_vllm_server):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager(
            fast_url=f"http://127.0.0.1:{fake_vllm_server}",
            smart_url=f"http://127.0.0.1:{fake_vllm_server}",
        )
        assert mgr.health_check("nonexistent") is False


class TestVLLMManagerQuery:
    def test_query_fast_routes_to_vllm(self, fake_vllm_server, gpu_mode_file):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager(
            fast_url=f"http://127.0.0.1:{fake_vllm_server}",
            smart_url=f"http://127.0.0.1:{fake_vllm_server}",
            gpu_mode_file=gpu_mode_file,
        )
        result = mgr.query_fast("Hello")
        assert "Response from" in result

    def test_query_smart_routes_to_vllm(self, fake_vllm_server, gpu_mode_file):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager(
            fast_url=f"http://127.0.0.1:{fake_vllm_server}",
            smart_url=f"http://127.0.0.1:{fake_vllm_server}",
            gpu_mode_file=gpu_mode_file,
        )
        result = mgr.query_smart("Hello")
        assert "Response from" in result


class TestVLLMManagerFallback:
    def test_falls_back_to_ollama_cloud_when_vllm_down(self, gpu_mode_file):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager(
            fast_url="http://127.0.0.1:19999",
            smart_url="http://127.0.0.1:19998",
            gpu_mode_file=gpu_mode_file,
        )
        with patch.object(mgr, "_query_ollama_cloud", return_value="cloud response") as mock:
            result = mgr.query_fast("Hello")
            mock.assert_called_once()
            assert result == "cloud response"

    def test_training_mode_skips_vllm(self, fake_vllm_server, tmp_path):
        from src.serving.vllm_manager import VLLMManager

        mode_file = tmp_path / "gpu-mode.json"
        mode_file.write_text(json.dumps({"mode": "training"}))

        mgr = VLLMManager(
            fast_url=f"http://127.0.0.1:{fake_vllm_server}",
            smart_url=f"http://127.0.0.1:{fake_vllm_server}",
            gpu_mode_file=str(mode_file),
        )
        with patch.object(mgr, "_query_ollama_cloud", return_value="cloud fallback") as mock:
            result = mgr.query_fast("Hello")
            mock.assert_called_once()
            assert result == "cloud fallback"


class TestVLLMManagerGPUMode:
    def test_get_gpu_mode_serving(self, gpu_mode_file):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager(gpu_mode_file=gpu_mode_file)
        assert mgr.get_gpu_mode() == "serving"

    def test_get_gpu_mode_training(self, tmp_path):
        from src.serving.vllm_manager import VLLMManager

        mode_file = tmp_path / "gpu-mode.json"
        mode_file.write_text(json.dumps({"mode": "training"}))
        mgr = VLLMManager(gpu_mode_file=str(mode_file))
        assert mgr.get_gpu_mode() == "training"

    def test_get_gpu_mode_missing_file(self, tmp_path):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager(gpu_mode_file=str(tmp_path / "nonexistent.json"))
        assert mgr.get_gpu_mode() == "serving"  # default to serving

    def test_get_active_backends(self, fake_vllm_server):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager(
            fast_url=f"http://127.0.0.1:{fake_vllm_server}",
            smart_url=f"http://127.0.0.1:{fake_vllm_server}",
        )
        backends = mgr.get_active_backends()
        assert backends["fast"] is True
        assert backends["smart"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_vllm_manager.py -v
```

Expected: FAIL — `VLLMManager.__init__` doesn't accept `fast_url`, `smart_url`, `gpu_mode_file`.

- [ ] **Step 3: Rewrite VLLMManager as client-only**

Replace `src/serving/vllm_manager.py` entirely:

```python
"""Client-only vLLM manager — routes queries to vLLM instances managed by systemd.

No subprocess management. Health checks via HTTP. Falls back to Ollama Cloud
during training, Ollama local as emergency.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from src.serving.config import (
    GPU_MODE_FILE,
    OLLAMA_CLOUD_CONFIG,
    OLLAMA_FALLBACK_CONFIG,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class VLLMManager:
    """Client-only manager for dual vLLM instances with tiered fallback.

    Fallback chain (serving mode):
        vLLM instance → Ollama Cloud (397b) → Ollama local (14b)

    Fallback chain (training mode):
        Ollama Cloud (397b) → Ollama local (14b)
    """

    def __init__(
        self,
        fast_url: str = "http://localhost:8100",
        smart_url: str = "http://localhost:8200",
        fast_model: str = "docwain-fast",
        smart_model: str = "docwain-smart",
        gpu_mode_file: str = GPU_MODE_FILE,
    ) -> None:
        self._instances: Dict[str, Dict[str, str]] = {
            "fast": {"url": fast_url.rstrip("/"), "model": fast_model},
            "smart": {"url": smart_url.rstrip("/"), "model": smart_model},
        }
        self._gpu_mode_file = gpu_mode_file

    # -- Status ----------------------------------------------------------------

    def health_check(self, name: str) -> bool:
        """Return True if the named vLLM instance responds on /health."""
        instance = self._instances.get(name)
        if instance is None:
            return False
        url = f"{instance['url']}/health"
        try:
            req = urllib_request.Request(url, method="GET")
            with urllib_request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (HTTPError, URLError, OSError, TimeoutError):
            return False

    def get_gpu_mode(self) -> str:
        """Read GPU mode from state file. Returns 'serving' or 'training'."""
        try:
            with open(self._gpu_mode_file, "r") as f:
                data = json.load(f)
            return data.get("mode", "serving")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return "serving"

    def get_active_backends(self) -> Dict[str, bool]:
        """Return health status of all backends."""
        return {
            "fast": self.health_check("fast"),
            "smart": self.health_check("smart"),
            "gpu_mode": self.get_gpu_mode(),
        }

    # -- Query interface -------------------------------------------------------

    def query(
        self,
        instance_name: str,
        prompt: str,
        system_prompt: str = "",
        guided_json: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Query a vLLM instance with tiered fallback.

        In training mode, skips vLLM entirely and goes to Ollama Cloud.
        """
        # Training mode — skip vLLM, use Ollama Cloud
        if self.get_gpu_mode() == "training":
            logger.info("GPU in training mode — routing to Ollama Cloud")
            return self._query_ollama_cloud(prompt, system_prompt, max_tokens, temperature)

        # Try vLLM
        instance = self._instances.get(instance_name)
        if instance and self.health_check(instance_name):
            result = self._query_vllm(instance, prompt, system_prompt, guided_json, max_tokens, temperature)
            if result is not None:
                return result

        # Fallback: Ollama Cloud
        logger.info("vLLM '%s' unavailable — falling back to Ollama Cloud", instance_name)
        cloud_result = self._query_ollama_cloud(prompt, system_prompt, max_tokens, temperature)
        if cloud_result:
            return cloud_result

        # Emergency fallback: Ollama local
        logger.warning("Ollama Cloud unavailable — emergency fallback to Ollama local")
        return self._query_ollama_local(prompt, system_prompt, max_tokens, temperature)

    def query_fast(
        self,
        prompt: str,
        system_prompt: str = "",
        guided_json: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Query the fast (14B) instance."""
        return self.query("fast", prompt, system_prompt, guided_json, max_tokens, temperature)

    def query_smart(
        self,
        prompt: str,
        system_prompt: str = "",
        guided_json: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Query the smart (27B) instance."""
        return self.query("smart", prompt, system_prompt, guided_json, max_tokens, temperature)

    # -- Private: vLLM ---------------------------------------------------------

    def _query_vllm(
        self,
        instance: Dict[str, str],
        prompt: str,
        system_prompt: str,
        guided_json: Optional[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        """POST to vLLM OpenAI-compatible endpoint. Returns None on failure."""
        url = f"{instance['url']}/v1/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": instance["model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if guided_json is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "guided_output",
                    "strict": True,
                    "schema": guided_json,
                },
            }

        try:
            data = json.dumps(body).encode()
            req = urllib_request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode())
            return result["choices"][0]["message"]["content"].strip()
        except (HTTPError, URLError, OSError, KeyError, json.JSONDecodeError) as exc:
            logger.error("vLLM query failed (%s): %s", instance["url"], exc)
            return None

    # -- Private: Ollama Cloud -------------------------------------------------

    def _query_ollama_cloud(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Query Ollama Cloud (397b) — primary fallback."""
        cfg = OLLAMA_CLOUD_CONFIG
        url = f"{cfg.host}/api/generate"

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        body = {
            "model": cfg.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        if cfg.api_key:
            body["api_key"] = cfg.api_key

        try:
            data = json.dumps(body).encode()
            req = urllib_request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=cfg.timeout_s) as resp:
                result = json.loads(resp.read().decode())
            text = result.get("response", "")
            return _THINK_RE.sub("", text).strip()
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            logger.error("Ollama Cloud query failed: %s", exc)
            return ""

    # -- Private: Ollama local (emergency) -------------------------------------

    def _query_ollama_local(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Query local Ollama — emergency fallback."""
        cfg = OLLAMA_FALLBACK_CONFIG
        url = f"{cfg.host}/api/generate"

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        body = {
            "model": cfg.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }

        try:
            data = json.dumps(body).encode()
            req = urllib_request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=cfg.timeout_s) as resp:
                result = json.loads(resp.read().decode())
            text = result.get("response", "")
            return _THINK_RE.sub("", text).strip()
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            logger.error("Ollama local fallback failed: %s", exc)
            return ""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_vllm_manager.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/serving/vllm_manager.py tests/test_vllm_manager.py
git commit -m "refactor(serving): rewrite VLLMManager as client-only with tiered fallback"
```

---

### Task 3: Update serving __init__ exports

**Files:**
- Modify: `src/serving/__init__.py`

- [ ] **Step 1: Update exports to remove subprocess-related items, add new configs**

```python
"""DocWain serving layer — vLLM client, intent routing, fast-path handler."""

from src.serving.config import (
    VLLMInstanceConfig,
    FAST_PATH_CONFIG,
    SMART_PATH_CONFIG,
    OLLAMA_FALLBACK_CONFIG,
    OLLAMA_CLOUD_CONFIG,
    GPU_MODE_FILE,
    get_openai_base_url,
)
from src.serving.vllm_manager import VLLMManager
from src.serving.model_router import IntentRouter, RouterResult
from src.serving.fast_path import FastPathHandler

__all__ = [
    "VLLMInstanceConfig",
    "FAST_PATH_CONFIG",
    "SMART_PATH_CONFIG",
    "OLLAMA_FALLBACK_CONFIG",
    "OLLAMA_CLOUD_CONFIG",
    "GPU_MODE_FILE",
    "get_openai_base_url",
    "VLLMManager",
    "IntentRouter",
    "RouterResult",
    "FastPathHandler",
]
```

- [ ] **Step 2: Commit**

```bash
git add src/serving/__init__.py
git commit -m "feat(serving): update exports for client-only VLLMManager"
```

---

### Task 4: Wire VLLMManager into AppState and app lifecycle

**Files:**
- Modify: `src/api/rag_state.py`
- Modify: `src/api/app_lifespan.py`

- [ ] **Step 1: Add vllm_manager field to AppState**

In `src/api/rag_state.py`, add after the `graph_augmenter` field:

```python
@dataclass
class AppState:
    embedding_model: Any
    reranker: Any
    qdrant_client: Any
    redis_client: Any
    ollama_client: Any
    rag_system: Any
    llm_gateway: Any = None
    multi_agent_gateway: Any = None
    graph_augmenter: Any = None
    vllm_manager: Any = None  # src.serving.vllm_manager.VLLMManager — dual vLLM client
    instance_ids: Dict[str, str] = field(default_factory=dict)
    qdrant_index_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
```

- [ ] **Step 2: Initialize VLLMManager in app_lifespan.py**

In `src/api/app_lifespan.py`, add after the LLM gateway initialization (after `set_llm_gateway(llm_gateway)` around line 98):

```python
    # vLLM dual-instance client (systemd manages the processes)
    vllm_manager = None
    if Config.VLLM.ENABLED:
        from src.serving.vllm_manager import VLLMManager
        vllm_manager = VLLMManager(
            fast_url=Config.VLLM.FAST_URL,
            smart_url=Config.VLLM.SMART_URL,
            fast_model=Config.VLLM.FAST_MODEL,
            smart_model=Config.VLLM.SMART_MODEL,
            gpu_mode_file=Config.VLLM.GPU_MODE_FILE,
        )
        backends = vllm_manager.get_active_backends()
        logger.info("vLLM backends: %s", backends)
```

Then in the `AppState(...)` constructor call (around line 204), add:

```python
    state = AppState(
        embedding_model=embedding_model,
        reranker=reranker,
        qdrant_client=qdrant_client,
        redis_client=redis_client,
        ollama_client=ollama_client,
        rag_system=rag_system,
        llm_gateway=llm_gateway,
        multi_agent_gateway=multi_agent_gateway,
        graph_augmenter=graph_augmenter,
        vllm_manager=vllm_manager,  # <-- add this
    )
```

- [ ] **Step 3: Ensure query pipeline receives vllm_manager from AppState**

Check `src/query/pipeline.py` — the clients dict is assembled by the API route handler. Find where `run_query_pipeline` is called and ensure `vllm_manager` is included. Search for where clients dict is built:

```bash
grep -rn "run_query_pipeline\|clients.*vllm_manager\|vllm_manager.*clients" src/api/
```

If clients is built in an API route handler, add:

```python
clients = {
    "vllm_manager": app.state.rag_state.vllm_manager,  # already unpacked in pipeline.py line 87
    "llm_gateway": app.state.rag_state.llm_gateway,
    "qdrant_client": app.state.rag_state.qdrant_client,
    ...
}
```

The pipeline already unpacks `vllm_manager` from clients at line 87 and passes it to IntentRouter, QueryPlanner, and ResponseGenerator. No changes needed in pipeline.py itself.

- [ ] **Step 4: Commit**

```bash
git add src/api/rag_state.py src/api/app_lifespan.py
git commit -m "feat(api): wire VLLMManager into AppState and app lifecycle"
```

---

### Task 5: Create systemd unit files

**Files:**
- Create: `systemd/docwain-vllm-fast.service`
- Create: `systemd/docwain-vllm-smart.service`

- [ ] **Step 1: Create the fast instance service**

Create `systemd/docwain-vllm-fast.service`:

```ini
[Unit]
Description=DocWain vLLM Fast Instance (14B)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/PycharmProjects/DocWain
Environment="PATH=/home/ubuntu/PycharmProjects/DocWain/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model models/docwain-v2-active \
    --served-model-name docwain-fast \
    --port 8100 \
    --host 0.0.0.0 \
    --dtype fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.25 \
    --enable-prefix-caching \
    --guided-decoding-backend outlines \
    --tensor-parallel-size 1
Restart=on-failure
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Create the smart instance service**

Create `systemd/docwain-vllm-smart.service`:

```ini
[Unit]
Description=DocWain vLLM Smart Instance (27B)
After=network.target docwain-vllm-fast.service
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/PycharmProjects/DocWain
Environment="PATH=/home/ubuntu/PycharmProjects/DocWain/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-27B \
    --served-model-name docwain-smart \
    --port 8200 \
    --host 0.0.0.0 \
    --dtype fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.65 \
    --enable-chunked-prefill \
    --guided-decoding-backend outlines \
    --tensor-parallel-size 1
Restart=on-failure
RestartSec=10
TimeoutStartSec=600
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 3: Create the model symlink directory**

```bash
mkdir -p models/docwain-v2-active
mkdir -p models/docwain-v2-runs
```

If there's an existing fine-tuned GGUF, symlink it:

```bash
# Example — update path to actual latest model
ln -sfn /home/ubuntu/PycharmProjects/DocWain/finetune_artifacts/v2_upgrade/excel_csv/iter_1/model/merged_16bit models/docwain-v2-active
```

- [ ] **Step 4: Commit**

```bash
git add -f systemd/
git commit -m "feat(systemd): add vLLM fast and smart service unit files"
```

---

### Task 6: Create GPU scheduler daemon

**Files:**
- Create: `scripts/gpu_scheduler.py`
- Create: `systemd/docwain-gpu-scheduler.service`
- Test: `tests/test_gpu_scheduler.py`

- [ ] **Step 1: Write failing tests for GPU scheduler logic**

Create `tests/test_gpu_scheduler.py`:

```python
"""Tests for GPU scheduler decision logic."""

import json
import pytest
from pathlib import Path


def _make_scheduler(tmp_path, mode="serving", training_pending=False):
    """Create a GPUScheduler with controllable state files."""
    mode_file = tmp_path / "gpu-mode.json"
    mode_file.write_text(json.dumps({"mode": mode}))

    queue_file = tmp_path / "training-queue.json"
    queue_file.write_text(json.dumps({"pending": training_pending}))

    from scripts.gpu_scheduler import GPUScheduler
    return GPUScheduler(
        gpu_mode_file=str(mode_file),
        training_queue_file=str(queue_file),
        vllm_fast_url="http://127.0.0.1:19999",
        vllm_smart_url="http://127.0.0.1:19998",
        idle_threshold_minutes=30,
    )


class TestGPUSchedulerDecision:
    def test_serving_no_training_pending(self, tmp_path):
        sched = _make_scheduler(tmp_path, mode="serving", training_pending=False)
        action = sched.decide()
        assert action == "keep_serving"

    def test_serving_training_pending_not_idle(self, tmp_path):
        sched = _make_scheduler(tmp_path, mode="serving", training_pending=True)
        sched._last_request_time = __import__("time").time()  # just now
        action = sched.decide()
        assert action == "keep_serving"

    def test_serving_training_pending_idle(self, tmp_path):
        sched = _make_scheduler(tmp_path, mode="serving", training_pending=True)
        sched._last_request_time = __import__("time").time() - 3600  # 1 hour ago
        action = sched.decide()
        assert action == "enter_training"

    def test_training_complete(self, tmp_path):
        sched = _make_scheduler(tmp_path, mode="training", training_pending=False)
        action = sched.decide()
        assert action == "resume_serving"

    def test_training_in_progress(self, tmp_path):
        sched = _make_scheduler(tmp_path, mode="training", training_pending=True)
        action = sched.decide()
        assert action == "continue_training"


class TestGPUSchedulerModeFile:
    def test_write_mode(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        sched.set_gpu_mode("training")

        mode_file = tmp_path / "gpu-mode.json"
        data = json.loads(mode_file.read_text())
        assert data["mode"] == "training"
        assert "since" in data

    def test_read_mode(self, tmp_path):
        sched = _make_scheduler(tmp_path, mode="training")
        assert sched.get_gpu_mode() == "training"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_gpu_scheduler.py -v
```

Expected: FAIL — `scripts.gpu_scheduler` does not exist.

- [ ] **Step 3: Write the GPU scheduler**

Create `scripts/gpu_scheduler.py`:

```python
#!/usr/bin/env python3
"""GPU Scheduler — manages coexistence of vLLM serving and training.

Runs as a daemon (via systemd). Every 5 minutes:
1. Checks vLLM request rate
2. Checks training queue
3. Decides: keep serving, enter training, or resume serving
4. Executes the decision (stop/start systemd services, run trainer)

Usage:
    python scripts/gpu_scheduler.py
    python scripts/gpu_scheduler.py --once  # single check, no loop
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_DIR / "finetune_artifacts" / "v2_upgrade" / "gpu_scheduler.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [gpu-scheduler] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class GPUScheduler:
    """Decision engine for GPU time-sharing between vLLM and training."""

    def __init__(
        self,
        gpu_mode_file: str = "/tmp/docwain-gpu-mode.json",
        training_queue_file: str = "finetune_artifacts/v2_upgrade/training_queue.json",
        vllm_fast_url: str = "http://localhost:8100",
        vllm_smart_url: str = "http://localhost:8200",
        idle_threshold_minutes: int = 30,
    ) -> None:
        self._gpu_mode_file = gpu_mode_file
        self._training_queue_file = training_queue_file
        self._vllm_fast_url = vllm_fast_url
        self._vllm_smart_url = vllm_smart_url
        self._idle_threshold_s = idle_threshold_minutes * 60
        self._last_request_time: float = time.time()

    # -- State -----------------------------------------------------------------

    def get_gpu_mode(self) -> str:
        try:
            with open(self._gpu_mode_file) as f:
                return json.load(f).get("mode", "serving")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return "serving"

    def set_gpu_mode(self, mode: str) -> None:
        data = {
            "mode": mode,
            "since": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._gpu_mode_file, "w") as f:
            json.dump(data, f)
        logger.info("GPU mode set to: %s", mode)

    def is_training_pending(self) -> bool:
        try:
            with open(self._training_queue_file) as f:
                return json.load(f).get("pending", False)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False

    def clear_training_queue(self) -> None:
        data = {"pending": False, "cleared_at": datetime.now(timezone.utc).isoformat()}
        with open(self._training_queue_file, "w") as f:
            json.dump(data, f)

    # -- vLLM metrics ----------------------------------------------------------

    def _check_vllm_request_rate(self) -> Optional[float]:
        """Query vLLM /metrics for recent request count. Returns None if unavailable."""
        import urllib.request
        import urllib.error

        try:
            url = f"{self._vllm_fast_url}/metrics"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                text = resp.read().decode()
            # Parse Prometheus format: vllm:num_requests_running
            for line in text.splitlines():
                if line.startswith("vllm:num_requests_running"):
                    val = float(line.split()[-1])
                    if val > 0:
                        self._last_request_time = time.time()
                    return val
        except (urllib.error.URLError, OSError, ValueError):
            pass
        return None

    def _is_idle(self) -> bool:
        """Check if vLLM has been idle beyond the threshold."""
        self._check_vllm_request_rate()
        elapsed = time.time() - self._last_request_time
        return elapsed >= self._idle_threshold_s

    # -- Decision --------------------------------------------------------------

    def decide(self) -> str:
        """Return action: keep_serving, enter_training, continue_training, resume_serving."""
        mode = self.get_gpu_mode()
        pending = self.is_training_pending()

        if mode == "training":
            if not pending:
                return "resume_serving"
            return "continue_training"

        # mode == "serving"
        if not pending:
            return "keep_serving"

        if self._is_idle():
            return "enter_training"

        return "keep_serving"

    # -- Actions ---------------------------------------------------------------

    def enter_training_mode(self) -> None:
        """Stop vLLM, set training mode, run trainer."""
        logger.info("Entering training mode — stopping vLLM instances")
        self.set_gpu_mode("training")

        # Wait for in-flight requests to drain
        time.sleep(10)

        # Stop vLLM (smart first — larger)
        _systemctl("stop", "docwain-vllm-smart")
        _systemctl("stop", "docwain-vllm-fast")

        logger.info("vLLM stopped — starting training")
        self._run_training()

    def resume_serving_mode(self) -> None:
        """Start vLLM, set serving mode."""
        logger.info("Resuming serving mode — starting vLLM instances")

        # Update model symlink if new weights exist
        self._hot_swap_model()

        _systemctl("start", "docwain-vllm-fast")
        _systemctl("start", "docwain-vllm-smart")

        self.set_gpu_mode("serving")
        self.clear_training_queue()
        logger.info("vLLM instances started — serving mode active")

    def _run_training(self) -> None:
        """Run the autonomous trainer as a subprocess."""
        cmd = [
            sys.executable, "-m", "src.finetune.v2.autonomous_trainer", "--resume",
        ]
        logger.info("Starting trainer: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=86400,  # 24h max
        )
        if result.returncode == 0:
            logger.info("Training completed successfully")
        else:
            logger.error("Training failed (rc=%d): %s", result.returncode, result.stderr[-500:])

        self.clear_training_queue()

    def _hot_swap_model(self) -> None:
        """Update the model symlink to the latest fine-tuned weights."""
        state_file = PROJECT_DIR / "finetune_artifacts" / "v2_upgrade" / "state.json"
        if not state_file.exists():
            return

        try:
            state = json.loads(state_file.read_text())
            checkpoint = state.get("last_checkpoint")
            if not checkpoint:
                return

            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_dir():
                checkpoint_path = PROJECT_DIR / checkpoint
            if not checkpoint_path.is_dir():
                logger.warning("Checkpoint not found: %s", checkpoint)
                return

            symlink = PROJECT_DIR / "models" / "docwain-v2-active"
            if symlink.is_symlink():
                current = symlink.resolve()
                if current == checkpoint_path.resolve():
                    logger.info("Model symlink already up to date")
                    return

            symlink.unlink(missing_ok=True)
            symlink.symlink_to(checkpoint_path.resolve())
            logger.info("Model symlink updated: %s → %s", symlink, checkpoint_path.resolve())
        except Exception as exc:
            logger.error("Hot-swap failed: %s", exc)

    # -- Main loop -------------------------------------------------------------

    def run_loop(self, interval_s: int = 300) -> None:
        """Run the scheduler loop forever."""
        logger.info("GPU scheduler started (check every %ds, idle threshold %ds)",
                     interval_s, self._idle_threshold_s)
        while True:
            try:
                action = self.decide()
                logger.info("Decision: %s", action)

                if action == "enter_training":
                    self.enter_training_mode()
                    self.resume_serving_mode()
                elif action == "resume_serving":
                    self.resume_serving_mode()
                # keep_serving, continue_training — do nothing
            except Exception as exc:
                logger.error("Scheduler error: %s", exc, exc_info=True)

            time.sleep(interval_s)


def _systemctl(action: str, service: str) -> bool:
    """Run systemctl action on a service. Returns True on success."""
    try:
        result = subprocess.run(
            ["sudo", "systemctl", action, service],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            logger.info("systemctl %s %s — OK", action, service)
            return True
        logger.warning("systemctl %s %s — failed: %s", action, service, result.stderr.strip())
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.error("systemctl %s %s — error: %s", action, service, exc)
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DocWain GPU Scheduler")
    parser.add_argument("--once", action="store_true", help="Single check, no loop")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--idle-minutes", type=int, default=30, help="Idle threshold in minutes")
    args = parser.parse_args()

    os.chdir(str(PROJECT_DIR))

    scheduler = GPUScheduler(idle_threshold_minutes=args.idle_minutes)

    if args.once:
        action = scheduler.decide()
        print(f"Decision: {action}")
        if action == "enter_training":
            scheduler.enter_training_mode()
            scheduler.resume_serving_mode()
        elif action == "resume_serving":
            scheduler.resume_serving_mode()
    else:
        scheduler.run_loop(interval_s=args.interval)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_gpu_scheduler.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Create the systemd service file**

Create `systemd/docwain-gpu-scheduler.service`:

```ini
[Unit]
Description=DocWain GPU Scheduler
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/PycharmProjects/DocWain
Environment="PATH=/home/ubuntu/PycharmProjects/DocWain/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python scripts/gpu_scheduler.py --interval 300 --idle-minutes 30
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 6: Commit**

```bash
git add scripts/gpu_scheduler.py tests/test_gpu_scheduler.py
git add -f systemd/docwain-gpu-scheduler.service
git commit -m "feat(scheduler): add GPU scheduler for vLLM/training coexistence"
```

---

### Task 7: Install and activate systemd services

**Files:** systemd units created in Task 5 and 6

- [ ] **Step 1: Create model symlink**

```bash
mkdir -p models/docwain-v2-runs
# Point to current best model (iter_1 merged checkpoint)
ln -sfn /home/ubuntu/PycharmProjects/DocWain/finetune_artifacts/v2_upgrade/excel_csv/iter_1/model/merged_16bit models/docwain-v2-active
```

- [ ] **Step 2: Initialize GPU mode file**

```bash
echo '{"mode": "serving", "since": "'$(date -Iseconds)'"}' > /tmp/docwain-gpu-mode.json
```

- [ ] **Step 3: Install systemd units**

```bash
sudo cp systemd/docwain-vllm-fast.service /etc/systemd/system/
sudo cp systemd/docwain-vllm-smart.service /etc/systemd/system/
sudo cp systemd/docwain-gpu-scheduler.service /etc/systemd/system/
sudo systemctl daemon-reload
```

- [ ] **Step 4: Start vLLM fast instance and verify**

```bash
sudo systemctl start docwain-vllm-fast
# Wait for health check
sleep 30
curl -s http://localhost:8100/health
# Expected: 200 OK
sudo systemctl status docwain-vllm-fast
```

- [ ] **Step 5: Start vLLM smart instance and verify**

```bash
sudo systemctl start docwain-vllm-smart
sleep 60
curl -s http://localhost:8200/health
sudo systemctl status docwain-vllm-smart
```

- [ ] **Step 6: Quick smoke test — query both instances**

```bash
# Fast instance
curl -s http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"docwain-fast","messages":[{"role":"user","content":"Hello, what can you do?"}],"max_tokens":100}' | python3 -m json.tool

# Smart instance
curl -s http://localhost:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"docwain-smart","messages":[{"role":"user","content":"Compare the concepts of liability and indemnity in contract law."}],"max_tokens":200}' | python3 -m json.tool
```

- [ ] **Step 7: Enable services for auto-start on boot**

```bash
sudo systemctl enable docwain-vllm-fast
sudo systemctl enable docwain-vllm-smart
sudo systemctl enable docwain-gpu-scheduler
```

- [ ] **Step 8: Start GPU scheduler**

```bash
sudo systemctl start docwain-gpu-scheduler
sudo systemctl status docwain-gpu-scheduler
```

---

### Task 8: Integration test — full query pipeline through vLLM

**Files:**
- Create: `tests/test_vllm_integration.py`

- [ ] **Step 1: Write integration test**

```python
"""Integration test — verifies the full query pipeline routes through vLLM.

Requires vLLM fast instance running on port 8100.
Skip if not available.
"""

import json
import pytest
from urllib import request as urllib_request
from urllib.error import URLError


def _vllm_available(port: int) -> bool:
    try:
        req = urllib_request.Request(f"http://localhost:{port}/health")
        with urllib_request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (URLError, OSError):
        return False


@pytest.mark.skipif(
    not _vllm_available(8100),
    reason="vLLM fast instance not running on port 8100",
)
class TestVLLMIntegration:
    def test_vllm_manager_query_fast(self):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager()
        result = mgr.query_fast("What is DocWain?")
        assert len(result) > 10
        assert isinstance(result, str)

    def test_vllm_manager_health_check(self):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager()
        assert mgr.health_check("fast") is True
        backends = mgr.get_active_backends()
        assert backends["fast"] is True

    def test_vllm_manager_fallback_on_bad_instance(self):
        from src.serving.vllm_manager import VLLMManager

        mgr = VLLMManager(smart_url="http://localhost:19999")
        # Smart is down, should fall back to Ollama Cloud or local
        result = mgr.query_smart("Hello")
        # Should get some response (from fallback), not empty
        assert isinstance(result, str)

    def test_gpu_mode_serving(self, tmp_path):
        from src.serving.vllm_manager import VLLMManager

        mode_file = tmp_path / "mode.json"
        mode_file.write_text(json.dumps({"mode": "serving"}))
        mgr = VLLMManager(gpu_mode_file=str(mode_file))
        assert mgr.get_gpu_mode() == "serving"


@pytest.mark.skipif(
    not _vllm_available(8100),
    reason="vLLM fast instance not running on port 8100",
)
class TestVLLMIntentRouting:
    def test_intent_router_classifies(self):
        from src.serving.vllm_manager import VLLMManager
        from src.serving.model_router import IntentRouter

        mgr = VLLMManager()
        router = IntentRouter(mgr)
        result = router.route("What is the total amount on this invoice?")
        assert result.route in ("fast", "smart")
        assert result.intent is not None
```

- [ ] **Step 2: Run integration tests**

```bash
pytest tests/test_vllm_integration.py -v
```

Expected: PASS if vLLM is running, SKIP otherwise.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vllm_integration.py
git commit -m "test: add vLLM integration tests for full pipeline"
```

---

### Task 9: Final verification and cleanup

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/test_vllm_manager.py tests/test_gpu_scheduler.py tests/test_vllm_integration.py -v
```

- [ ] **Step 2: Verify all services are running**

```bash
sudo systemctl status docwain-vllm-fast docwain-vllm-smart docwain-gpu-scheduler
```

- [ ] **Step 3: Verify fallback chain works**

```bash
# Test: stop smart, query smart — should fallback
sudo systemctl stop docwain-vllm-smart
python3 -c "
from src.serving.vllm_manager import VLLMManager
mgr = VLLMManager()
print('Smart health:', mgr.health_check('smart'))
print('Response:', mgr.query_smart('Hello')[:100])
"
# Restart smart
sudo systemctl start docwain-vllm-smart
```

- [ ] **Step 4: Verify training mode fallback**

```bash
# Set training mode
echo '{"mode": "training"}' > /tmp/docwain-gpu-mode.json
python3 -c "
from src.serving.vllm_manager import VLLMManager
mgr = VLLMManager()
print('GPU mode:', mgr.get_gpu_mode())
print('Response:', mgr.query_fast('Hello')[:100])
"
# Reset to serving
echo '{"mode": "serving"}' > /tmp/docwain-gpu-mode.json
```

- [ ] **Step 5: Final commit with any remaining changes**

```bash
git add -A
git status
# Only commit if there are changes
git commit -m "chore: finalize vLLM production wiring"
```
