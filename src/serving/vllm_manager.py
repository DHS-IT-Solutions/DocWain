"""Manages vLLM instance lifecycle and provides a unified query interface."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any, Dict, Optional
from urllib import request
from urllib.error import HTTPError, URLError

from src.serving.config import (
    FAST_PATH_CONFIG,
    OLLAMA_FALLBACK_CONFIG,
    SMART_PATH_CONFIG,
    VLLMInstanceConfig,
    get_openai_base_url,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VLLMManager:
    """Thread-safe manager for dual vLLM serving instances with Ollama fallback.

    Usage::

        mgr = VLLMManager()
        mgr.start_all()
        answer = mgr.query_fast("What is this document about?")
        mgr.stop_all()
    """

    def __init__(
        self,
        fast_config: VLLMInstanceConfig = FAST_PATH_CONFIG,
        smart_config: VLLMInstanceConfig = SMART_PATH_CONFIG,
    ) -> None:
        self._configs: Dict[str, VLLMInstanceConfig] = {
            fast_config.name: fast_config,
            smart_config.name: smart_config,
        }
        self._processes: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # -- Lifecycle ------------------------------------------------------------

    def start_instance(self, config: VLLMInstanceConfig) -> bool:
        """Start a vLLM instance as a subprocess. Returns True on success."""
        with self._lock:
            if config.name in self._processes:
                proc = self._processes[config.name]
                if proc.poll() is None:
                    logger.info("vLLM instance '%s' already running (pid %d)", config.name, proc.pid)
                    return True

            cmd = ["python", "-m", "vllm.entrypoints.openai.api_server"] + config.to_vllm_args()
            logger.info("Starting vLLM instance '%s': %s", config.name, " ".join(cmd))

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self._processes[config.name] = proc
                self._configs[config.name] = config
            except FileNotFoundError:
                logger.error("vLLM not installed — cannot start instance '%s'", config.name)
                return False
            except Exception as exc:
                logger.error("Failed to start vLLM instance '%s': %s", config.name, exc)
                return False

        # Wait for readiness (up to 120 s).
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if self.health_check(config.name):
                logger.info("vLLM instance '%s' is ready on port %d", config.name, config.port)
                return True
            time.sleep(2.0)

        logger.warning("vLLM instance '%s' did not become healthy within timeout", config.name)
        return False

    def stop_instance(self, name: str) -> None:
        """Stop a running vLLM instance."""
        with self._lock:
            proc = self._processes.pop(name, None)
        if proc is None:
            return
        logger.info("Stopping vLLM instance '%s' (pid %d)", name, proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("Force-killing vLLM instance '%s'", name)
            proc.kill()
            proc.wait(timeout=5)

    def health_check(self, name: str) -> bool:
        """Return True if the named vLLM instance is responding on /health."""
        config = self._configs.get(name)
        if config is None:
            return False
        url = f"http://localhost:{config.port}/health"
        try:
            req = request.Request(url, method="GET")
            with request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (HTTPError, URLError, OSError, TimeoutError):
            return False

    def start_all(self) -> Dict[str, bool]:
        """Start both fast and smart instances. Returns {name: success}."""
        results = {}
        for name, config in list(self._configs.items()):
            results[name] = self.start_instance(config)
        return results

    def stop_all(self) -> None:
        """Stop all running vLLM instances."""
        for name in list(self._processes.keys()):
            self.stop_instance(name)

    # -- Query interface ------------------------------------------------------

    def query(
        self,
        instance_name: str,
        prompt: str,
        system_prompt: str = "",
        guided_json: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Send a chat completion request to the named vLLM instance.

        Falls back to Ollama if the instance is not available.
        """
        config = self._configs.get(instance_name)
        if config is None:
            logger.warning("Unknown instance '%s' — falling back to Ollama", instance_name)
            return self._query_ollama(prompt, system_prompt, max_tokens, temperature)

        if not self.health_check(instance_name):
            logger.info(
                "vLLM instance '%s' unavailable — falling back to Ollama",
                instance_name,
            )
            return self._query_ollama(prompt, system_prompt, max_tokens, temperature)

        return self._query_vllm(config, prompt, system_prompt, guided_json, max_tokens, temperature)

    def query_fast(
        self,
        prompt: str,
        system_prompt: str = "",
        guided_json: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Convenience: query the fast (14B) instance."""
        return self.query("fast", prompt, system_prompt, guided_json, max_tokens, temperature)

    def query_smart(
        self,
        prompt: str,
        system_prompt: str = "",
        guided_json: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Convenience: query the smart (27B) instance."""
        return self.query("smart", prompt, system_prompt, guided_json, max_tokens, temperature)

    # -- Private helpers ------------------------------------------------------

    def _query_vllm(
        self,
        config: VLLMInstanceConfig,
        prompt: str,
        system_prompt: str,
        guided_json: Optional[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """POST to the OpenAI-compatible /v1/chat/completions endpoint."""
        url = f"{get_openai_base_url(config)}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": config.model,
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

        payload = json.dumps(body).encode()
        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
            content: str = data["choices"][0]["message"]["content"]
            return content.strip()
        except (HTTPError, URLError, OSError, KeyError, json.JSONDecodeError) as exc:
            logger.error("vLLM query to '%s' failed: %s — falling back to Ollama", config.name, exc)
            return self._query_ollama(prompt, system_prompt, max_tokens, temperature)

    def _query_ollama(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Fallback: query the local Ollama HTTP API at localhost:11434."""
        url = f"{OLLAMA_FALLBACK_CONFIG.host}/api/generate"

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        body: Dict[str, Any] = {
            "model": OLLAMA_FALLBACK_CONFIG.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        payload = json.dumps(body).encode()
        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=OLLAMA_FALLBACK_CONFIG.timeout_s) as resp:
                data = json.loads(resp.read().decode())
            text: str = data.get("response", "")
            # Strip <think>...</think> blocks that Qwen models sometimes emit.
            import re
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
        except (HTTPError, URLError, OSError, KeyError, json.JSONDecodeError) as exc:
            logger.error("Ollama fallback query failed: %s", exc)
            return ""
