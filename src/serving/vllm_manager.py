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
        vLLM instance -> Ollama Cloud (397b) -> Ollama local (14b)

    Fallback chain (training mode):
        Ollama Cloud (397b) -> Ollama local (14b)
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

    def get_active_backends(self) -> Dict[str, Any]:
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
        if self.get_gpu_mode() == "training":
            logger.info("GPU in training mode — routing to Ollama Cloud")
            return self._query_ollama_cloud(prompt, system_prompt, max_tokens, temperature)

        instance = self._instances.get(instance_name)
        if instance and self.health_check(instance_name):
            result = self._query_vllm(instance, prompt, system_prompt, guided_json, max_tokens, temperature)
            if result is not None:
                return result

        logger.info("vLLM '%s' unavailable — falling back to Ollama Cloud", instance_name)
        cloud_result = self._query_ollama_cloud(prompt, system_prompt, max_tokens, temperature)
        if cloud_result:
            return cloud_result

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
