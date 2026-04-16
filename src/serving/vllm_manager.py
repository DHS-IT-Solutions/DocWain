"""Client-only vLLM manager — routes queries to vLLM instances managed by systemd.

No subprocess management. Health checks via HTTP. Falls back to Ollama Cloud
during training, Ollama local as emergency.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Generator, Optional
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
        # Cache max_model_len per instance (populated lazily)
        self._max_model_len: Dict[str, int] = {}

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

    def get_max_model_len(self, name: str) -> int:
        """Fetch max_model_len from the vLLM /v1/models endpoint (cached)."""
        if name in self._max_model_len:
            return self._max_model_len[name]
        instance = self._instances.get(name)
        if not instance:
            return 4096
        url = f"{instance['url']}/v1/models"
        try:
            req = urllib_request.Request(url, method="GET")
            with urllib_request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
            for model_info in result.get("data", []):
                mml = model_info.get("max_model_len")
                if mml:
                    self._max_model_len[name] = int(mml)
                    return int(mml)
        except (HTTPError, URLError, OSError, json.JSONDecodeError, ValueError) as exc:
            logger.debug("Could not fetch max_model_len for %s: %s", name, exc)
        return 4096  # safe default

    def _clamp_max_tokens(self, instance_name: str, prompt_text: str, max_tokens: int) -> int:
        """Clamp max_tokens so prompt + generation fits within the model's context window."""
        model_limit = self.get_max_model_len(instance_name)
        # Conservative token estimate: ~4 chars per token (Qwen tokenizer typical ratio)
        estimated_prompt_tokens = len(prompt_text) // 4
        available = max(model_limit - estimated_prompt_tokens, 256)
        if max_tokens > available:
            logger.info(
                "Clamping max_tokens %d -> %d for %s (model_limit=%d, est_prompt_tokens=%d)",
                max_tokens, available, instance_name, model_limit, estimated_prompt_tokens,
            )
            return available
        return max_tokens

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
            full_text = f"{system_prompt}\n{prompt}" if system_prompt else prompt
            clamped = self._clamp_max_tokens(instance_name, full_text, max_tokens)
            result = self._query_vllm(instance, prompt, system_prompt, guided_json, clamped, temperature)
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

    # -- Streaming interface ---------------------------------------------------

    def stream_query(
        self,
        instance_name: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """Stream tokens from a vLLM instance.

        Raises on failure so the caller (gateway) can try another instance.
        """
        if self.get_gpu_mode() == "training":
            logger.info("GPU in training mode — streaming unavailable, falling back to non-streaming")
            yield self._query_ollama_cloud(prompt, system_prompt, max_tokens, temperature)
            return

        instance = self._instances.get(instance_name)
        if not instance or not self.health_check(instance_name):
            raise ConnectionError(f"vLLM instance '{instance_name}' is not available")

        full_text = f"{system_prompt}\n{prompt}" if system_prompt else prompt
        clamped = self._clamp_max_tokens(instance_name, full_text, max_tokens)
        yield from self._stream_vllm(instance, prompt, system_prompt, clamped, temperature)

    def stream_query_fast(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """Stream tokens from the fast (14B) instance."""
        yield from self.stream_query("fast", prompt, system_prompt, max_tokens, temperature)

    def stream_query_smart(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """Stream tokens from the smart (27B) instance."""
        yield from self.stream_query("smart", prompt, system_prompt, max_tokens, temperature)

    # -- Private: vLLM ---------------------------------------------------------

    @staticmethod
    def _prepare_messages(prompt: str, system_prompt: str) -> list:
        """Build chat messages with /no_think to disable Qwen3 thinking mode."""
        messages = []
        # Prepend /no_think to suppress <think> blocks — saves tokens and latency
        sys_content = f"/no_think\n{system_prompt}" if system_prompt else "/no_think"
        messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": prompt})
        return messages

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

        messages = self._prepare_messages(prompt, system_prompt)

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
            text = result["choices"][0]["message"]["content"].strip()
            return _THINK_RE.sub("", text).strip()
        except (HTTPError, URLError, OSError, KeyError, json.JSONDecodeError) as exc:
            logger.error("vLLM query failed (%s): %s", instance["url"], exc)
            return None

    def _stream_vllm(
        self,
        instance: Dict[str, str],
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Generator[str, None, None]:
        """POST to vLLM with stream=True, yield token deltas from SSE chunks."""
        url = f"{instance['url']}/v1/chat/completions"

        messages = self._prepare_messages(prompt, system_prompt)

        body: Dict[str, Any] = {
            "model": instance["model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        data = json.dumps(body).encode()
        req = urllib_request.Request(
            url, data=data,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        try:
            resp = urllib_request.urlopen(req, timeout=300)
        except HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            logger.error("vLLM stream request failed (%s %s): %s", exc.code, exc.reason, error_body)
            raise
        with resp:
            sse_buf = ""
            in_think = False  # stateful: inside <think>...</think>
            think_buf = ""    # partial tag accumulator
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace")
                sse_buf += line
                # SSE lines are newline-delimited
                while "\n" in sse_buf:
                    sse_line, sse_buf = sse_buf.split("\n", 1)
                    sse_line = sse_line.strip()
                    if not sse_line.startswith("data:"):
                        continue
                    payload = sse_line[len("data:"):].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content")
                    if not token:
                        continue

                    # Stateful <think>...</think> filter
                    for ch in token:
                        if in_think:
                            think_buf += ch
                            if think_buf.endswith("</think>"):
                                in_think = False
                                think_buf = ""
                        else:
                            think_buf += ch
                            if think_buf == "<think>":
                                in_think = True
                                # don't clear think_buf — keep accumulating until </think>
                            elif "<think>".startswith(think_buf):
                                # partial match — keep buffering
                                pass
                            else:
                                # not a think tag — flush buffer
                                yield think_buf
                                think_buf = ""

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
