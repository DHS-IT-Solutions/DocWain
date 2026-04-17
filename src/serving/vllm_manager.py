"""Client-only vLLM manager — routes queries to the unified vLLM instance.

Single DocWain model, no fast/smart split. Health checks via HTTP. Falls back
to Ollama Cloud during training, Ollama local as emergency.
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

_DEFAULT_MAX_MODEL_LEN = 4096


class VLLMManager:
    """Client-only manager for the unified DocWain vLLM instance.

    Fallback chain (serving mode):
        vLLM -> Ollama Cloud (397b) -> Ollama local (14b)

    Fallback chain (training mode):
        Ollama Cloud (397b) -> Ollama local (14b)
    """

    def __init__(
        self,
        url: str = "http://localhost:8100",
        model: str = "docwain-fast",
        gpu_mode_file: str = GPU_MODE_FILE,
    ) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._gpu_mode_file = gpu_mode_file
        self._cached_max_model_len: Optional[int] = None

    # -- Status ----------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True if the vLLM instance responds on /health."""
        try:
            req = urllib_request.Request(f"{self._url}/health", method="GET")
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
        """Return health status of the unified backend."""
        return {
            "docwain": self.health_check(),
            "gpu_mode": self.get_gpu_mode(),
        }

    def get_max_model_len(self) -> int:
        """Fetch max_model_len from the vLLM /v1/models endpoint (cached)."""
        if self._cached_max_model_len is not None:
            return self._cached_max_model_len
        try:
            req = urllib_request.Request(f"{self._url}/v1/models", method="GET")
            with urllib_request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
            for model_info in result.get("data", []):
                mml = model_info.get("max_model_len")
                if mml:
                    self._cached_max_model_len = int(mml)
                    return int(mml)
        except (HTTPError, URLError, OSError, json.JSONDecodeError, ValueError) as exc:
            logger.debug("Could not fetch max_model_len: %s", exc)
        return _DEFAULT_MAX_MODEL_LEN

    def _clamp_max_tokens(self, prompt_text: str, max_tokens: int) -> int:
        """Clamp max_tokens so prompt + generation fits within the context window."""
        model_limit = self.get_max_model_len()
        estimated_prompt_tokens = len(prompt_text) // 4
        available = max(model_limit - estimated_prompt_tokens, 256)
        if max_tokens > available:
            logger.info(
                "Clamping max_tokens %d -> %d (model_limit=%d, est_prompt_tokens=%d)",
                max_tokens, available, model_limit, estimated_prompt_tokens,
            )
            return available
        return max_tokens

    # -- Query interface -------------------------------------------------------

    def query(
        self,
        prompt: str,
        system_prompt: str = "",
        guided_json: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        *,
        require_vllm: bool = False,
    ) -> str:
        """Query the unified DocWain vLLM instance.

        By default, uses the tiered fallback chain
        (vLLM → Ollama Cloud → Ollama local) so training / background
        inference can degrade gracefully. Extraction and embedding
        callers should pass ``require_vllm=True`` — those paths must
        stay on the in-house model for accuracy consistency, and a
        silent fallback to a 3rd-party cloud produces results we can't
        reproduce or audit (impact #7).

        When ``require_vllm=True``:
            * training-mode GPU → ``ConnectionError`` (no Ollama)
            * vLLM down or returns None → ``ConnectionError``
            * no Ollama fallback is ever attempted
        """
        if require_vllm:
            if self.get_gpu_mode() == "training":
                raise ConnectionError(
                    "vLLM is in training mode; require_vllm=True caller cannot "
                    "fall back to Ollama — wait for serving mode."
                )
            if not self.health_check():
                raise ConnectionError(
                    "vLLM instance is unreachable; require_vllm=True caller "
                    "refuses to fall back to a 3rd-party backend."
                )
            full_text = f"{system_prompt}\n{prompt}" if system_prompt else prompt
            clamped = self._clamp_max_tokens(full_text, max_tokens)
            result = self._query_vllm(
                prompt, system_prompt, guided_json, clamped, temperature,
            )
            if result is None:
                raise ConnectionError(
                    "vLLM returned no result; require_vllm=True caller "
                    "refuses to fall back."
                )
            return result

        if self.get_gpu_mode() == "training":
            logger.info("GPU in training mode — routing to Ollama Cloud")
            return self._query_ollama_cloud(prompt, system_prompt, max_tokens, temperature)

        if self.health_check():
            full_text = f"{system_prompt}\n{prompt}" if system_prompt else prompt
            clamped = self._clamp_max_tokens(full_text, max_tokens)
            result = self._query_vllm(prompt, system_prompt, guided_json, clamped, temperature)
            if result is not None:
                return result

        logger.info("vLLM unavailable — falling back to Ollama Cloud")
        cloud_result = self._query_ollama_cloud(prompt, system_prompt, max_tokens, temperature)
        if cloud_result:
            return cloud_result

        logger.warning("Ollama Cloud unavailable — emergency fallback to Ollama local")
        return self._query_ollama_local(prompt, system_prompt, max_tokens, temperature)

    # -- Streaming interface ---------------------------------------------------

    def stream_query(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """Stream tokens from the unified vLLM instance.

        Raises on failure so the caller (gateway) can fall back to Ollama.
        """
        if self.get_gpu_mode() == "training":
            logger.info("GPU in training mode — streaming unavailable, degrading to non-streaming")
            yield self._query_ollama_cloud(prompt, system_prompt, max_tokens, temperature)
            return

        if not self.health_check():
            raise ConnectionError("DocWain vLLM instance is not available")

        full_text = f"{system_prompt}\n{prompt}" if system_prompt else prompt
        clamped = self._clamp_max_tokens(full_text, max_tokens)
        yield from self._stream_vllm(prompt, system_prompt, clamped, temperature)

    # -- Private: vLLM ---------------------------------------------------------

    @staticmethod
    def _prepare_messages(prompt: str, system_prompt: str) -> list:
        """Build chat messages with /no_think to disable Qwen3 thinking mode."""
        messages = []
        sys_content = f"/no_think\n{system_prompt}" if system_prompt else "/no_think"
        messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _query_vllm(
        self,
        prompt: str,
        system_prompt: str,
        guided_json: Optional[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        """POST to vLLM OpenAI-compatible endpoint. Returns None on failure."""
        url = f"{self._url}/v1/chat/completions"
        messages = self._prepare_messages(prompt, system_prompt)

        body: Dict[str, Any] = {
            "model": self._model,
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
            logger.error("vLLM query failed (%s): %s", self._url, exc)
            return None

    def _stream_vllm(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Generator[str, None, None]:
        """POST to vLLM with stream=True, yield token deltas from SSE chunks."""
        url = f"{self._url}/v1/chat/completions"
        messages = self._prepare_messages(prompt, system_prompt)

        body: Dict[str, Any] = {
            "model": self._model,
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
            in_think = False
            think_buf = ""
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace")
                sse_buf += line
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
                            elif "<think>".startswith(think_buf):
                                pass
                            else:
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
