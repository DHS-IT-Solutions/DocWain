"""vLLM HTTP client with OpenAI-compatible vision payload support.

Talks to the local vLLM server serving DocWain (default port 8100). Builds
multi-part messages with a single page image per call (OpenAI vision format).
Thin wrapper by design — no retry logic, no batching, no streaming. Callers
handle errors via VisionClientError.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


class VisionClientError(Exception):
    """Raised when the vision server returns an error or unparseable response."""


@dataclass
class VisionResponse:
    """Raw text output + usage metadata from a single vision call."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    wall_ms: float
    model: str


class VisionClient:
    def __init__(self, *, base_url: str, model: str, timeout_s: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def build_payload(
        self,
        *,
        system: str,
        user_text: str,
        image_bytes: bytes,
        image_mime: str = "image/png",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{image_mime};base64,{b64}"
        return {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }

    def call(
        self,
        *,
        system: str,
        user_text: str,
        image_bytes: bytes,
        image_mime: str = "image/png",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VisionResponse:
        import time
        payload = self.build_payload(
            system=system,
            user_text=user_text,
            image_bytes=image_bytes,
            image_mime=image_mime,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        url = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()
        try:
            r = httpx.post(url, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise VisionClientError(f"vision call failed: {exc}") from exc
        wall_ms = (time.perf_counter() - t0) * 1000.0
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
        except Exception as exc:
            raise VisionClientError(f"unparseable vision response: {exc}") from exc
        return VisionResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            wall_ms=float(wall_ms),
            model=self.model,
        )
