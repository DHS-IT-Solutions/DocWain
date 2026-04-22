"""
LLM Gateway - unified interface to language model backends.

Response generation:
    Primary: Ollama Cloud qwen3.5:397b (high-quality reasoning)
    Fallback: Azure GPT-4o

Document processing uses get_local_client() → local qwen3:14b (fast).

Public API:
    create_llm_gateway() -> LLMGateway
    get_llm_gateway()    -> LLMGateway
    set_llm_gateway(gw)  -> None
"""

from __future__ import annotations

import os
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.api.config import Config
from src.utils.logging_utils import get_logger
from src.llm.health import VLLMHealthMonitor

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    text: str
    thinking: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Thinking-block parser
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_thinking(raw: str) -> Tuple[str, Optional[str]]:
    """Split ``<think>...</think>`` blocks from Qwen3 output.

    Handles both closed ``<think>...</think>`` and unclosed ``<think>...``
    (truncated responses where thinking consumed all tokens).

    Returns:
        (answer_text, thinking_text_or_None)
    """
    match = _THINK_RE.search(raw)
    if match:
        thinking = match.group(1).strip()
        answer = _THINK_RE.sub("", raw).strip()
        return answer, thinking or None

    # Handle unclosed <think> tag (truncated response)
    if "<think>" in raw:
        idx = raw.index("<think>")
        before = raw[:idx].strip()
        after = raw[idx + len("<think>"):].strip()
        # Everything after <think> is thinking; everything before is answer
        if before:
            return before, after or None
        # Only thinking, no answer — return thinking as extractable content
        return "", after or None

    return raw.strip(), None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_gateway_instance: Optional[LLMGateway] = None
_gateway_lock = threading.Lock()


# ---------------------------------------------------------------------------
# LLMGateway
# ---------------------------------------------------------------------------

class LLMGateway:
    """Unified gateway to LLM backends.

    Prioritises vLLM (via ``OpenAICompatibleClient``).  Falls back to Ollama
    only when vLLM is disabled or unhealthy (intended for local dev).
    """

    def __init__(self) -> None:
        self._primary = None  # OpenAICompatibleClient
        self._fallback = None  # OllamaClient (dev only)
        self._health_monitor: Optional[VLLMHealthMonitor] = None

        # Expose for backward compat
        self.model_name: Optional[str] = None
        self.backend: str = "unknown"

        # Stats
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "requests": 0,
            "failures": 0,
            "fallback_used": 0,
            "last_error": None,
            "last_request_ts": None,
        }
        self._created_at = time.time()

        # Cooldown tracking (populated on repeated failures)
        self._cooldown_until: float = 0.0

        self._init_clients()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_clients(self) -> None:
        """Create backend clients in priority order.

        Primary:   vLLM serving the unified DocWain model (local, OpenAI-compat).
        Fallback1: Ollama Cloud qwen3.5:397b (remote, used when vLLM unhealthy).
        Fallback2: Azure GPT-4o (used when both above fail).

        This keeps the fallback chain intact for training windows when
        vLLM is paused by the GPU scheduler. Document processing still
        uses ``get_local_client()`` (a separate path) for ingestion-time
        classification/extraction.
        """
        self._primary = None
        self._fallback = None

        # --- vLLM (primary — unified DocWain model) ---
        if getattr(Config.VLLM, "ENABLED", True):
            try:
                from src.llm.clients import OpenAICompatibleClient
                vllm_client = OpenAICompatibleClient(
                    endpoint=Config.VLLM.URL.rstrip("/") + "/v1",
                    model_name=Config.VLLM.MODEL,
                    api_key=Config.VLLM.API_KEY or "EMPTY",
                )
                # Cheap reachability probe — /v1/models. Avoids wiring an
                # unhealthy primary that would fail every request before
                # fallback kicks in.
                import urllib.request
                with urllib.request.urlopen(
                    Config.VLLM.URL.rstrip("/") + "/v1/models", timeout=5
                ) as r:
                    if r.status != 200:
                        raise RuntimeError(f"vLLM /v1/models returned {r.status}")
                self._primary = vllm_client
                self.backend = "vllm"
                self.model_name = vllm_client.model_name
                logger.info(
                    "vLLM primary ready (endpoint=%s, model=%s)",
                    Config.VLLM.URL, vllm_client.model_name,
                )
            except Exception as exc:
                logger.warning(
                    "vLLM primary unavailable (%s) — falling back to Ollama Cloud",
                    exc,
                )

        # --- Ollama Cloud (fallback #1 — used when vLLM down, e.g. training window) ---
        cloud_model = os.getenv("OLLAMA_CLOUD_MODEL", "qwen3.5:397b")
        ollama_client = None
        try:
            from src.llm.clients import OllamaClient
            ollama_client = OllamaClient(model_name=cloud_model)
            logger.info("Ollama Cloud client initialised (model=%s)", ollama_client.model_name)
        except Exception as exc:
            logger.warning("Failed to create Ollama Cloud client: %s", exc)

        if self._primary is None and ollama_client is not None:
            self._primary = ollama_client
            self.backend = "ollama"
            self.model_name = ollama_client.model_name
            logger.info("Ollama Cloud promoted to primary (vLLM unavailable)")
        elif ollama_client is not None:
            # vLLM is primary; keep Ollama as the active fallback.
            self._fallback = ollama_client

        # --- Azure GPT-4o (fallback #2 — used when vLLM and Ollama Cloud down) ---
        try:
            from src.llm.clients import OpenAIClient
            endpoint = Config.AzureGpt4o.AZUREGPT4O_ENDPOINT
            api_key = Config.AzureGpt4o.AZUREGPT4O_API_KEY
            deployment = Config.AzureGpt4o.AZUREGPT4O_DEPLOYMENT
            api_version = Config.AzureGpt4o.AZUREGPT4O_Version
            if not endpoint or not api_key:
                raise ValueError("AZUREGPT4O_ENDPOINT or AZUREGPT4O_API_KEY not configured")
            azure_client = OpenAIClient(
                endpoint=endpoint,
                api_key=api_key,
                deployment=deployment,
                api_version=api_version,
            )
            if self._primary is None:
                self._primary = azure_client
                self.backend = "azure_openai"
                self.model_name = azure_client.model_name
                logger.info("Azure GPT-4o promoted to primary (vLLM + Ollama unavailable)")
            elif self._fallback is None:
                self._fallback = azure_client
                logger.info("Azure GPT-4o registered as fallback (model=%s)", azure_client.model_name)
            # else: vLLM primary + Ollama fallback already wired; Azure not needed as tertiary.
        except Exception as exc:
            logger.warning("Failed to create Azure GPT-4o fallback: %s", exc)

        if self._primary is None:
            logger.error("No LLM backend available - all calls will fail")

    def _pick_client(self):
        """Return the best available client (Ollama Cloud primary, Azure GPT-4o fallback)."""
        if self._primary is not None:
            # If primary is in cooldown, try fallback
            if hasattr(self._primary, "in_cooldown") and self._primary.in_cooldown():
                if self._fallback is not None:
                    self._record_fallback()
                    logger.info("Primary LLM in cooldown — using Ollama fallback")
                    return self._fallback
            return self._primary

        if self._fallback is not None:
            self._record_fallback()
            return self._fallback

        raise RuntimeError("No LLM backend configured")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        think: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text. Returns the answer string (backward compatible).

        Args:
            prompt: User prompt text.
            system: Optional system prompt.
            think: Enable Qwen3 thinking mode (``<think>`` tags).
            temperature: Sampling temperature (default from Config.LLM).
            max_tokens: Max generation tokens (default from Config.LLM).
            **kwargs: Forwarded to the underlying client.

        Returns:
            Generated answer text (thinking blocks stripped).
        """
        resp = self._do_generate(
            prompt, system=system, think=think,
            temperature=temperature, max_tokens=max_tokens, **kwargs,
        )
        return resp.text

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        system: str = "",
        think: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text and return ``(text, metadata_dict)``.

        Backward-compatible with the old gateway signature.
        """
        if options:
            kwargs.setdefault("options", options)
        resp = self._do_generate(
            prompt, system=system, think=think,
            temperature=temperature, max_tokens=max_tokens, **kwargs,
        )
        meta: Dict[str, Any] = {
            "usage": resp.usage,
            "backend": self._active_backend_name(),
        }
        if resp.thinking:
            meta["thinking"] = resp.thinking
        return resp.text, meta

    def chat_with_metadata(
        self,
        messages: List[Dict[str, str]],
        *,
        think: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Chat-based generation with system/user messages.

        When *think* is True and the primary client is vLLM, passes
        ``extra_body={"chat_template_kwargs": {"enable_thinking": True}}``
        so the server can activate Qwen3 thinking mode.
        """
        client = self._pick_client()

        temperature = temperature if temperature is not None else Config.LLM.TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else Config.LLM.MAX_TOKENS

        opts = dict(options or {})
        opts.setdefault("temperature", temperature)
        opts.setdefault("max_tokens", max_tokens)
        opts.setdefault("top_p", Config.LLM.TOP_P)

        self._record_request()

        try:
            raw, usage_meta = client.chat_with_metadata(
                messages, options=opts, thinking=think, **kwargs,
            )
        except Exception as primary_exc:
            self._record_failure(primary_exc)
            if self._fallback is not None and client is not self._fallback:
                logger.warning(
                    "Primary chat LLM (%s) failed: %s — retrying with fallback",
                    getattr(client, "backend", "unknown"), primary_exc,
                )
                self._record_fallback()
                # Fallback may not support chat; flatten to prompt
                if hasattr(self._fallback, "chat_with_metadata"):
                    raw, usage_meta = self._fallback.chat_with_metadata(
                        messages, options=opts, thinking=think, **kwargs,
                    )
                else:
                    flat = self._messages_to_prompt(messages)
                    raw, usage_meta = self._fallback.generate_with_metadata(
                        flat, options=opts, thinking=think, **kwargs,
                    )
            else:
                raise

        answer, thinking = _split_thinking(raw)

        # If thinking consumed all tokens and answer is empty, extract key points from thinking
        if not answer.strip() and thinking and len(thinking) > 50:
            logger.warning("LLM response empty (thinking consumed all tokens) — extracting from thinking block")
            # Try to find the last substantive paragraph in thinking
            lines = [l.strip() for l in thinking.split('\n') if l.strip()]
            # Look for draft/final/response sections in thinking
            for marker in ['Draft:', 'Final', 'Response:', 'Answer:', 'Summary:', 'Result:',
                           '## ', '**', 'Here is', 'The document', 'Based on']:
                for i, line in enumerate(lines):
                    if marker.lower() in line.lower():
                        answer = '\n'.join(lines[i:])
                        break
                if answer.strip():
                    break
            if not answer.strip():
                # Fallback: use last portion of thinking as the answer
                answer = '\n'.join(lines[-10:]) if len(lines) > 10 else '\n'.join(lines)

        meta: Dict[str, Any] = {
            "usage": usage_meta,
            "backend": self._active_backend_name(),
        }
        if thinking:
            meta["thinking"] = thinking

        return answer, meta

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "Generator[str, None, None]":
        """Stream tokens from the primary backend.

        Tries vLLM streaming first, then Ollama streaming, then falls back
        to yielding the full non-streaming response as a single chunk.
        """
        from typing import Generator  # noqa: F811

        temperature = temperature if temperature is not None else Config.LLM.TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else Config.LLM.MAX_TOKENS

        # Try vLLM streaming via the unified VLLMManager
        try:
            from src.api.rag_state import get_app_state
            app_state = get_app_state()
            vllm_mgr = getattr(app_state, "vllm_manager", None) if app_state else None
            if vllm_mgr is not None:
                try:
                    yield from vllm_mgr.stream_query(
                        prompt, system_prompt=system,
                        max_tokens=max_tokens, temperature=temperature,
                    )
                    return
                except Exception as stream_exc:
                    logger.warning("vLLM stream failed: %s — falling back to Ollama stream", stream_exc)
        except Exception as exc:
            logger.warning("vLLM streaming unavailable: %s — trying Ollama stream", exc)

        # Try Ollama streaming
        client = self._pick_client()
        if hasattr(client, "generate_stream"):
            try:
                full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
                yield from client.generate_stream(
                    full_prompt,
                    options={"temperature": temperature, "max_tokens": max_tokens},
                )
                return
            except Exception as exc:
                logger.warning("Ollama streaming failed: %s — falling back to non-streaming", exc)

        # Final fallback: full generation as single chunk
        yield self.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Classification helper
    # ------------------------------------------------------------------

    def classify(self, prompt: str, **kwargs: Any) -> str:
        """Convenience method for classification tasks (low temperature)."""
        return self.generate(prompt, temperature=0.05, max_tokens=256, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """Send a trivial request to warm up the backend."""
        try:
            self.generate("Say OK.", max_tokens=8)
            logger.info("LLM warm-up complete")
        except Exception as exc:
            logger.warning("LLM warm-up failed: %s", exc)

    def health_check(self) -> Dict[str, Any]:
        """Return a health summary dict."""
        return {
            "healthy": self._primary is not None,
            "primary": {
                "available": self._primary is not None,
                "backend": getattr(self._primary, "backend", None),
                "model": getattr(self._primary, "model_name", None),
            },
            "fallback": {
                "available": self._fallback is not None,
                "backend": getattr(self._fallback, "backend", None),
                "model": getattr(self._fallback, "model_name", None),
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return {
                **dict(self._stats),
                "uptime_seconds": round(time.time() - self._created_at, 1),
            }

    def in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    def shutdown(self) -> None:
        """Stop background threads."""
        if self._health_monitor:
            self._health_monitor.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_generate(
        self,
        prompt: str,
        *,
        system: str = "",
        think: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Unified generation logic with automatic fallback.

        If the primary client fails, automatically retries with the fallback
        client instead of propagating the error.
        """
        client = self._pick_client()

        temperature = temperature if temperature is not None else Config.LLM.TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else Config.LLM.MAX_TOKENS

        # Extract extra options to merge later (avoids duplicate kwarg for 'options')
        extra_options = kwargs.pop("options", None)

        self._record_request()

        opts = {"temperature": temperature, "max_tokens": max_tokens}
        if extra_options:
            opts.update(extra_options)

        # Suppress thinking for structured output (JSON) prompts on qwen3 models
        if not think and not system.startswith("/no_think"):
            json_markers = ("json only", "strict json", "valid json", "return json")
            combined = (system + " " + prompt).lower()
            if any(m in combined for m in json_markers):
                system = "/no_think\n" + system

        # Try vLLM first (fast local inference) before Ollama Cloud
        vllm_used = False
        try:
            from src.api.rag_state import get_app_state
            app_state = get_app_state()
            vllm_mgr = getattr(app_state, "vllm_manager", None) if app_state else None
            if vllm_mgr is not None and vllm_mgr.health_check():
                vllm_result = vllm_mgr.query(
                    prompt, system_prompt=system,
                    max_tokens=max_tokens, temperature=temperature,
                )
                if vllm_result:
                    raw = vllm_result
                    usage_meta = {"model": vllm_mgr._model, "backend": "vllm"}
                    vllm_used = True
        except Exception as vllm_exc:
            logger.debug("vLLM generate failed: %s — falling back to primary", vllm_exc)

        if not vllm_used:
            try:
                raw, usage_meta = self._call_client(
                    client, prompt, system=system, think=think,
                    opts=opts, **kwargs,
                )
            except Exception as primary_exc:
                self._record_failure(primary_exc)
                # If we have a fallback and didn't already use it, try fallback
                if self._fallback is not None and client is not self._fallback:
                    logger.warning(
                        "Primary LLM (%s) failed: %s — retrying with fallback (%s)",
                        getattr(client, "backend", "unknown"),
                        primary_exc,
                        getattr(self._fallback, "backend", "unknown"),
                    )
                    self._record_fallback()
                    try:
                        raw, usage_meta = self._call_client(
                            self._fallback, prompt, system=system, think=think,
                            opts=opts, **kwargs,
                        )
                    except Exception:
                        logger.exception("Fallback LLM also failed")
                        raise
                else:
                    raise

        answer, thinking = _split_thinking(raw)
        return LLMResponse(text=answer, thinking=thinking, usage=usage_meta)

    def _call_client(
        self,
        client: Any,
        prompt: str,
        *,
        system: str = "",
        think: bool = False,
        opts: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Dispatch a generation call to a specific client."""
        backend_name = getattr(client, "backend", "")
        if system and backend_name == "gemini":
            kwargs["system_instruction"] = system
            return client.generate_with_metadata(
                prompt, options=opts, thinking=think, **kwargs,
            )
        elif system and backend_name == "azure_openai" and hasattr(client, "chat_with_metadata"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            return client.chat_with_metadata(
                messages, options=opts, **kwargs,
            )
        else:
            full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
            return client.generate_with_metadata(
                full_prompt, options=opts, thinking=think, **kwargs,
            )

    def _vllm_chat(
        self,
        client,
        messages: List[Dict[str, str]],
        *,
        think: bool = False,
        opts: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict]:
        """Call vLLM via OpenAICompatibleClient, injecting thinking kwargs when needed."""
        call_kwargs: Dict[str, Any] = dict(kwargs)
        call_opts = dict(opts or {})

        if think:
            call_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True}
            }

        # Format messages into a single prompt for the client
        prompt = self._messages_to_prompt(messages)

        return client.generate_with_metadata(prompt, options=call_opts, **call_kwargs)

    @staticmethod
    def _build_messages(prompt: str, system: str = "") -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Flatten chat messages into a single prompt string for clients
        that only accept a prompt argument."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    def _active_backend_name(self) -> str:
        client = self._pick_client()
        return getattr(client, "backend", "unknown")

    def _record_request(self) -> None:
        with self._stats_lock:
            self._stats["requests"] += 1
            self._stats["last_request_ts"] = time.time()

    def _record_failure(self, exc: Exception) -> None:
        with self._stats_lock:
            self._stats["failures"] += 1
            self._stats["last_error"] = str(exc)

    def _record_fallback(self) -> None:
        with self._stats_lock:
            self._stats["fallback_used"] += 1


# ---------------------------------------------------------------------------
# Module-level public API
# ---------------------------------------------------------------------------

def create_llm_gateway(
    model_name: Optional[str] = None,
    backend_override: Optional[str] = None,
) -> LLMGateway:
    """Create (or recreate) the global LLMGateway singleton.

    Args are accepted for backward compatibility but ignored — the gateway
    reads its configuration from ``Config.VLLM`` and ``Config.LLM``.
    """
    global _gateway_instance
    with _gateway_lock:
        if _gateway_instance is not None:
            _gateway_instance.shutdown()
        _gateway_instance = LLMGateway()
        return _gateway_instance


def get_llm_gateway() -> LLMGateway:
    """Return the existing singleton, creating it on first call."""
    global _gateway_instance
    if _gateway_instance is None:
        with _gateway_lock:
            if _gateway_instance is None:
                _gateway_instance = LLMGateway()
    return _gateway_instance


def set_llm_gateway(gateway: LLMGateway) -> None:
    """Replace the global singleton (useful for testing)."""
    global _gateway_instance
    with _gateway_lock:
        _gateway_instance = gateway
