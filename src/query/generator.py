"""Phase 3 — Response generation via the 27B model."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports for serving layer
try:
    from src.serving.vllm_manager import VLLMManager
    from src.serving.model_router import IntentRouter
except ImportError:
    VLLMManager = None
    IntentRouter = None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GeneratedResponse:
    """Structured output from Phase 3 generation."""
    response_text: str
    chart_spec: Optional[Dict[str, Any]] = None
    alerts: Optional[List[Dict[str, Any]]] = None
    confidence: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    thinking: Optional[str] = None


# ---------------------------------------------------------------------------
# Regex patterns for parsing structured output
# ---------------------------------------------------------------------------

_RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL)
_CHART_RE = re.compile(r"<chart_spec>(.*?)</chart_spec>", re.DOTALL)
_ALERTS_RE = re.compile(r"<alerts>(.*?)</alerts>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# System prompt for response generation
# ---------------------------------------------------------------------------

_GENERATION_SYSTEM_PROMPT = """\
You are DocWain, a senior document intelligence expert. You analyze documents and \
produce accurate, well-structured responses.

RULES:
1. Lead with the answer. No preamble.
2. Every factual claim MUST be grounded in the provided evidence. If a value is not \
in the evidence, say "not specified in the documents".
3. Synthesize across sources. Connect facts and explain why something matters.
4. Match response depth to query complexity.

OUTPUT FORMAT:
Wrap your main answer in <response>...</response> tags.

If a chart would help the user understand the data, include a <chart_spec>...</chart_spec> \
block with valid JSON describing the chart:
{
  "type": "bar|line|pie|table",
  "title": "Chart title",
  "data": [{"label": "...", "value": ...}, ...],
  "x_label": "...",
  "y_label": "..."
}

If you detect important alerts (deadlines, risks, anomalies, missing data), include \
an <alerts>...</alerts> block with valid JSON:
[
  {"level": "warning|info|critical", "message": "..."}
]

If you need to think through complex reasoning, use <think>...</think> before <response>.
"""


# ---------------------------------------------------------------------------
# ResponseGenerator
# ---------------------------------------------------------------------------

class ResponseGenerator:
    """Phase 3: Generate the final response using 27B with structured context."""

    def __init__(self, vllm_manager: Any = None, llm_gateway: Any = None):
        """Initialize the generator.

        Args:
            vllm_manager: VLLMManager for local 27B inference.
            llm_gateway: Fallback LLMGateway (Ollama Cloud / Azure).
        """
        self._vllm = vllm_manager
        self._gateway = llm_gateway

    def generate(
        self,
        query: str,
        context: str,
        router_result: Optional[Any] = None,
    ) -> GeneratedResponse:
        """Generate a response for the user query given assembled context.

        Args:
            query: The user's original query.
            context: Multi-block context string from context_assembler.
            router_result: Router result with intent / suggested_format hints.

        Returns:
            GeneratedResponse with parsed fields.
        """
        user_prompt = self._build_prompt(query, context, router_result)
        raw_output, thinking = self._call_llm(user_prompt)

        if not raw_output:
            return GeneratedResponse(
                response_text="I was unable to generate a response. Please try again.",
                confidence=0.0,
                thinking=thinking,
            )

        return self._parse_output(raw_output, thinking)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        context: str,
        router_result: Optional[Any],
    ) -> str:
        parts: List[str] = []

        # Context comes first so the model sees evidence before the question
        if context:
            parts.append(context)

        parts.append(f"User query: {query}")

        if router_result is not None:
            intent = getattr(router_result, "intent", None)
            fmt = getattr(router_result, "suggested_format", None)
            if intent:
                parts.append(f"Detected intent: {intent}")
            if fmt:
                parts.append(f"Suggested format: {fmt}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM call with fallback chain
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> tuple[str, Optional[str]]:
        """Call unified vLLM (preferred) or gateway (fallback). Returns (text, thinking)."""
        if self._vllm is not None:
            try:
                raw = self._vllm.generate(
                    prompt=user_prompt,
                    system=_GENERATION_SYSTEM_PROMPT,
                    temperature=0.4,
                    max_tokens=4096,
                )
                text, thinking = self._split_thinking(raw)
                return text, thinking
            except Exception as exc:
                logger.warning("vLLM generation failed, trying gateway fallback: %s", exc)

        # Fallback to LLMGateway
        if self._gateway is not None:
            try:
                result_text, meta = self._gateway.generate_with_metadata(
                    user_prompt,
                    system=_GENERATION_SYSTEM_PROMPT,
                    think=True,
                    max_tokens=4096,
                )
                thinking = meta.get("thinking")
                return result_text, thinking
            except Exception as exc:
                logger.error("Gateway generation also failed: %s", exc)

        logger.error("No LLM backend available for generation")
        return "", None

    @staticmethod
    def _split_thinking(raw: str) -> tuple[str, Optional[str]]:
        """Extract <think>...</think> blocks from model output."""
        match = _THINK_RE.search(raw)
        if match:
            thinking = match.group(1).strip()
            answer = _THINK_RE.sub("", raw).strip()
            return answer, thinking or None
        # Handle unclosed <think> tag
        if "<think>" in raw:
            idx = raw.index("<think>")
            before = raw[:idx].strip()
            after = raw[idx + len("<think>"):].strip()
            if before:
                return before, after or None
            return "", after or None
        return raw.strip(), None

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(self, raw: str, thinking: Optional[str]) -> GeneratedResponse:
        """Parse structured output blocks from the model's response."""
        # Extract <response> block
        response_match = _RESPONSE_RE.search(raw)
        if response_match:
            response_text = response_match.group(1).strip()
        else:
            # Fallback: use the entire output as the response
            # Strip out any chart_spec/alerts blocks first
            response_text = _CHART_RE.sub("", raw)
            response_text = _ALERTS_RE.sub("", response_text).strip()

        # Extract <chart_spec> block
        chart_spec = None
        chart_match = _CHART_RE.search(raw)
        if chart_match:
            chart_spec = self._safe_parse_json(chart_match.group(1).strip(), "chart_spec")

        # Extract <alerts> block
        alerts = None
        alerts_match = _ALERTS_RE.search(raw)
        if alerts_match:
            alerts = self._safe_parse_json(alerts_match.group(1).strip(), "alerts")
            # Ensure alerts is a list
            if isinstance(alerts, dict):
                alerts = [alerts]
            elif not isinstance(alerts, list):
                alerts = None

        # Derive a basic confidence from response quality signals
        confidence = self._estimate_confidence(response_text, chart_spec, alerts)

        return GeneratedResponse(
            response_text=response_text,
            chart_spec=chart_spec,
            alerts=alerts,
            confidence=confidence,
            sources=[],
            thinking=thinking,
        )

    @staticmethod
    def _safe_parse_json(text: str, label: str) -> Any:
        """Attempt to parse JSON; return None on failure."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to parse %s JSON: %s", label, exc)
            return None

    @staticmethod
    def _estimate_confidence(
        response_text: str,
        chart_spec: Optional[Dict[str, Any]],
        alerts: Optional[List[Dict[str, Any]]],
    ) -> float:
        """Heuristic confidence estimate based on response characteristics."""
        if not response_text:
            return 0.0

        score = 0.5  # baseline

        word_count = len(response_text.split())
        if word_count > 30:
            score += 0.1
        if word_count > 100:
            score += 0.1

        # Penalise hedging language
        hedges = ["not specified", "not found", "unable to determine", "no information"]
        hedge_count = sum(1 for h in hedges if h.lower() in response_text.lower())
        score -= hedge_count * 0.1

        # Bonus for structured output
        if chart_spec:
            score += 0.05
        if alerts:
            score += 0.05

        return max(0.0, min(1.0, round(score, 2)))


__all__ = ["GeneratedResponse", "ResponseGenerator"]
