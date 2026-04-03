"""LLM-based intent routing for the dual-model serving layer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# -- All recognised intent types ----------------------------------------------

INTENT_TYPES: List[str] = [
    "greeting",
    "identity",
    "lookup",
    "list",
    "count",
    "summarize",
    "compare",
    "analyze",
    "investigate",
    "extract",
    "generate",
    "aggregate",
    "rank",
    "timeline",
    "visualize",
]

_FAST_INTENTS = frozenset({
    "greeting", "identity", "lookup", "list", "count", "extract",
})

_SMART_INTENTS = frozenset({
    "compare", "analyze", "investigate", "generate", "aggregate",
    "rank", "timeline", "visualize", "summarize",
})

# -- Data models --------------------------------------------------------------

@dataclass
class RouterResult:
    """Result of intent classification and routing."""

    intent: str
    complexity: str  # "simple" | "moderate" | "complex"
    route: str       # "fast" | "smart"
    requires_kg: bool = False
    requires_visualization: bool = False


# -- Guided JSON schema for the classification prompt -------------------------

_ROUTER_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": INTENT_TYPES,
        },
        "complexity": {
            "type": "string",
            "enum": ["simple", "moderate", "complex"],
        },
        "requires_kg": {"type": "boolean"},
        "requires_visualization": {"type": "boolean"},
    },
    "required": ["intent", "complexity", "requires_kg", "requires_visualization"],
    "additionalProperties": False,
}

_CLASSIFICATION_SYSTEM = (
    "You are a query intent classifier for a document intelligence system.\n"
    "Classify the user query into exactly one intent and assess its complexity.\n\n"
    "Intent types:\n"
    "- greeting: social greetings, hellos\n"
    "- identity: questions about the assistant itself\n"
    "- lookup: find a specific fact, value, or entity\n"
    "- list: enumerate items matching criteria\n"
    "- count: how many of something\n"
    "- summarize: condense information\n"
    "- compare: contrast two or more items\n"
    "- analyze: in-depth analysis or reasoning\n"
    "- investigate: multi-step research across documents\n"
    "- extract: pull structured data from documents\n"
    "- generate: create new content based on documents\n"
    "- aggregate: combine data across multiple sources\n"
    "- rank: order items by criteria\n"
    "- timeline: temporal ordering of events\n"
    "- visualize: request for charts, graphs, diagrams\n\n"
    "Complexity levels:\n"
    "- simple: single fact retrieval, greeting, identity\n"
    "- moderate: requires some reasoning or filtering\n"
    "- complex: multi-step reasoning, cross-document analysis, generation\n\n"
    "Set requires_kg to true if the query would benefit from knowledge graph relationships.\n"
    "Set requires_visualization to true if the answer should include a chart or diagram.\n\n"
    "Respond with valid JSON only."
)


# -- Keyword-based fallback heuristic ----------------------------------------

_KEYWORD_RULES: List[tuple[list[str], str]] = [
    (["hello", "hi ", "hey", "good morning", "good afternoon", "good evening"], "greeting"),
    (["who are you", "what are you", "your name", "about yourself"], "identity"),
    (["how many", "count", "number of", "total"], "count"),
    (["list ", "enumerate", "show all", "all the", "give me all"], "list"),
    (["compare", "versus", " vs ", "difference between", "contrast"], "compare"),
    (["summarize", "summary", "summarise", "overview", "brief"], "summarize"),
    (["analyze", "analyse", "analysis", "investigate", "deep dive"], "analyze"),
    (["timeline", "chronolog", "sequence of events", "time order"], "timeline"),
    (["chart", "graph", "plot", "visualize", "visualise", "diagram"], "visualize"),
    (["rank", "top ", "best ", "worst ", "highest", "lowest", "order by"], "rank"),
    (["aggregate", "combine", "across all", "total across"], "aggregate"),
    (["extract", "pull out", "get the", "find the"], "extract"),
    (["generate", "create", "write", "draft", "compose"], "generate"),
]


def _keyword_classify(query: str) -> RouterResult:
    """Classify a query using simple keyword matching (fallback)."""
    q = query.lower().strip()

    intent = "lookup"  # default
    for keywords, label in _KEYWORD_RULES:
        if any(kw in q for kw in keywords):
            intent = label
            break

    if intent in _FAST_INTENTS:
        route = "fast"
        complexity = "simple"
    elif intent in ("summarize", "extract"):
        route = "smart"
        complexity = "moderate"
    else:
        route = "smart"
        complexity = "complex"

    requires_viz = intent == "visualize"
    requires_kg = intent in ("investigate", "compare", "timeline", "aggregate")

    return RouterResult(
        intent=intent,
        complexity=complexity,
        route=route,
        requires_kg=requires_kg,
        requires_visualization=requires_viz,
    )


# -- IntentRouter class -------------------------------------------------------

class IntentRouter:
    """Routes queries to the fast or smart model path based on intent classification.

    Uses the 14B model (via ``VLLMManager.query_fast``) with guided JSON output
    for reliable classification. Falls back to keyword heuristics if the model
    is unavailable.

    Usage::

        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager()
        router = IntentRouter(mgr)
        result = router.route("How many invoices are overdue?")
        # result.route == "fast", result.intent == "count"
    """

    def __init__(self, vllm_manager: Any) -> None:
        self._mgr = vllm_manager

    def route(self, query: str, profile_context: Optional[str] = None) -> RouterResult:
        """Classify a query and decide the serving route.

        Args:
            query: The user's natural-language question.
            profile_context: Optional domain context to improve classification.

        Returns:
            A ``RouterResult`` with intent, complexity, and route.
        """
        user_prompt = f"Classify this query:\n\n{query}"
        if profile_context:
            user_prompt += f"\n\nProfile context: {profile_context}"

        try:
            raw = self._mgr.query_fast(
                prompt=user_prompt,
                system_prompt=_CLASSIFICATION_SYSTEM,
                guided_json=_ROUTER_JSON_SCHEMA,
                max_tokens=256,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("LLM-based routing failed (%s) — using keyword fallback", exc)
            return _keyword_classify(query)

        if not raw:
            logger.info("Empty response from router LLM — using keyword fallback")
            return _keyword_classify(query)

        return self._parse_response(raw, query)

    # -- Private helpers ------------------------------------------------------

    def _parse_response(self, raw: str, query: str) -> RouterResult:
        """Parse the JSON classification response from the LLM."""
        # Strip markdown fences if present.
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse router JSON: %s — using keyword fallback", raw[:200])
            return _keyword_classify(query)

        intent = data.get("intent", "lookup")
        if intent not in INTENT_TYPES:
            logger.warning("Unknown intent '%s' from router — defaulting to lookup", intent)
            intent = "lookup"

        complexity = data.get("complexity", "moderate")
        if complexity not in ("simple", "moderate", "complex"):
            complexity = "moderate"

        # Determine route from intent, overriding if complexity demands it.
        if intent in _FAST_INTENTS and complexity != "complex":
            route = "fast"
        else:
            route = "smart"

        requires_kg = bool(data.get("requires_kg", False))
        requires_viz = bool(data.get("requires_visualization", False))

        result = RouterResult(
            intent=intent,
            complexity=complexity,
            route=route,
            requires_kg=requires_kg,
            requires_visualization=requires_viz,
        )
        logger.info(
            "Routed query — intent=%s complexity=%s route=%s kg=%s viz=%s",
            result.intent, result.complexity, result.route,
            result.requires_kg, result.requires_visualization,
        )
        return result
