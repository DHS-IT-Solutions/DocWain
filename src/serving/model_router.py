"""LLM-based intent classification for the DocWain serving layer.

Historical context: this module used to route between a 14B "fast" and 27B
"smart" vLLM instance. DocWain is now a single unified model; the classifier
survives because intent still drives retrieval strategy, KG expansion, and
visualization choices — not model selection.

Phase 4 additions (ERRATA §9): a new async ``classify_query`` returns a
frozen :class:`ClassifiedQuery` carrying the canonical ``query_text``,
``intent`` label (extended with ``analyze``/``diagnose``/``recommend``),
a ``format_hint`` enum (``auto`` | ``compact`` | ``rich``), detected URLs,
and entities. The classifier reuses the existing understand-stage LLM call;
no new model calls are introduced. URL detection and the compact-override
heuristic are deterministic regex over the raw input so they never depend
on the model's JSON being shaped correctly.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass
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
    "overview",
    # Phase 4 additions — analytical intents that route to rich-mode templates.
    "diagnose",
    "recommend",
]

_SIMPLE_INTENTS = frozenset({
    "greeting", "identity", "lookup", "list", "count", "extract",
})


@dataclass
class RouterResult:
    """Result of intent classification."""

    intent: str
    complexity: str  # "simple" | "moderate" | "complex"
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

    intent = "lookup"
    for keywords, label in _KEYWORD_RULES:
        if any(kw in q for kw in keywords):
            intent = label
            break

    if intent in _SIMPLE_INTENTS:
        complexity = "simple"
    elif intent in ("summarize", "extract"):
        complexity = "moderate"
    else:
        complexity = "complex"

    requires_viz = intent == "visualize"
    requires_kg = intent in ("investigate", "compare", "timeline", "aggregate")

    return RouterResult(
        intent=intent,
        complexity=complexity,
        requires_kg=requires_kg,
        requires_visualization=requires_viz,
    )


class IntentRouter:
    """Classifies queries by intent for downstream retrieval/generation logic.

    Uses the unified DocWain vLLM instance with guided JSON output. Falls back
    to keyword heuristics if the model is unavailable.

    Usage::

        from src.serving.vllm_manager import VLLMManager
        mgr = VLLMManager()
        router = IntentRouter(mgr)
        result = router.route("How many invoices are overdue?")
        # result.intent == "count"
    """

    def __init__(self, vllm_manager: Any) -> None:
        self._mgr = vllm_manager

    def route(self, query: str, profile_context: Optional[str] = None) -> RouterResult:
        """Classify a query into an intent and complexity."""
        user_prompt = f"Classify this query:\n\n{query}"
        if profile_context:
            user_prompt += f"\n\nProfile context: {profile_context}"

        try:
            raw = self._mgr.query(
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

    def _parse_response(self, raw: str, query: str) -> RouterResult:
        """Parse the JSON classification response from the LLM."""
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

        requires_kg = bool(data.get("requires_kg", False))
        requires_viz = bool(data.get("requires_visualization", False))

        result = RouterResult(
            intent=intent,
            complexity=complexity,
            requires_kg=requires_kg,
            requires_visualization=requires_viz,
        )
        logger.info(
            "Classified query — intent=%s complexity=%s kg=%s viz=%s",
            result.intent, result.complexity,
            result.requires_kg, result.requires_visualization,
        )
        return result


# ---------------------------------------------------------------------------
# Phase 4 — ClassifiedQuery + async classify_query (ERRATA §9)
# ---------------------------------------------------------------------------
#
# The Phase 4 consumer (core_agent, shape resolution) expects a single frozen
# dataclass carrying the raw query plus the extracted intent / format_hint /
# urls / entities. Earlier phases passed these around as loose kwargs; the
# frozen shape eliminates hasattr/getattr fallbacks downstream.

# Valid intents for the Phase 4 classifier. Ordered so the LLM sees the
# long-standing labels first, then the Phase 4 additions.
VALID_INTENTS: tuple[str, ...] = (
    "greeting", "identity", "lookup", "list", "count",
    "summarize", "compare", "overview", "investigate",
    "extract", "aggregate",
    "analyze", "diagnose", "recommend",
)

# Compact override markers — if the user typed any of these the classifier
# escalates FormatHint.AUTO to FormatHint.COMPACT regardless of LLM output.
_COMPACT_OVERRIDE_MARKERS: tuple[str, ...] = (
    "tl;dr", "tldr", "one paragraph", "one line", "one-line",
    "keep it short", "short answer", "in brief", "just the answer",
    "no report", "bullet", "short",
)

_URL_RE = re.compile(r"https?://[^\s\"')>]+", re.IGNORECASE)


class FormatHint(str, enum.Enum):
    """User-visible response shape hint extracted from the query.

    ``AUTO`` is the default — the shape resolver decides based on intent,
    pack quality, and the rich-mode flag. ``COMPACT`` and ``RICH`` are
    explicit overrides (e.g. "tl;dr please" forces compact).
    """

    AUTO = "auto"
    COMPACT = "compact"
    RICH = "rich"


@dataclass(frozen=True)
class ClassifiedQuery:
    """Canonical classifier output (ERRATA §9).

    ``query_text`` is always populated from the classifier input so downstream
    builders (Phase 4 prompt assembly, Phase 5 URL-as-prompt) read it
    directly without hasattr/getattr fallbacks. ``urls`` is a deterministic
    regex extraction independent of the LLM's JSON shape.
    """

    query_text: str
    intent: str
    format_hint: FormatHint
    entities: List[str]
    urls: List[str]


async def _call_classifier_llm(prompt: str) -> str:
    """Placeholder LLM seam — overridden by patched tests.

    The real implementation lives in :class:`IntentRouter` (above) and is
    threaded through the serving layer. Tests patch this function directly
    so the async ``classify_query`` surface can be exercised without a
    vLLM dependency.
    """
    raise NotImplementedError(
        "classify_query is designed to be patched in tests; "
        "production serving routes through IntentRouter.route()"
    )


def _build_classifier_prompt(query_text: str) -> str:
    """Build the guided-JSON classification prompt for Phase 4 labels.

    Kept short — no persona, no SME instructions. This LLM call is pure
    classification; rich-mode formatting lives in generation/prompts.py.
    """
    return (
        "Classify the following user query. Respond ONLY with JSON.\n"
        f'Query: "{query_text}"\n\n'
        "Schema:\n"
        '{\n'
        '  "intent": one of '
        + ", ".join(f'"{i}"' for i in VALID_INTENTS)
        + ",\n"
        '  "format_hint": one of "auto" | "compact" | "rich",\n'
        '  "entities": [string, ...],\n'
        '  "urls": [string, ...]\n'
        "}"
    )


def _safe_parse(raw: str) -> Dict[str, Any]:
    """Parse the classifier LLM output into a dict, falling back to {}."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}


def _looks_like_compact_override(q: str) -> bool:
    """Deterministic compact-override detection against the raw input."""
    lowered = q.lower()
    return any(marker in lowered for marker in _COMPACT_OVERRIDE_MARKERS)


async def classify_query(query_text: str) -> ClassifiedQuery:
    """Async intent classifier for the Phase 4 rich-mode path.

    Preserves the legacy understand-stage contract (one LLM call per query)
    but widens the output shape to the ERRATA §9 ``ClassifiedQuery``. When
    the LLM returns malformed JSON or an unknown intent label we fall back
    to ``overview`` + ``FormatHint.AUTO`` — safer than guessing a richer
    intent that could waste pack tokens.
    """
    try:
        raw = await _call_classifier_llm(_build_classifier_prompt(query_text))
    except NotImplementedError:
        raw = ""
    parsed = _safe_parse(raw)

    intent = parsed.get("intent")
    if intent not in VALID_INTENTS:
        intent = "overview"

    hint_raw = parsed.get("format_hint", "auto")
    try:
        hint = FormatHint(hint_raw)
    except ValueError:
        hint = FormatHint.AUTO

    # Deterministic post-parse: a clear compact-override phrase in the raw
    # query always wins, even if the LLM returned AUTO.
    if hint is FormatHint.AUTO and _looks_like_compact_override(query_text):
        hint = FormatHint.COMPACT

    # URL detection is regex-based — independent of LLM output.
    urls = _URL_RE.findall(query_text)

    entities_raw = parsed.get("entities", [])
    entities = [e for e in entities_raw if isinstance(e, str)]

    return ClassifiedQuery(
        query_text=query_text,
        intent=intent,
        format_hint=hint,
        entities=entities,
        urls=urls,
    )
