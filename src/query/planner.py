"""Phase 1 — Query Planning via 27B model with guided JSON output."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports for serving layer (may not exist yet)
try:
    from src.serving.vllm_manager import VLLMManager
    from src.serving.model_router import IntentRouter
except ImportError:
    VLLMManager = None
    IntentRouter = None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in a query execution plan."""
    id: str
    action: str  # search | knowledge_search | kg_lookup | kg_search | cross_reference | spreadsheet_query | generate
    query: str
    collection: str = ""
    top_k: int = 10
    depends_on: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPlan:
    """Complete execution plan produced by Phase 1."""
    intent: str
    complexity: str  # simple | moderate | complex
    steps: List[PlanStep] = field(default_factory=list)
    requires_kg: bool = False
    requires_visualization: bool = False
    domain_pack: str = "generic"
    region: str = ""


# ---------------------------------------------------------------------------
# JSON schema for guided decoding
# ---------------------------------------------------------------------------

_PLAN_JSON_SCHEMA = {
    "type": "object",
    "required": ["intent", "complexity", "steps"],
    "properties": {
        "intent": {"type": "string"},
        "complexity": {"type": "string", "enum": ["simple", "moderate", "complex"]},
        "requires_kg": {"type": "boolean"},
        "requires_visualization": {"type": "boolean"},
        "domain_pack": {"type": "string"},
        "region": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "action", "query"],
                "properties": {
                    "id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": [
                            "search",
                            "knowledge_search",
                            "kg_lookup",
                            "kg_search",
                            "cross_reference",
                            "spreadsheet_query",
                            "generate",
                        ],
                    },
                    "query": {"type": "string"},
                    "collection": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "params": {"type": "object"},
                },
            },
        },
    },
}

_PLANNING_SYSTEM_PROMPT = (
    "You are a query planner for a document intelligence system. "
    "Given a user query and profile context, produce an execution plan as JSON.\n\n"
    "Available actions:\n"
    "- search: vector search on user document collection\n"
    "- knowledge_search: vector search on domain knowledge pack collection\n"
    "- kg_lookup: Neo4j Cypher query for entity/relationship lookup\n"
    "- kg_search: broader KG traversal (1-2 hops)\n"
    "- cross_reference: compare results from two dependent steps to find agreements/conflicts\n"
    "- spreadsheet_query: lookup structured data from MongoDB computed profiles\n"
    "- generate: final response generation (should be the last step, depends on all retrieval steps)\n\n"
    "Rules:\n"
    "1. Every plan MUST end with a 'generate' step.\n"
    "2. The generate step must depend on all retrieval steps.\n"
    "3. Independent retrieval steps can run in parallel (no depends_on between them).\n"
    "4. cross_reference must depend on exactly 2 prior steps.\n"
    "5. Use knowledge_search only when domain expertise is needed beyond user documents.\n"
    "6. Use kg_lookup/kg_search only when entity relationships matter.\n"
    "7. For simple factual queries, a single search + generate is sufficient.\n"
    "8. Set complexity: simple (1-2 steps), moderate (3-4 steps), complex (5+ steps).\n"
    "9. Set requires_visualization: true if the answer benefits from a chart.\n"
    "10. Set domain_pack to the relevant domain (e.g., 'medical', 'legal', 'finance', 'generic').\n"
)


# ---------------------------------------------------------------------------
# QueryPlanner
# ---------------------------------------------------------------------------

class QueryPlanner:
    """Phase 1: Use the 27B model to plan query execution."""

    def __init__(self, vllm_manager: Any = None):
        self._vllm = vllm_manager

    def plan(
        self,
        query: str,
        profile_context: Optional[Dict[str, Any]] = None,
        router_result: Optional[Any] = None,
    ) -> QueryPlan:
        """Produce a QueryPlan for the given query.

        Args:
            query: The user's natural language query.
            profile_context: Pre-computed profile intelligence dict
                (domain, doc_count, entity_summary, etc.).
            router_result: Result from the 14B intent router (intent, confidence, etc.).

        Returns:
            A QueryPlan with ordered steps.
        """
        profile_context = profile_context or {}
        plan_dict = self._plan_via_27b(query, profile_context, router_result)
        if plan_dict is None:
            logger.info("27B planner unavailable; using default plan")
            plan_dict = self._default_plan(query, profile_context, router_result)
        return self._dict_to_plan(plan_dict)

    # ------------------------------------------------------------------
    # 27B planning
    # ------------------------------------------------------------------

    def _plan_via_27b(
        self,
        query: str,
        profile_context: Dict[str, Any],
        router_result: Optional[Any],
    ) -> Optional[Dict[str, Any]]:
        """Call the 27B model with guided_json to produce a plan."""
        if self._vllm is None:
            return None

        user_prompt = self._build_user_prompt(query, profile_context, router_result)

        try:
            raw = self._vllm.generate(
                prompt=user_prompt,
                system=_PLANNING_SYSTEM_PROMPT,
                model="smart",
                guided_json=_PLAN_JSON_SCHEMA,
                temperature=0.3,
                max_tokens=2048,
            )
            parsed = json.loads(raw)
            # Validate minimal structure
            if "steps" not in parsed or not isinstance(parsed["steps"], list):
                logger.warning("27B plan missing steps, falling back")
                return None
            return parsed
        except json.JSONDecodeError as exc:
            logger.warning("27B plan JSON parse error: %s", exc)
            return None
        except Exception as exc:
            logger.warning("27B planner call failed: %s", exc)
            return None

    def _build_user_prompt(
        self,
        query: str,
        profile_context: Dict[str, Any],
        router_result: Optional[Any],
    ) -> str:
        parts = [f"User query: {query}"]

        if profile_context:
            ctx_lines = []
            if profile_context.get("primary_domain"):
                ctx_lines.append(f"Domain: {profile_context['primary_domain']}")
            if profile_context.get("doc_count"):
                ctx_lines.append(f"Documents in profile: {profile_context['doc_count']}")
            if profile_context.get("entity_summary"):
                ctx_lines.append(f"Key entities: {profile_context['entity_summary']}")
            if profile_context.get("has_spreadsheets"):
                ctx_lines.append("Profile contains spreadsheet data.")
            if profile_context.get("has_kg"):
                ctx_lines.append("Knowledge graph is available.")
            if ctx_lines:
                parts.append("Profile context:\n" + "\n".join(ctx_lines))

        if router_result is not None:
            intent = getattr(router_result, "intent", None) or ""
            confidence = getattr(router_result, "confidence", None)
            if intent:
                parts.append(f"Detected intent: {intent}")
            if confidence is not None:
                parts.append(f"Intent confidence: {confidence:.2f}")

        parts.append("Produce the execution plan JSON.")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Default fallback plan
    # ------------------------------------------------------------------

    def _default_plan(
        self,
        query: str,
        profile_context: Dict[str, Any],
        router_result: Optional[Any],
    ) -> Dict[str, Any]:
        """Simple two-step plan: search user_docs then generate."""
        intent = "factual"
        if router_result is not None:
            intent = getattr(router_result, "intent", "factual") or "factual"

        steps = [
            {
                "id": "s1",
                "action": "search",
                "query": query,
                "collection": "user_docs",
                "top_k": 15,
                "depends_on": [],
                "params": {},
            },
            {
                "id": "g1",
                "action": "generate",
                "query": query,
                "collection": "",
                "top_k": 0,
                "depends_on": ["s1"],
                "params": {},
            },
        ]

        domain = profile_context.get("primary_domain", "generic")

        return {
            "intent": intent,
            "complexity": "simple",
            "steps": steps,
            "requires_kg": False,
            "requires_visualization": False,
            "domain_pack": domain,
            "region": "",
        }

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_plan(d: Dict[str, Any]) -> QueryPlan:
        """Convert a raw dict into a typed QueryPlan."""
        steps = []
        for s in d.get("steps", []):
            steps.append(
                PlanStep(
                    id=s.get("id", ""),
                    action=s.get("action", "search"),
                    query=s.get("query", ""),
                    collection=s.get("collection", ""),
                    top_k=int(s.get("top_k", 10)),
                    depends_on=s.get("depends_on", []),
                    params=s.get("params", {}),
                )
            )
        return QueryPlan(
            intent=d.get("intent", "factual"),
            complexity=d.get("complexity", "simple"),
            steps=steps,
            requires_kg=bool(d.get("requires_kg", False)),
            requires_visualization=bool(d.get("requires_visualization", False)),
            domain_pack=d.get("domain_pack", "generic"),
            region=d.get("region", ""),
        )


__all__ = ["PlanStep", "QueryPlan", "QueryPlanner"]
