"""Main query pipeline orchestrator — unified-model path: route, plan, execute, generate, verify."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.query.planner import QueryPlan, QueryPlanner
from src.query.executor import ExecutionClients, ExecutionResult, PlanExecutor
from src.query.context_assembler import assemble_context
from src.query.generator import GeneratedResponse, ResponseGenerator
from src.query.confidence import VerificationResult, verify_response

logger = logging.getLogger(__name__)

# Lazy imports for serving layer (may not be built yet)
try:
    from src.serving.vllm_manager import VLLMManager
    from src.serving.model_router import IntentRouter, RouterResult
except ImportError:
    VLLMManager = None
    IntentRouter = None
    RouterResult = None

_MAX_RE_RETRIEVAL = 2
_MIN_CONFIDENCE = 0.7


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class QueryPipelineResult:
    """Final output of the query pipeline."""
    response: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    chart_spec: Optional[Dict[str, Any]] = None
    alerts: Optional[List[Dict[str, Any]]] = None
    grounded: bool = False
    context_found: bool = False
    confidence: float = 0.0
    route_taken: str = "intelligence"
    plan: Optional[QueryPlan] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_query_pipeline(
    query: str,
    profile_id: str,
    subscription_id: str,
    clients: Dict[str, Any],
    profile_intelligence: Optional[Dict[str, Any]] = None,
) -> QueryPipelineResult:
    """Run the full query pipeline.

    Flow:
        1. Route query via unified-model intent router.
        2. Plan -> Execute -> Assemble -> Generate+Verify.
        3. Re-retrieval loop if confidence < 0.7 (max 2 iterations).
        4. Return QueryPipelineResult.

    Args:
        query: User's natural language query.
        profile_id: Profile scope.
        subscription_id: Subscription / collection scope.
        clients: Dict with keys: qdrant_client, neo4j_driver, mongo_db,
                 embedder, vllm_manager, llm_gateway (any may be None).
        profile_intelligence: Pre-computed profile intelligence dict.

    Returns:
        QueryPipelineResult with response, sources, chart, alerts, confidence.
    """
    t0 = time.monotonic()
    profile_intelligence = profile_intelligence or {}

    # Unpack client references
    vllm_manager = clients.get("vllm_manager")
    llm_gateway = clients.get("llm_gateway")
    qdrant_client = clients.get("qdrant_client")
    neo4j_driver = clients.get("neo4j_driver")
    mongo_db = clients.get("mongo_db")
    embedder = clients.get("embedder")

    # -----------------------------------------------------------------
    # Step 1: Route the query
    # -----------------------------------------------------------------
    router_result = _route_query(query, vllm_manager)
    route_intent = _get_intent(router_result)

    # -----------------------------------------------------------------
    # Step 2: Plan -> Execute -> Assemble -> Generate
    # -----------------------------------------------------------------
    exec_clients = ExecutionClients(
        qdrant_client=qdrant_client,
        neo4j_driver=neo4j_driver,
        mongo_db=mongo_db,
        embedder=embedder,
        profile_id=profile_id,
        subscription_id=subscription_id,
    )

    planner = QueryPlanner(vllm_manager=vllm_manager)
    generator = ResponseGenerator(vllm_manager=vllm_manager, llm_gateway=llm_gateway)
    executor = PlanExecutor()

    plan = planner.plan(query, profile_intelligence, router_result)
    execution = executor.execute(plan, exec_clients)
    context = assemble_context(execution, profile_intelligence, plan)
    gen_result = generator.generate(query, context, router_result)

    # -----------------------------------------------------------------
    # Step 3: Verify and re-retrieve if needed
    # -----------------------------------------------------------------
    verification = verify_response(gen_result.response_text, context, query)
    loop_count = 0

    while not verification.passed and loop_count < _MAX_RE_RETRIEVAL:
        loop_count += 1
        refined_query = verification.refined_query or query
        logger.info(
            "Re-retrieval loop %d: confidence=%.2f, refined_query=%s",
            loop_count,
            verification.confidence,
            refined_query[:80],
        )

        # Re-plan with refined query
        plan = planner.plan(refined_query, profile_intelligence, router_result)
        execution = executor.execute(plan, exec_clients)
        context = assemble_context(execution, profile_intelligence, plan)
        gen_result = generator.generate(query, context, router_result)
        verification = verify_response(gen_result.response_text, context, query)

    # -----------------------------------------------------------------
    # Step 4: Collect sources from execution results
    # -----------------------------------------------------------------
    sources = _collect_sources(execution)
    context_found = bool(sources)
    grounded = context_found and verification.confidence >= 0.5

    duration = round(time.monotonic() - t0, 3)

    return QueryPipelineResult(
        response=gen_result.response_text,
        sources=sources,
        chart_spec=gen_result.chart_spec,
        alerts=gen_result.alerts,
        grounded=grounded,
        context_found=context_found,
        confidence=verification.confidence,
        route_taken="intelligence",
        plan=plan,
        metadata={
            "intent": route_intent,
            "complexity": plan.complexity,
            "domain_pack": plan.domain_pack,
            "execution_success": execution.success,
            "execution_duration": execution.duration_seconds,
            "re_retrieval_loops": loop_count,
            "verification_reasons": verification.reasons,
            "duration_seconds": duration,
            "thinking": gen_result.thinking,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _route_query(query: str, vllm_manager: Any) -> Any:
    """Classify the query via the unified-model intent router."""
    if IntentRouter is None or vllm_manager is None:
        return None
    try:
        router = IntentRouter(vllm_manager=vllm_manager)
        return router.route(query)
    except Exception as exc:
        logger.warning("Intent router failed: %s", exc)
        return None


def _get_intent(router_result: Any) -> str:
    """Extract intent string from router result."""
    if router_result is None:
        return "unknown"
    return getattr(router_result, "intent", "unknown") or "unknown"


def _collect_sources(execution: ExecutionResult) -> List[Dict[str, Any]]:
    """Collect unique sources from all executed steps."""
    sources: List[Dict[str, Any]] = []
    seen_keys: set = set()

    for step_id, sr in execution.step_results.items():
        if sr.error or sr.action in ("cross_reference",):
            continue
        for item in sr.data:
            source_name = item.get("source", "")
            content_preview = (item.get("content", "") or "")[:80]
            key = (source_name, content_preview)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            sources.append({
                "source": source_name,
                "page": item.get("page"),
                "section": item.get("section", ""),
                "relevance_score": item.get("relevance_score", 0.0),
                "step_id": step_id,
                "action": sr.action,
            })

    # Sort by relevance
    sources.sort(key=lambda s: s.get("relevance_score", 0), reverse=True)
    return sources


__all__ = ["QueryPipelineResult", "run_query_pipeline"]
