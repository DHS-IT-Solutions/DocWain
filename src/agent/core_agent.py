"""Core Agent Orchestrator — UNDERSTAND -> RETRIEVE -> REASON -> COMPOSE.

Ties the entire DocWain intelligence pipeline together.  For complex
queries it spawns dynamic sub-agents in parallel via ThreadPoolExecutor.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Generator, List, Optional, Set

from src.agent.intent import IntentAnalyzer, QueryUnderstanding
from src.agent.subagent import DynamicSubAgent
from src.agent.url_case_selector import (
    CaseSelection,
    RetrievalSignal,
    select_case,
)
from src.generation.composer import compose_response
from src.generation.reasoner import Reasoner, ReasonerResult
from src.retrieval.context_builder import build_context
from src.retrieval.ephemeral_merge import merge_ephemeral
from src.retrieval.reranker import rerank_chunks
from src.retrieval.retriever import UnifiedRetriever
from src.retrieval.types import RetrievalBundle
from src.agent.domain_dispatch import DomainDispatcher
from src.tools.url_ephemeral_source import (
    EphemeralResult,
    UrlEphemeralSource,
)
from src.tools.url_fetcher import DomainPolicy, FetcherConfig
from src.tools.web_search import detect_urls_in_query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords for query expansion filtering
# ---------------------------------------------------------------------------

_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "only",
    "own", "same", "than", "too", "very", "just", "about", "up", "it",
    "its", "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "any", "many", "much", "tell", "show", "give", "get",
    "document", "documents", "file", "files", "please",
}

# ---------------------------------------------------------------------------
# Query expansion synonyms by task type
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dynamic evidence count by task type
# ---------------------------------------------------------------------------

_EVIDENCE_TOP_K: Dict[str, int] = {
    "lookup": 15,
    "extract": 20,
    "list": 30,
    "summarize": 25,
    "overview": 30,
    "compare": 20,
    "investigate": 20,
    "aggregate": 15,
}

_TASK_SYNONYMS: Dict[str, List[str]] = {
    "extract": ["extract", "find", "identify", "locate", "what is", "what are"],
    "compare": ["compare", "contrast", "difference", "versus", "vs", "similarity"],
    "summarize": ["summarize", "summary", "key points", "highlights"],
    "overview": ["overview", "tell me about", "what do we have", "describe the documents", "about the documents"],
    "investigate": ["investigate", "analyze", "examine", "assess", "evaluate", "risk"],
    "lookup": ["what", "who", "when", "where", "how much"],
    "list": ["list", "enumerate", "name", "all", "each"],
    "aggregate": ["total", "count", "sum", "average", "how many"],
}

# ---------------------------------------------------------------------------
# Conversational response fragments
# ---------------------------------------------------------------------------

_CONVERSATIONAL_RESPONSES: Dict[str, str] = {
    "greeting": (
        "Welcome to DocWain. I can help you with:\n\n"
        "- **Search & Extract** — Find specific data, names, dates, amounts from your documents\n"
        "- **Compare & Analyze** — Side-by-side comparisons, rankings, pattern detection\n"
        "- **Summarize** — Executive summaries, key findings, document overviews\n"
        "- **Generate** — Draft emails, reports, cover letters from document evidence\n"
        "- **Translate** — Convert document content to other languages\n\n"
        "Just ask a question about the documents in your profile."
    ),
    "farewell": "Feel free to come back anytime. Your documents and session history are saved.",
    "thanks": "Happy to help. Let me know if there's anything else you'd like to explore in your documents.",
    "meta": (
        "I'm DocWain, a document intelligence expert. I can search, extract, compare, "
        "rank, summarize, and analyze information across all the documents in your profile. "
        "I support PDF, Word, Excel, images, and more. "
        "Try asking me something like:\n\n"
        "- \"List all candidates with Python experience\"\n"
        "- \"Compare the top 3 contracts by payment terms\"\n"
        "- \"Summarize the key findings across all documents\"\n"
        "- \"Draft an email based on the document data\"\n"
    ),
}

_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|greetings|yo)\b",
    re.IGNORECASE,
)
_FAREWELL_RE = re.compile(
    r"^\s*(?:bye|goodbye|see\s+you)\b",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^\s*(?:thanks|thank\s+you|cheers)\b",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"(?:who\s+are\s+you|what\s+can\s+you\s+(?:do|help)|how\s+(?:do\s+I|can\s+I|to)\s+(?:upload|use|start|get\s+started)|what\s+(?:types?\s+of\s+)?files?\s+(?:can|do)|show\s+me\s+example|help\s+me)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Phase 4 — module-level wiring helpers (test-patchable seams)
# ---------------------------------------------------------------------------


async def _load_adapter(subscription_id: str, profile_domain: str):
    """Thin wrapper around the module-level adapter loader.

    Kept at module scope so tests can patch it without constructing a full
    CoreAgent + AdapterLoader singleton. In production the singleton is
    initialised at FastAPI lifespan; this helper routes through it.
    """
    from src.intelligence.sme.adapter_loader import get_adapter_loader
    return get_adapter_loader().load(subscription_id, profile_domain)


async def _is_rich_mode_enabled(subscription_id: str) -> bool:
    """Delegates to the Phase 1 flag resolver (ERRATA §4).

    Wrapped in a try/except so pre-Phase-1 deployments (where the resolver
    singleton isn't initialised) degrade to compact without raising into
    the hot path.
    """
    try:
        from src.config.feature_flags import (
            ENABLE_RICH_MODE,
            get_flag_resolver,
        )
        return get_flag_resolver().is_enabled(subscription_id, ENABLE_RICH_MODE)
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# CoreAgent
# ---------------------------------------------------------------------------


class CoreAgent:
    """Orchestrates the full UNDERSTAND -> RETRIEVE -> REASON -> COMPOSE pipeline."""

    MAX_SUBAGENTS = 5
    SUBAGENT_TIMEOUT = 30.0

    def __init__(
        self,
        llm_gateway: Any,
        qdrant_client: Any,
        embedder: Any,
        mongodb: Any,
        kg_query_service: Any = None,
        cross_encoder: Any = None,
        *,
        sme_retriever: Any = None,
        sme_kg_client: Any = None,
        hybrid_searcher: Any = None,
        redis_client: Any = None,
    ) -> None:
        self._llm = llm_gateway
        self._qdrant = qdrant_client
        self._embedder = embedder
        self._mongodb = mongodb
        self._intent_analyzer = IntentAnalyzer(llm_gateway=llm_gateway)
        self._retriever = UnifiedRetriever(qdrant_client=qdrant_client, embedder=embedder)
        self._reasoner = Reasoner(llm_gateway=llm_gateway)
        self.kg_query_service = kg_query_service
        self._cross_encoder = cross_encoder
        self._domain_dispatcher = DomainDispatcher(llm_gateway=llm_gateway)
        # ------------------------------------------------------------------
        # Phase 3 additions — SME 4-layer retrieval inputs. All optional so
        # pre-Phase-3 callers keep working unchanged; the orchestrator
        # returns Layer A only when these are not wired.
        # ------------------------------------------------------------------
        self._sme_retriever = sme_retriever  # src.retrieval.sme_retrieval.SMERetrieval
        self._sme_kg_client = sme_kg_client  # src.retrieval.unified_retriever.UnifiedRetrieverKGClient
        self._hybrid_searcher = hybrid_searcher  # Phase 1 HybridSearcher (optional)
        self._redis_client_injected = redis_client  # Phase 3 QA fast-path
        # Phase 3 Task 8/10 — retrieval cache bound to the injected redis
        # client. Constructed lazily because we want the same instance
        # across handle() calls within a CoreAgent lifetime but we don't
        # want to pay the Redis lookup if the feature flag is off.
        self._retrieval_cache_obj = None  # type: Any
        # Phase 3 Task 9/10 — intent gate. Stateless; shared across calls.
        self._intent_gate_obj = None  # type: Any
        # Phase 5 — dedicated executor for URL fetch. Separate from the
        # UNDERSTAND+RETRIEVE ThreadPoolExecutor because URL fetch must
        # outlive that block (it may resolve during rerank or later) and
        # must not block its `with`-exit. The executor is shared across
        # calls; threads are daemons and the pool auto-shrinks on idle.
        self._url_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="docwain-url-fetch",
        )

    # ------------------------------------------------------------------
    # Phase 5 — URL-as-prompt helpers
    # ------------------------------------------------------------------
    def _build_fetcher_config(self, subscription_id: str) -> FetcherConfig:
        """Construct a :class:`FetcherConfig` for a subscription.

        Pulls optional allow/block domain lists from Config when present
        (``Config.UrlFetcher.ALLOWED_DOMAINS``, ``BLOCKED_DOMAINS``). Safe
        defaults otherwise. External I/O safety timeouts (fetch 15s,
        extract 30s) come from the dataclass defaults — no internal
        wall-clock timeouts are added on top.
        """
        allowed: tuple[str, ...] = ()
        blocked: tuple[str, ...] = ()
        try:
            from src.api.config import Config
            cfg_mod = getattr(Config, "UrlFetcher", None)
            if cfg_mod is not None:
                allowed = tuple(
                    getattr(cfg_mod, "ALLOWED_DOMAINS", ()) or ()
                )
                blocked = tuple(
                    getattr(cfg_mod, "BLOCKED_DOMAINS", ()) or ()
                )
        except Exception:  # noqa: BLE001
            pass
        return FetcherConfig(
            domain_policy=DomainPolicy(
                allowed_domains=allowed,
                blocked_domains=blocked,
            ),
        )

    def _is_url_as_prompt_enabled(self, subscription_id: str) -> bool:
        """Return True iff the Phase 5 flag is ON for *subscription_id*.

        Fail-closed: any resolver failure (singleton uninitialised, store
        error, etc.) returns False so the URL leg is a no-op.
        """
        try:
            from src.config.feature_flags import (
                ENABLE_URL_AS_PROMPT,
                get_flag_resolver,
            )
            return bool(get_flag_resolver().is_enabled(
                subscription_id, ENABLE_URL_AS_PROMPT,
            ))
        except Exception:  # noqa: BLE001
            return False

    def _kick_off_url_fetch(
        self,
        *,
        urls: List[str],
        subscription_id: str,
        profile_id: str,
        session_id: str,
    ) -> Any:
        """Submit the URL fetch to the dedicated executor and return the future.

        Dispatch is deliberately isolated from the UNDERSTAND+RETRIEVE
        executor — URL work may complete well after the profile leg, and we
        must not let the with-exit of that block join on it. No internal
        timeout is applied; external fetch/extract safety inside
        :class:`UrlEphemeralSource` is the only timeout.
        """
        fetcher_cfg = self._build_fetcher_config(subscription_id)
        ephemeral = UrlEphemeralSource(
            embedder=self._embedder,
            fetcher_config=fetcher_cfg,
        )

        def _run() -> EphemeralResult:
            try:
                return ephemeral.fetch_all(
                    urls,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    session_id=session_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ephemeral url batch failed entirely: %s", exc,
                    exc_info=True,
                )
                return EphemeralResult(
                    chunks=[],
                    warnings=[{
                        "url": ",".join(urls),
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                        "error_class": type(exc).__name__,
                    }],
                )

        return self._url_executor.submit(_run)

    # ------------------------------------------------------------------
    # Phase 3 — four-layer SME retrieval orchestration
    # ------------------------------------------------------------------

    def _retrieve_four_layers(
        self,
        *,
        query: str,
        subscription_id: str,
        profile_id: str,
        query_understanding: Dict[str, Any],
    ) -> "RetrievalBundle":
        """Dispatch the Phase 3 four-layer parallel retrieval and return
        a :class:`RetrievalBundle`.

        Layer A is the existing dense-or-hybrid chunk retrieval. Layer B
        uses :class:`src.retrieval.unified_retriever.UnifiedRetriever`'s
        :meth:`retrieve_layer_b` helper. Layer C calls the injected
        :class:`SMERetrieval` (gated on ``ENABLE_SME_RETRIEVAL``). Layer D
        is a no-op placeholder until Phase 5.

        Layers B and C only run when the caller has wired them in via the
        constructor — pre-Phase-3 deployments keep receiving an empty
        bundle for those slots without error. Per ERRATA §11 a layer
        failure degrades into ``bundle.degraded_layers`` (full name only,
        single append); the remaining layers always return.
        """
        from src.retrieval.types import RetrievalBundle as _RB
        from src.retrieval.unified_retriever import UnifiedRetriever as _SMEUnified
        # Pluggable top-K defaults — Task 6 will tune these.
        intent = (query_understanding or {}).get("intent") or (
            query_understanding or {}
        ).get("task_type", "lookup")

        # Build the SME-side UnifiedRetriever lazily; it is cheap.
        sme_unified = _SMEUnified(
            qdrant_client=self._qdrant,
            kg_client=self._sme_kg_client,
            sme=self._sme_retriever,
        )

        # Default layer callables — fall through to empty when the
        # dependency isn't wired so tests and pre-Phase-3 callers succeed.
        def _layer_a_fn() -> List[Dict[str, Any]]:
            try:
                return sme_unified.retrieve_layer_a(
                    query=query,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    query_understanding=query_understanding,
                    top_k=10,
                )
            except Exception:  # noqa: BLE001
                raise

        def _layer_b_fn() -> List[Dict[str, Any]]:
            if self._sme_kg_client is None:
                return []
            return sme_unified.retrieve_layer_b(
                query=query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                top_k=5,
                entities=(query_understanding or {}).get("entities") or None,
            )

        def _layer_c_fn() -> List[Dict[str, Any]]:
            if self._sme_retriever is None:
                return []
            from src.retrieval.unified_retriever import _sme_hit_to_dict
            hits = self._sme_retriever.retrieve(
                query=query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                top_k=5,
            )
            return [_sme_hit_to_dict(h) for h in hits]

        def _layer_d_fn() -> List[Dict[str, Any]]:
            return []

        # Phase 3 Task 9 — consult the shared intent gate so simple
        # intents skip B/C entirely. Task 10 passes the same gate to the
        # four-layer orchestrator so gated layers never hit their
        # backing store.
        gate = self._intent_gate()
        bundle = sme_unified.retrieve_four_layers(
            query=query,
            subscription_id=subscription_id,
            profile_id=profile_id,
            query_understanding=query_understanding,
            layer_a_fn=_layer_a_fn,
            layer_b_fn=_layer_b_fn,
            layer_c_fn=_layer_c_fn,
            layer_d_fn=_layer_d_fn,
            gate=gate,
        )
        return bundle

    # ------------------------------------------------------------------
    # Phase 3 Task 10 — SME pack assembly + doc_context injection
    # ------------------------------------------------------------------

    def _intent_gate(self) -> Any:
        """Lazily construct and return the shared :class:`IntentGate`."""
        if self._intent_gate_obj is None:
            from src.retrieval.intent_gating import IntentGate
            self._intent_gate_obj = IntentGate()
        return self._intent_gate_obj

    def _retrieval_cache(self) -> Any:
        """Lazily construct and return the shared :class:`RetrievalCache`.

        Uses the injected Redis client if present, else the app-state
        client. Returns a cache bound to ``None`` when Redis isn't
        explicitly wired — important for unit tests that construct
        :class:`CoreAgent` without a redis_client so Phase 3 doesn't
        blocking-connect against a non-existent Redis. Production
        deployments inject the app-state client at construction time.
        """
        if self._retrieval_cache_obj is None:
            from src.retrieval.retrieval_cache import RetrievalCache
            # Only prefer the app-state client when the env is set up —
            # bare ``CoreAgent(...)`` constructions (tests) skip Redis.
            client = self._redis_client_injected
            if client is None:
                try:
                    from src.api.rag_state import get_app_state
                    app_state = get_app_state()
                    if app_state is not None and hasattr(
                        app_state, "redis_client"
                    ):
                        client = app_state.redis_client
                except Exception:  # noqa: BLE001
                    client = None
            self._retrieval_cache_obj = RetrievalCache(redis_client=client)
        return self._retrieval_cache_obj

    def _resolve_adapter(
        self, subscription_id: str, profile_domain: str
    ) -> Any:
        """Resolve the domain adapter for pack assembly.

        Returns a thin stub with the default ``retrieval_caps`` shape
        when the :class:`AdapterLoader` singleton isn't initialized —
        this happens in unit tests and in the pre-SME production
        deployment. The stub always satisfies
        :class:`PackAssembler` so Phase 3 wiring never explodes on
        missing infrastructure.
        """
        try:
            from src.intelligence.sme.adapter_loader import get_adapter_loader
            return get_adapter_loader().load(subscription_id, profile_domain)
        except Exception:  # noqa: BLE001
            # Stub adapter — PackAssembler reads ``.retrieval_caps.max_pack_tokens``.
            class _StubCaps:
                max_pack_tokens = {"generic": 4000}
            class _StubAdapter:
                retrieval_caps = _StubCaps()
            return _StubAdapter()

    def _build_sme_pack(
        self,
        *,
        query: str,
        subscription_id: str,
        profile_id: str,
        profile_domain: str,
        intent: str,
        query_understanding: Dict[str, Any],
    ) -> List[Any]:
        """Run the full Phase 3 SME retrieval + assembly pipeline.

        Pipeline:

        1. Retrieval-cache lookup keyed by
           ``(sub, prof, query_fingerprint, flag_set_version)``. Cache
           stores the raw :class:`RetrievalBundle`; a hit skips the four
           layer dispatch entirely.
        2. On miss — :meth:`_retrieve_four_layers` dispatches Layer
           A/B/C/D in parallel, honouring the intent gate.
        3. :func:`merge_layers` unions the four outputs and tags Layer
           C + inferred Layer B as ``sme_backed=True``.
        4. :func:`rerank_merged_candidates` applies the cross-encoder
           blend + SME-intent bonus.
        5. :func:`mmr_select` picks a diverse top-K (default 10 items).
        6. :class:`PackAssembler` compresses Layer C and enforces the
           adapter's per-intent token budget.

        Returns a ``List[PackedItem]`` (frozen dataclasses) ready for
        ``doc_context["sme_pack"]``. The assembled list is always
        returned — even when retrieval fails every layer the empty list
        is safe for the reasoner path.
        """
        from src.config.feature_flags import (
            ENABLE_CROSS_ENCODER_RERANK,
            get_flag_set_version,
            get_flag_resolver,
        )
        from src.retrieval.merge import (
            merge_layers,
            mmr_select,
            rerank_merged_candidates,
        )
        from src.retrieval.pack_assembler import PackAssembler
        from src.retrieval.retrieval_cache import _query_fingerprint

        fp = _query_fingerprint(query)
        version = get_flag_set_version()
        cache = self._retrieval_cache()
        bundle = cache.get(
            subscription_id=subscription_id,
            profile_id=profile_id,
            query_fingerprint=fp,
            flag_set_version=f"v{version}",
        )
        if bundle is None:
            bundle = self._retrieve_four_layers(
                query=query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                query_understanding=query_understanding,
            )
            cache.set(
                subscription_id=subscription_id,
                profile_id=profile_id,
                query_fingerprint=fp,
                flag_set_version=f"v{version}",
                bundle=bundle,
            )

        merged = merge_layers(bundle)
        if not merged:
            return []

        # Cross-encoder rerank — gated per-subscription by
        # ENABLE_CROSS_ENCODER_RERANK. When flag off, we still call
        # rerank_merged_candidates so the score-sort path runs, but we
        # pass enable_cross_encoder=False to skip the CE model call.
        enable_ce = False
        try:
            resolver = get_flag_resolver()
            enable_ce = bool(
                resolver.is_enabled(subscription_id, ENABLE_CROSS_ENCODER_RERANK)
            )
        except Exception:  # noqa: BLE001
            enable_ce = False

        reranked = rerank_merged_candidates(
            query=query,
            candidates=merged,
            cross_encoder=self._cross_encoder,
            top_k=10,
            intent=intent,
            enable_cross_encoder=enable_ce,
        )
        diverse = mmr_select(items=reranked, top_k=10, lam=0.7)

        adapter = self._resolve_adapter(subscription_id, profile_domain or "generic")
        pack = PackAssembler(adapter).assemble(items=diverse, intent=intent)
        return pack

    # ------------------------------------------------------------------
    # Phase 3 Task 7 — QA-cache fast path
    # ------------------------------------------------------------------

    def _qa_fast_path_lookup(
        self,
        *,
        query: str,
        subscription_id: str,
        profile_id: str,
        min_confidence: float = 0.85,
    ) -> Optional[Dict[str, Any]]:
        """Look up ``qa_idx:{sub}:{prof}:{fingerprint}`` in Redis.

        Returns an AnswerPayload-shaped dict when a hit exists at or above
        ``min_confidence``; otherwise ``None`` so the caller falls through
        to the full retrieve+reason flow. All Redis errors degrade to a
        miss — the fast path must never raise into the hot path.

        The fingerprint is computed via
        :func:`src.intelligence.qa_generator.qa_index_fingerprint` so the
        Phase 2 emission side (``emit_qa_index``) and this read side stay
        in lock-step.
        """
        try:
            from src.intelligence.qa_generator import qa_index_fingerprint
        except Exception:  # noqa: BLE001
            return None
        redis_client = self._redis_client_injected or self._get_redis_client()
        if redis_client is None:
            return None
        fingerprint = qa_index_fingerprint(query)
        key = f"qa_idx:{subscription_id}:{profile_id}:{fingerprint}"
        try:
            raw = redis_client.get(key)
        except Exception:  # noqa: BLE001
            logger.debug("qa_fast_path redis.get failed", exc_info=True)
            return None
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8")
            except Exception:  # noqa: BLE001
                return None
        try:
            import json
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            logger.debug("qa_fast_path json decode failed", exc_info=True)
            return None

        answer = data.get("answer")
        if not answer:
            return None
        metadata_block = data.get("metadata") or {}
        confidence = metadata_block.get("confidence")
        if confidence is None:
            confidence = data.get("confidence")
        if confidence is not None and float(confidence) < float(min_confidence):
            return None

        # Shape the payload to match the existing AnswerPayload dict so
        # the caller treats the fast-path and the full pipeline
        # identically. No Reasoner invocation happens on this path.
        return {
            "response": str(answer),
            "sources": list(metadata_block.get("sources") or []),
            "grounded": True,
            "context_found": True,
            "metadata": {
                "task_type": metadata_block.get("task_type", "lookup"),
                "engine": "docwain_core_agent_qa_fast_path",
                "qa_id": data.get("qa_id") or metadata_block.get("qa_id", ""),
                "qa_fast_path_hit": True,
                "qa_confidence": confidence,
                "qa_index_fingerprint": fingerprint,
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        user_id: str,
        session_id: str,
        conversation_history: Optional[List[Dict[str, str]]],
        *,
        agent_name: Optional[str] = None,
        document_id: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Run the full pipeline and return an AnswerPayload dict."""
        if not subscription_id or not str(subscription_id).strip():
            raise ValueError("subscription_id is required")
        if not profile_id or not str(profile_id).strip():
            raise ValueError("profile_id is required")

        # --- Phase 3 Task 7 — QA fast path ---
        # Before spending UNDERSTAND+RETRIEVE+REASON budget, probe Redis
        # for a pre-grounded Q&A pair at the query fingerprint. Hit →
        # short-circuit to the cached answer immediately (Reasoner skipped).
        _qa_t0 = time.monotonic()
        _qa_hit = self._qa_fast_path_lookup(
            query=query,
            subscription_id=subscription_id,
            profile_id=profile_id,
        )
        if _qa_hit is not None:
            _qa_hit.setdefault("metadata", {})["timing"] = {
                "qa_fast_path_ms": round(
                    (time.monotonic() - _qa_t0) * 1000, 1
                ),
            }
            logger.info(
                "[QA_FAST_PATH] hit profile=%s sub=%s fingerprint=%s",
                profile_id,
                subscription_id,
                _qa_hit["metadata"].get("qa_index_fingerprint"),
            )
            return _qa_hit

        timing: Dict[str, float] = {}

        # --- PHASE 5: URL-AS-PROMPT KICK-OFF ---
        # When ENABLE_URL_AS_PROMPT is on for the subscription, detect any
        # URLs in the query and kick off the fetch+extract+chunk+embed
        # pipeline in parallel with UNDERSTAND and RETRIEVE. NO internal
        # timeout — the external fetch/extract safety is the only guard.
        url_list: List[str] = []
        cleaned_query = query
        url_case: CaseSelection = CaseSelection.NONE
        _ephemeral_future = None
        ephemeral_warnings: List[Dict[str, Any]] = []
        ephemeral_chunks: List[Any] = []
        if self._is_url_as_prompt_enabled(subscription_id):
            url_list, cleaned_query = detect_urls_in_query(query)
            if url_list:
                _ephemeral_future = self._kick_off_url_fetch(
                    urls=url_list,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    session_id=session_id,
                )
                logger.info(
                    "[URL_AS_PROMPT] kicked off fetch of %d url(s) for sub=%s",
                    len(url_list), subscription_id,
                )

        # --- UNDERSTAND ---
        t0 = time.monotonic()
        doc_intelligence = self._load_doc_intelligence(subscription_id)
        doc_intelligence_dict = {
            d.get("document_id", ""): d.get("intelligence", d)
            for d in doc_intelligence
        }

        # KG probe — extract entities from query and find related docs/chunks
        kg_hints: Dict[str, Any] = {}
        if self.kg_query_service:
            try:
                query_entities = self.kg_query_service.extract_entities(query)
                if query_entities:
                    kg_result = self.kg_query_service.query(
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        domain_hint=None,
                        entities=query_entities,
                    )
                    kg_hints = {
                        "target_doc_ids": kg_result.doc_ids,
                        "target_chunk_ids": kg_result.chunk_ids,
                        "entities": query_entities,
                    }
            except Exception as exc:
                logger.debug("KG probe failed (non-fatal): %s", exc)

        # Trim doc_intelligence for intent analysis (only needs summaries/topics, not full entities)
        trimmed_intel = []
        for d in doc_intelligence[:10]:  # cap at 10 docs
            trimmed = {
                "document_id": d.get("document_id", ""),
                "profile_id": d.get("profile_id", ""),
                "profile_name": d.get("profile_name", ""),
            }
            intel = d.get("intelligence") or {}
            trimmed["summary"] = (intel.get("summary") or "")[:200]
            trimmed["answerable_topics"] = (intel.get("answerable_topics") or [])[:5]
            trimmed["document_type"] = intel.get("document_type", "")
            trimmed_intel.append(trimmed)

        # --- PARALLEL: UNDERSTAND + PRE-FETCH RETRIEVE ---
        # Launch intent analysis (LLM, ~20s) and a broad retrieval (vector search, ~2s)
        # concurrently so the retrieval result is ready by the time intent finishes.
        prefetch_result = None
        _intent_future = None
        _prefetch_future = None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _par_executor:
                # Thread 1: Intent analysis (LLM call)
                _intent_future = _par_executor.submit(
                    self._intent_analyzer.analyze,
                    query, subscription_id, profile_id, trimmed_intel, conversation_history,
                    kg_hints,
                )

                # Thread 2: Broad pre-fetch retrieval — no intent filtering yet, use KG doc
                # hints if available as a light scope hint.
                _prefetch_doc_ids = kg_hints.get("target_doc_ids") or None
                _prefetch_kwargs = {
                    "query": query,
                    "subscription_id": subscription_id,
                    "profile_ids": [profile_id],
                }
                if _prefetch_doc_ids:
                    _prefetch_kwargs["document_ids"] = _prefetch_doc_ids
                _prefetch_future = _par_executor.submit(
                    lambda kw=_prefetch_kwargs: self._retriever.retrieve(**kw),
                )

                understanding = _intent_future.result(timeout=45.0)
                prefetch_result = _prefetch_future.result(timeout=45.0)

            logger.debug("Parallel UNDERSTAND+RETRIEVE complete")
        except Exception as _par_exc:
            logger.warning(
                "Parallel UNDERSTAND+RETRIEVE failed (%s) — falling back to sequential",
                _par_exc,
            )
            # Salvage any future that already completed to avoid re-running it.
            understanding = None
            if _intent_future is not None and _intent_future.done() and not _intent_future.cancelled():
                try:
                    understanding = _intent_future.result(timeout=0)
                except Exception:
                    pass
            if _prefetch_future is not None and _prefetch_future.done() and not _prefetch_future.cancelled():
                try:
                    prefetch_result = _prefetch_future.result(timeout=0)
                except Exception:
                    prefetch_result = None

            if understanding is None:
                # If intent timed out, use safe defaults instead of re-running
                # the same slow LLM call (which would double total latency).
                if isinstance(_par_exc, (concurrent.futures.TimeoutError, TimeoutError)):
                    logger.warning("Intent analysis timed out — using safe defaults")
                    understanding = self._intent_analyzer._safe_defaults(query)
                    self._intent_analyzer._enrich_relevant_documents(
                        understanding, query, trimmed_intel, kg_hints,
                    )
                else:
                    understanding = self._intent_analyzer.analyze(
                        query, subscription_id, profile_id, trimmed_intel,
                        conversation_history, kg_hints=kg_hints,
                    )

        timing["understand_ms"] = round((time.monotonic() - t0) * 1000, 1)

        logger.info(
            "[RAG_QUERY] query=%r profile=%s subscription=%s user=%s task_type=%s",
            query[:100], profile_id, subscription_id, user_id,
            getattr(understanding, "task_type", "?"),
        )

        # --- Fetch document index + intelligence for profile awareness ---
        # Always fetch both — doc_intelligence is the richest context source
        # and is needed even for specific queries (extract, lookup, investigate)
        doc_index_entries: List[str] = []
        doc_intelligence_entries: List[str] = []
        try:
            from qdrant_client.models import Filter as _QFilter, FieldCondition as _QFC, MatchValue as _QMV
            from src.api.vector_store import build_collection_name
            _collection = build_collection_name(subscription_id)

            # Always fetch doc_index (compact, ~50 tokens per doc)
            _idx_points, _ = self._qdrant.scroll(
                collection_name=_collection,
                scroll_filter=_QFilter(must=[
                    _QFC(key="profile_id", match=_QMV(value=str(profile_id))),
                    _QFC(key="resolution", match=_QMV(value="doc_index")),
                ]),
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
            doc_index_entries = [
                (p.payload or {}).get("canonical_text", "")
                for p in _idx_points
                if (p.payload or {}).get("canonical_text")
            ]

            # Always fetch doc_intelligence — critical context for ALL query types
            _intel_points, _ = self._qdrant.scroll(
                collection_name=_collection,
                scroll_filter=_QFilter(must=[
                    _QFC(key="profile_id", match=_QMV(value=str(profile_id))),
                    _QFC(key="resolution", match=_QMV(value="doc_intelligence")),
                ]),
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
            doc_intelligence_entries = [
                (p.payload or {}).get("canonical_text", "")
                for p in _intel_points
                if (p.payload or {}).get("canonical_text")
            ]

            logger.info(
                "[DOC_INDEX] Fetched %d doc_index + %d doc_intelligence entries for profile %s",
                len(doc_index_entries), len(doc_intelligence_entries), profile_id,
            )
        except Exception as _di_exc:
            logger.debug("[DOC_INDEX] Fetch failed (non-fatal): %s", _di_exc)

        if understanding.is_conversational:
            return self._handle_conversational(query)

        # --- DOMAIN DISPATCH (only when explicitly requested) ---
        if agent_name:
            domain_result = self._domain_dispatcher.try_handle(
                query=understanding.resolved_query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                evidence=[],
                doc_context={},
                agent_name=agent_name,
                document_id=document_id,
            )
            if domain_result is not None:
                domain_result.setdefault("metadata", {})["timing"] = timing
                return domain_result

        # --- RETRIEVE (filter pre-fetched result with intent, or run focused retrieval) ---
        t0 = time.monotonic()
        profile_ids = self._resolve_profile_scope(understanding, profile_id)

        document_ids: Optional[List[str]] = None
        if document_id:
            document_ids = [document_id]
        elif understanding.relevant_documents:
            doc_ids = [
                d.get("document_id", "") for d in understanding.relevant_documents
                if d.get("document_id")
            ]
            if doc_ids:
                document_ids = doc_ids

        # Merge KG-hinted doc IDs into retrieval scope
        kg_doc_ids = kg_hints.get("target_doc_ids", [])
        if kg_doc_ids:
            if document_ids is None:
                document_ids = list(kg_doc_ids)
            else:
                existing = set(document_ids)
                for did in kg_doc_ids:
                    if did not in existing:
                        document_ids.append(did)

        # Enhance query for better retrieval coverage (always computed — used by reranker)
        enhanced_query = self._enhance_query(
            understanding.resolved_query,
            understanding.task_type,
            doc_intelligence,
            understanding.entities,
        )

        # Use the pre-fetched result when available; apply intent-driven doc filtering.
        if prefetch_result is not None:
            # Filter pre-fetched chunks to intent-resolved document scope
            if understanding.relevant_documents:
                target_doc_ids = {
                    d.get("document_id") for d in understanding.relevant_documents
                    if d.get("document_id")
                }
                if target_doc_ids:
                    prefetch_result.chunks = [
                        c for c in prefetch_result.chunks
                        if getattr(c, "document_id", None) in target_doc_ids
                    ]

            if document_id:
                prefetch_result.chunks = [
                    c for c in prefetch_result.chunks
                    if getattr(c, "document_id", None) == document_id
                ]

            # If filtering left too few chunks, do a focused re-retrieval with the
            # enhanced (intent-resolved) query and the narrowed document scope.
            if len(prefetch_result.chunks) < 3 and (document_id or understanding.relevant_documents):
                logger.debug(
                    "Pre-fetch yielded %d chunks after filtering — running focused re-retrieval",
                    len(prefetch_result.chunks),
                )
                retrieval_result = self._retriever.retrieve(
                    enhanced_query,
                    subscription_id,
                    profile_ids,
                    document_ids=document_ids,
                )
            else:
                retrieval_result = prefetch_result
        else:
            # Fallback: sequential retrieval (parallel block failed)
            retrieval_result = self._retriever.retrieve(
                enhanced_query,
                subscription_id,
                profile_ids,
                document_ids=document_ids,
            )

        # Dynamic evidence count by task type
        evidence_top_k = _EVIDENCE_TOP_K.get(understanding.task_type, 15)

        # --- Profile isolation audit on retrieved chunks ---
        _raw_chunks = retrieval_result.chunks or []
        _chunk_profiles = set()
        _chunk_sources = set()
        _foreign_count = 0
        for _rc in _raw_chunks:
            _rc_pid = getattr(_rc, "profile_id", None) or (getattr(_rc, "metadata", {}) or {}).get("profile_id", "")
            _rc_src = (getattr(_rc, "metadata", {}) or {}).get("source_name", getattr(_rc, "document_id", "?"))
            _chunk_profiles.add(str(_rc_pid))
            _chunk_sources.add(str(_rc_src))
            if str(_rc_pid) and str(_rc_pid) != str(profile_id):
                _foreign_count += 1
        if _foreign_count:
            logger.error(
                "[PROFILE_ISOLATION_VIOLATION] %d/%d retrieved chunks belong to foreign profiles %s "
                "(expected=%s query=%r)",
                _foreign_count, len(_raw_chunks), _chunk_profiles - {str(profile_id)},
                profile_id, query[:80],
            )
        logger.info(
            "[RAG_RETRIEVAL] profile=%s chunks=%d sources=%s profiles_seen=%s",
            profile_id, len(_raw_chunks), list(_chunk_sources)[:10], list(_chunk_profiles),
        )

        logger.info(
            "[RAG_DEBUG] pre-rerank: %d chunks from %d docs, evidence_top_k=%d",
            len(retrieval_result.chunks),
            len(set(c.document_id for c in retrieval_result.chunks if c.document_id)),
            evidence_top_k,
        )
        reranked = rerank_chunks(
            understanding.resolved_query,  # rerank against original query, not expanded
            retrieval_result.chunks,
            top_k=evidence_top_k,
            cross_encoder=self._cross_encoder,
        )
        logger.info(
            "[RAG_DEBUG] post-rerank: %d chunks from %d docs",
            len(reranked),
            len(set(c.document_id for c in reranked if c.document_id)),
        )

        # --- PHASE 5: URL-AS-PROMPT CASE DISPATCH + MERGE ---
        if _ephemeral_future is not None:
            # Compute retrieval-signal strength off the reranked profile
            # chunks. "High similarity" threshold is the same 0.5 rerank
            # score floor used in the spec.
            _high_sim = sum(
                1 for c in reranked if getattr(c, "score", 0.0) >= 0.5
            )
            _sme_artifact_count = sum(
                1 for c in reranked
                if isinstance(getattr(c, "metadata", {}), dict)
                and (c.metadata or {}).get("provenance") == "sme_artifact"
            )
            signal = RetrievalSignal(
                sme_artifact_count=_sme_artifact_count,
                high_sim_chunk_count=_high_sim,
            )
            url_case = select_case(
                cleaned_query=cleaned_query,
                url_count=len(url_list),
                signal=signal,
            )
            try:
                # External safety lives inside UrlEphemeralSource; no
                # internal .result(timeout=...) is added here — per the
                # Phase 5 rule.
                ephemeral_result: EphemeralResult = _ephemeral_future.result()
                ephemeral_chunks = list(ephemeral_result.chunks)
                ephemeral_warnings = list(ephemeral_result.warnings)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ephemeral url fetch future failed: %s", exc,
                    exc_info=True,
                )
                ephemeral_chunks = []
                ephemeral_warnings = [{
                    "url": ",".join(url_list),
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "error_class": type(exc).__name__,
                }]
            if url_case in (CaseSelection.PRIMARY, CaseSelection.SUPPLEMENTARY):
                reranked = merge_ephemeral(
                    reranked, ephemeral_chunks, case=url_case,
                )
                logger.info(
                    "[URL_AS_PROMPT] merged %d ephemeral chunk(s) case=%s",
                    len(ephemeral_chunks), url_case.value,
                )

        evidence, doc_context = build_context(reranked, doc_intelligence_dict)
        logger.info("[RAG_DEBUG] evidence items: %d", len(evidence))

        # Inject KG entity relationships into doc_context for richer reasoning
        if kg_hints.get("target_doc_ids") and self.kg_query_service:
            try:
                kg_entities = kg_hints.get("entities", [])
                if kg_entities:
                    kg_context = [
                        f"{e.get('value', '')} ({e.get('type', '')})"
                        for e in (kg_entities if isinstance(kg_entities, list) else [])
                        if isinstance(e, dict) and e.get("value")
                    ]
                    if kg_context:
                        existing = doc_context.get("entities") or []
                        for kc in kg_context[:5]:
                            if kc not in existing:
                                existing.append(kc)
                        doc_context["entities"] = existing[:25]
            except Exception:
                logger.debug("KG context enrichment failed (non-fatal)")

        timing["retrieve_ms"] = round((time.monotonic() - t0) * 1000, 1)

        # --- POST-RETRIEVAL DOMAIN DISPATCH ---
        if agent_name:
            post_domain_result = self._domain_dispatcher.try_handle(
                query=understanding.resolved_query,
                subscription_id=subscription_id,
                profile_id=profile_id,
                evidence=evidence,
                doc_context=doc_context,
                agent_name=agent_name,
                document_id=document_id,
            )
            if post_domain_result is not None:
                post_domain_result.setdefault("metadata", {})["timing"] = timing
                return post_domain_result

        # --- KG CONTEXT ENRICHMENT (Redis hot cache → Neo4j fallback) ---
        profile_domain = "general"
        kg_context_text = ""
        try:
            from src.intelligence.hot_cache import (
                get_profile_domain,
                get_document_facts,
                get_document_summary,
                lookup_entities,
                get_top_relationships,
            )
            redis_client = self._get_redis_client()
            if redis_client:
                profile_domain = get_profile_domain(redis_client, profile_id)

                # Gather facts from documents used in evidence
                evidence_doc_ids = list({
                    e.get("document_id", "") for e in evidence if e.get("document_id")
                })
                kg_parts = []

                # Entity lookup from query
                query_words = [
                    w for w in understanding.resolved_query.split()
                    if w.lower() not in _STOPWORDS and len(w) > 2
                ]
                cached_entities = lookup_entities(redis_client, profile_id, query_words)
                if cached_entities:
                    entity_lines = [
                        f"- {e['name']} ({e.get('type', 'unknown')}): {e.get('context', '')}"
                        for e in cached_entities[:8]
                    ]
                    if entity_lines:
                        kg_parts.append("Known entities:\n" + "\n".join(entity_lines))

                # Facts from evidence documents
                for did in evidence_doc_ids[:3]:
                    facts = get_document_facts(redis_client, profile_id, did, max_facts=5)
                    for f in facts:
                        kg_parts.append(f"- Fact: {f.get('statement', '')}")

                # Top relationships
                rels = get_top_relationships(redis_client, profile_id, max_results=5)
                for r in rels:
                    kg_parts.append(
                        f"- Relationship: {r.get('subject', '')} {r.get('relation', '')} {r.get('object', '')}"
                    )

                if kg_parts:
                    kg_context_text = "\n".join(kg_parts[:20])

        except ImportError:
            logger.debug("Hot cache module not available — skipping KG enrichment")
        except Exception as exc:
            logger.debug("KG context enrichment failed (non-fatal): %s", exc)

        # --- Enrich doc_context with doc_index / doc_intelligence ---
        if doc_index_entries:
            doc_context["doc_index"] = doc_index_entries
        if doc_intelligence_entries:
            # Prioritize intelligence entries that match documents mentioned in the query
            _query_lower = understanding.resolved_query.lower().replace("_", " ").replace(".pdf", "")
            _prioritized = []
            _others = []
            for entry in doc_intelligence_entries:
                # Extract document filename from "Document: filename.pdf" line
                _doc_name = ""
                for _line in entry.split("\n")[:3]:
                    if _line.lower().startswith("document:"):
                        _doc_name = _line.split(":", 1)[1].strip()
                        break
                _match = False
                if _doc_name:
                    _name_normalized = _doc_name.lower().replace("_", " ").replace(".pdf", "")
                    _match = _name_normalized in _query_lower or _query_lower in _name_normalized
                if _match:
                    _prioritized.append(entry)
                else:
                    _others.append(entry)

            if _prioritized:
                # For specific-doc queries: send matching docs as PRIMARY, limit others
                # to avoid diluting the LLM's attention across 50+ entries
                _max_others = 5 if len(_prioritized) <= 3 else 0
                doc_context["doc_intelligence_summaries"] = _prioritized + _others[:_max_others]
                logger.info(
                    "[DOC_INDEX] Prioritized %d matching + %d other doc_intelligence entries",
                    len(_prioritized), min(len(_others), _max_others),
                )
            else:
                # When retrieval found 0 evidence chunks, sending all summaries
                # creates an enormous prompt that makes the reasoner slow.
                # Cap to a reasonable number to keep latency under control.
                if not evidence:
                    doc_context["doc_intelligence_summaries"] = doc_intelligence_entries[:8]
                    logger.info(
                        "[DOC_INDEX] Capped doc_intelligence to %d/%d (no evidence chunks)",
                        min(8, len(doc_intelligence_entries)), len(doc_intelligence_entries),
                    )
                else:
                    doc_context["doc_intelligence_summaries"] = doc_intelligence_entries

        # --- PHASE 3 TASK 10 — SME pack assembly ---
        # Run the four-layer SME retrieval + merge + rerank + MMR +
        # pack-assembly pipeline when ENABLE_SME_RETRIEVAL is on for the
        # subscription. The assembled pack lands in
        # ``doc_context["sme_pack"]`` as a list[PackedItem] for the
        # rich-mode consumer to read in Phase 4. Phase 3 does NOT modify
        # the prompt surface — the reasoner stays unchanged.
        try:
            from src.config.feature_flags import (
                ENABLE_SME_RETRIEVAL,
                get_flag_resolver,
            )
            _sme_on = False
            try:
                _sme_on = get_flag_resolver().is_enabled(
                    subscription_id, ENABLE_SME_RETRIEVAL
                )
            except Exception:  # noqa: BLE001
                _sme_on = False
            if _sme_on:
                _sme_t0 = time.monotonic()
                sme_pack = self._build_sme_pack(
                    query=understanding.resolved_query,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    profile_domain=(
                        doc_context.get("profile_domain") or "generic"
                    ) if doc_context else "generic",
                    intent=understanding.task_type,
                    query_understanding={
                        "intent": understanding.task_type,
                        "entities": list(getattr(understanding, "entities", []) or []),
                    },
                )
                if doc_context is None:
                    doc_context = {}
                doc_context["sme_pack"] = sme_pack
                timing["sme_pack_ms"] = round(
                    (time.monotonic() - _sme_t0) * 1000, 1
                )
        except Exception:  # noqa: BLE001
            logger.debug("SME pack assembly skipped (non-fatal)", exc_info=True)

        # --- PHASE 4 — rich-mode pack summary + adapter resolution ---
        # Computes PackSummary + persona + shape decision BEFORE the reasoner
        # runs. Stored on ``doc_context`` so the reasoner (or a downstream
        # rewrite pass) can consume it. Gated behind ENABLE_RICH_MODE; when
        # OFF this block is a no-op and the reasoner sees today's compact
        # path unchanged.
        sme_pack_summary = None
        sme_response_shape = None
        try:
            from src.config.feature_flags import (
                ENABLE_RICH_MODE,
                get_flag_resolver,
            )
            from src.generation.pack_summary import PackSummary
            from src.generation.prompts import (
                resolve_response_shape,
            )
            from src.serving.model_router import FormatHint

            _rich_on = False
            try:
                _rich_on = get_flag_resolver().is_enabled(
                    subscription_id, ENABLE_RICH_MODE
                )
            except Exception:  # noqa: BLE001
                _rich_on = False

            if _rich_on:
                _pack_items = (doc_context or {}).get("sme_pack") or []
                sme_pack_summary = PackSummary.from_packed_items(_pack_items)
                sme_response_shape = resolve_response_shape(
                    intent=understanding.task_type,
                    format_hint=FormatHint.AUTO,
                    pack=sme_pack_summary,
                    enable_rich_mode=True,
                )
                if doc_context is None:
                    doc_context = {}
                doc_context["sme_pack_summary"] = sme_pack_summary
                doc_context["sme_response_shape"] = sme_response_shape.value
        except Exception:  # noqa: BLE001
            logger.debug("Rich-mode pack summary skipped (non-fatal)", exc_info=True)

        # --- REASON ---
        t0 = time.monotonic()
        # Enable thinking for cloud backends — adds reasoning depth.
        # Ollama Cloud qwen3.5:397b and Azure GPT-4o both support it.
        use_thinking = self._llm.backend in ("gemini", "openai", "azure", "azure_openai", "ollama")

        # Inject expert insights into doc_context if available
        profile_expertise = None
        try:
            from src.api.rag_state import get_app_state
            from src.intelligence_v2.expert_intelligence import filter_insights_for_query
            _app = get_app_state()
            if _app and _app.profile_expertise_cache:
                profile_expertise = _app.profile_expertise_cache.get(profile_id)
                if profile_expertise:
                    relevant_insights = filter_insights_for_query(profile_expertise, understanding.resolved_query)
                    if relevant_insights:
                        if doc_context is None:
                            doc_context = {}
                        doc_context["expert_insights"] = relevant_insights
        except Exception:
            logger.debug("Could not load expert insights", exc_info=True)

        # Single-GPU: skip sub-agent decomposition — parallel LLM calls
        # serialize on Ollama, causing timeouts. Use the reasoner directly.
        reason_result = self._reasoner.reason(
            query=understanding.resolved_query,
            task_type=understanding.task_type,
            output_format=understanding.output_format,
            evidence=evidence,
            doc_context=doc_context,
            conversation_history=conversation_history,
            use_thinking=use_thinking,
            profile_domain=profile_domain,
            kg_context=kg_context_text,
            profile_expertise=profile_expertise,
        )
        timing["reason_ms"] = round((time.monotonic() - t0) * 1000, 1)

        # --- PHASE 4 — recommendation grounding post-pass ---
        # For recommend-intent rich responses, strip any claim that doesn't
        # trace to a Recommendation Bank entry or an inline [doc_id:chunk_id]
        # citation. Preserves the 0.0 hallucination invariant. The post-pass
        # is a no-op for any other intent or when shape is not RICH.
        try:
            from src.generation.prompts import ResponseShape
            from src.generation.recommendation_grounding import (
                enforce_recommendation_grounding,
            )

            if (
                understanding.task_type == "recommend"
                and sme_response_shape is ResponseShape.RICH
                and sme_pack_summary is not None
            ):
                rewritten, _report = enforce_recommendation_grounding(
                    reason_result.text,
                    bank_entries=list(sme_pack_summary.bank_entries),
                )
                reason_result.text = rewritten
                logger.info(
                    "[PHASE4_GROUNDING] intent=recommend kept=%d dropped=%d",
                    _report.kept_count, _report.dropped_count,
                )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Recommendation grounding post-pass skipped (non-fatal)",
                exc_info=True,
            )

        # --- COMPOSE ---
        metadata = {
            "usage": reason_result.usage,
            "timing": timing,
            "profiles_searched": retrieval_result.profiles_searched,
        }
        # Phase 5 — thread URL-case metadata into the response so callers
        # see provenance warnings, the selected case, and the source URLs.
        if url_list:
            metadata["url_case"] = url_case.value
            metadata["url_sources"] = list(url_list)
            if ephemeral_warnings:
                metadata["url_warnings"] = ephemeral_warnings
        result = compose_response(
            text=reason_result.text,
            evidence=evidence,
            grounded=reason_result.grounded,
            task_type=understanding.task_type,
            metadata=metadata,
        )

        _evidence_sources = list({e.get("source_name", e.get("document_id", "?")) for e in evidence})
        logger.info(
            "[RAG_RESPONSE] profile=%s grounded=%s evidence_count=%d "
            "sources=%s task_type=%s response_len=%d timing=%s query=%r",
            profile_id, reason_result.grounded, len(evidence),
            _evidence_sources[:5], understanding.task_type,
            len(reason_result.text), timing, query[:80],
        )

        # --- FEEDBACK SIGNAL (non-blocking) ---
        try:
            from src.intelligence.feedback_tracker import FeedbackTracker
            redis_client = self._get_redis_client()
            if redis_client:
                tracker = FeedbackTracker(redis_client)
                tracker.record_query_signal(
                    profile_id=profile_id,
                    query=query,
                    response=reason_result.text,
                    evidence=evidence,
                    grounded=reason_result.grounded,
                    confidence=result.get("metadata", {}).get("confidence"),
                    task_type=understanding.task_type,
                )
        except Exception:
            logger.debug("Feedback signal recording skipped", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # Streaming public API
    # ------------------------------------------------------------------

    def handle_stream(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        user_id: str,
        session_id: str,
        conversation_history: Optional[List[Dict[str, str]]],
        *,
        agent_name: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream response tokens. UNDERSTAND+RETRIEVE run synchronously,
        then REASON streams tokens as they arrive from the LLM.

        Yields raw text chunks. The caller handles metadata/persistence.
        """
        if not subscription_id or not str(subscription_id).strip():
            raise ValueError("subscription_id is required")
        if not profile_id or not str(profile_id).strip():
            raise ValueError("profile_id is required")

        # --- UNDERSTAND (reuse full pipeline logic via non-streaming handle
        #     up to the REASON step, then stream from there) ---

        # Build the same pre-REASON state as handle()
        doc_intelligence = self._load_doc_intelligence(subscription_id)
        doc_intelligence_dict = {
            d.get("document_id", ""): d.get("intelligence", d)
            for d in doc_intelligence
        }

        kg_hints: Dict[str, Any] = {}
        if self.kg_query_service:
            try:
                query_entities = self.kg_query_service.extract_entities(query)
                if query_entities:
                    kg_result = self.kg_query_service.query(
                        subscription_id=subscription_id,
                        profile_id=profile_id,
                        domain_hint=None,
                        entities=query_entities,
                    )
                    kg_hints = {
                        "target_doc_ids": kg_result.doc_ids,
                        "target_chunk_ids": kg_result.chunk_ids,
                        "entities": query_entities,
                    }
            except Exception as exc:
                logger.debug("KG probe failed (non-fatal): %s", exc)

        trimmed_intel = []
        for d in doc_intelligence[:10]:
            trimmed = {
                "document_id": d.get("document_id", ""),
                "profile_id": d.get("profile_id", ""),
                "profile_name": d.get("profile_name", ""),
            }
            intel = d.get("intelligence") or {}
            trimmed["summary"] = (intel.get("summary") or "")[:200]
            trimmed["answerable_topics"] = (intel.get("answerable_topics") or [])[:5]
            trimmed["document_type"] = intel.get("document_type", "")
            trimmed_intel.append(trimmed)

        # Parallel UNDERSTAND + pre-fetch RETRIEVE
        prefetch_result = None
        understanding = None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _par:
                _intent_fut = _par.submit(
                    self._intent_analyzer.analyze,
                    query, subscription_id, profile_id, trimmed_intel,
                    conversation_history, kg_hints,
                )
                _prefetch_kwargs = {
                    "query": query,
                    "subscription_id": subscription_id,
                    "profile_ids": [profile_id],
                }
                _pf_doc_ids = kg_hints.get("target_doc_ids")
                if _pf_doc_ids:
                    _prefetch_kwargs["document_ids"] = _pf_doc_ids
                _prefetch_fut = _par.submit(
                    lambda kw=_prefetch_kwargs: self._retriever.retrieve(**kw),
                )
                understanding = _intent_fut.result(timeout=45.0)
                prefetch_result = _prefetch_fut.result(timeout=45.0)
        except Exception as _par_exc:
            logger.warning("Parallel UNDERSTAND+RETRIEVE failed: %s", _par_exc)
            if understanding is None:
                if isinstance(_par_exc, (concurrent.futures.TimeoutError, TimeoutError)):
                    understanding = self._intent_analyzer._safe_defaults(query)
                    self._intent_analyzer._enrich_relevant_documents(
                        understanding, query, trimmed_intel, kg_hints,
                    )
                else:
                    understanding = self._intent_analyzer.analyze(
                        query, subscription_id, profile_id, trimmed_intel,
                        conversation_history, kg_hints=kg_hints,
                    )

        if understanding.is_conversational:
            result = self._handle_conversational(query)
            yield result.get("response", "")
            return

        # --- RETRIEVE ---
        profile_ids = self._resolve_profile_scope(understanding, profile_id)
        document_ids: Optional[List[str]] = None
        if document_id:
            document_ids = [document_id]
        elif understanding.relevant_documents:
            doc_ids = [d.get("document_id", "") for d in understanding.relevant_documents if d.get("document_id")]
            if doc_ids:
                document_ids = doc_ids

        kg_doc_ids = kg_hints.get("target_doc_ids", [])
        if kg_doc_ids:
            if document_ids is None:
                document_ids = list(kg_doc_ids)
            else:
                existing = set(document_ids)
                for did in kg_doc_ids:
                    if did not in existing:
                        document_ids.append(did)

        enhanced_query = self._enhance_query(
            understanding.resolved_query, understanding.task_type,
            doc_intelligence, understanding.entities,
        )

        if prefetch_result is not None:
            if understanding.relevant_documents:
                target_doc_ids = {d.get("document_id") for d in understanding.relevant_documents if d.get("document_id")}
                if target_doc_ids:
                    prefetch_result.chunks = [c for c in prefetch_result.chunks if getattr(c, "document_id", None) in target_doc_ids]
            if document_id:
                prefetch_result.chunks = [c for c in prefetch_result.chunks if getattr(c, "document_id", None) == document_id]
            if len(prefetch_result.chunks) < 3 and (document_id or understanding.relevant_documents):
                retrieval_result = self._retriever.retrieve(enhanced_query, subscription_id, profile_ids, document_ids=document_ids)
            else:
                retrieval_result = prefetch_result
        else:
            retrieval_result = self._retriever.retrieve(enhanced_query, subscription_id, profile_ids, document_ids=document_ids)

        evidence_top_k = _EVIDENCE_TOP_K.get(understanding.task_type, 15)
        reranked = rerank_chunks(
            understanding.resolved_query, retrieval_result.chunks,
            top_k=evidence_top_k, cross_encoder=self._cross_encoder,
        )
        evidence, doc_context = build_context(reranked, doc_intelligence_dict)

        if kg_hints.get("target_doc_ids") and self.kg_query_service:
            try:
                kg_entities = kg_hints.get("entities", [])
                if kg_entities:
                    kg_context_items = [
                        f"{e.get('value', '')} ({e.get('type', '')})"
                        for e in (kg_entities if isinstance(kg_entities, list) else [])
                        if isinstance(e, dict) and e.get("value")
                    ]
                    if kg_context_items:
                        existing = doc_context.get("entities") or []
                        for kc in kg_context_items[:5]:
                            if kc not in existing:
                                existing.append(kc)
                        doc_context["entities"] = existing[:25]
            except Exception:
                pass

        # doc_index / doc_intelligence enrichment
        try:
            from qdrant_client.models import Filter as _QFilter, FieldCondition as _QFC, MatchValue as _QMV
            from src.api.vector_store import build_collection_name
            _collection = build_collection_name(subscription_id)
            _intel_points, _ = self._qdrant.scroll(
                collection_name=_collection,
                scroll_filter=_QFilter(must=[
                    _QFC(key="profile_id", match=_QMV(value=str(profile_id))),
                    _QFC(key="resolution", match=_QMV(value="doc_intelligence")),
                ]),
                limit=200, with_payload=True, with_vectors=False,
            )
            doc_intelligence_entries = [
                (p.payload or {}).get("canonical_text", "")
                for p in _intel_points if (p.payload or {}).get("canonical_text")
            ]
            if doc_intelligence_entries:
                doc_context["doc_intelligence_summaries"] = doc_intelligence_entries
        except Exception:
            pass

        # KG context
        profile_domain = "general"
        kg_context_text = ""
        try:
            from src.intelligence.hot_cache import get_profile_domain, lookup_entities, get_document_facts, get_top_relationships
            redis_client = self._get_redis_client()
            if redis_client:
                profile_domain = get_profile_domain(redis_client, profile_id)
                kg_parts = []
                query_words = [w for w in understanding.resolved_query.split() if w.lower() not in _STOPWORDS and len(w) > 2]
                cached_entities = lookup_entities(redis_client, profile_id, query_words)
                if cached_entities:
                    entity_lines = [f"- {e['name']} ({e.get('type', 'unknown')}): {e.get('context', '')}" for e in cached_entities[:8]]
                    if entity_lines:
                        kg_parts.append("Known entities:\n" + "\n".join(entity_lines))
                evidence_doc_ids = list({e.get("document_id", "") for e in evidence if e.get("document_id")})
                for did in evidence_doc_ids[:3]:
                    facts = get_document_facts(redis_client, profile_id, did, max_facts=5)
                    for f in facts:
                        kg_parts.append(f"- Fact: {f.get('statement', '')}")
                rels = get_top_relationships(redis_client, profile_id, max_results=5)
                for r in rels:
                    kg_parts.append(f"- Relationship: {r.get('subject', '')} {r.get('relation', '')} {r.get('object', '')}")
                if kg_parts:
                    kg_context_text = "\n".join(kg_parts[:20])
        except ImportError:
            pass
        except Exception:
            pass

        use_thinking = self._llm.backend in ("gemini", "openai", "azure", "azure_openai", "ollama")

        # --- STREAM REASON ---
        yield from self._reasoner.reason_stream(
            query=understanding.resolved_query,
            task_type=understanding.task_type,
            output_format=understanding.output_format,
            evidence=evidence,
            doc_context=doc_context,
            conversation_history=conversation_history,
            use_thinking=use_thinking,
            profile_domain=profile_domain,
            kg_context=kg_context_text,
        )

    # ------------------------------------------------------------------
    # Redis client helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_redis_client():
        """Get the Redis client from app state or create one."""
        try:
            from src.api.rag_state import get_app_state
            app_state = get_app_state()
            if app_state and hasattr(app_state, "redis_client"):
                return app_state.redis_client
        except Exception:
            pass
        try:
            import redis
            from src.api.config import Config
            url = getattr(Config.Redis, "URL", None) or getattr(Config.Redis, "HOST", "localhost")
            return redis.Redis.from_url(url) if "://" in str(url) else redis.Redis(host=url)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Conversational handler
    # ------------------------------------------------------------------

    def _handle_conversational(self, query: str) -> Dict[str, Any]:
        """Return a short friendly response for conversational queries."""
        if _GREETING_RE.search(query):
            text = _CONVERSATIONAL_RESPONSES["greeting"]
        elif _FAREWELL_RE.search(query):
            text = _CONVERSATIONAL_RESPONSES["farewell"]
        elif _THANKS_RE.search(query):
            text = _CONVERSATIONAL_RESPONSES["thanks"]
        elif _META_RE.search(query):
            text = _CONVERSATIONAL_RESPONSES["meta"]
        else:
            text = _CONVERSATIONAL_RESPONSES["greeting"]

        return compose_response(
            text=text,
            evidence=[],
            grounded=False,
            task_type="conversational",
        )

    # ------------------------------------------------------------------
    # Complex query handler — spawns sub-agents
    # ------------------------------------------------------------------

    def _handle_complex(
        self,
        understanding: QueryUnderstanding,
        evidence: List[Dict[str, Any]],
        doc_context: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> ReasonerResult:
        """Split evidence across sub-tasks and run sub-agents in parallel."""
        sub_tasks = (understanding.sub_tasks or [])[:self.MAX_SUBAGENTS]
        if not sub_tasks:
            return self._reasoner.reason(
                query=understanding.resolved_query,
                task_type=understanding.task_type,
                output_format=understanding.output_format,
                evidence=evidence,
                doc_context=doc_context,
                conversation_history=conversation_history,
                use_thinking=False,
            )

        # Partition evidence round-robin
        partitions: List[List[Dict[str, Any]]] = [[] for _ in sub_tasks]
        for idx, item in enumerate(evidence):
            partitions[idx % len(sub_tasks)].append(item)

        agents = [
            DynamicSubAgent(
                llm_gateway=self._llm,
                role=task,
                evidence=partition,
                doc_context=doc_context,
            )
            for task, partition in zip(sub_tasks, partitions)
        ]

        results = []
        max_workers = min(len(agents), 3)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(agent.execute): agent for agent in agents}
            for future in as_completed(futures, timeout=self.SUBAGENT_TIMEOUT):
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.warning("Sub-agent future failed: %s", exc)

        synthesis_evidence = []
        for idx, r in enumerate(results, start=1):
            if r.success and r.text:
                synthesis_evidence.append({
                    "source_index": idx,
                    "source_name": f"sub-agent: {r.task[:50]}",
                    "section": r.task,
                    "page": 0,
                    "text": r.text,
                    "score": 1.0,
                    "document_id": "",
                    "profile_id": "",
                    "chunk_id": f"subagent-{idx}",
                })

        return self._reasoner.reason(
            query=understanding.resolved_query,
            task_type=understanding.task_type,
            output_format=understanding.output_format,
            evidence=synthesis_evidence if synthesis_evidence else evidence,
            doc_context=doc_context,
            conversation_history=conversation_history,
            use_thinking=False,
        )

    # ------------------------------------------------------------------
    # Profile scope resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_profile_scope(
        understanding: QueryUnderstanding,
        requesting_profile_id: str,
    ) -> List[str]:
        """Determine which profile IDs to search."""
        profile_ids = {requesting_profile_id}
        if understanding.cross_profile and understanding.relevant_documents:
            for doc in understanding.relevant_documents:
                pid = doc.get("profile_id")
                if pid:
                    profile_ids.add(pid)
        return list(profile_ids)

    # ------------------------------------------------------------------
    # Query enhancement for better retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _enhance_query(
        query: str,
        task_type: str,
        doc_intelligence: List[Dict[str, Any]],
        entities: List[str],
    ) -> str:
        """Expand the query with task synonyms, entities, and topic keywords.

        Adds relevant terms to improve dense retrieval recall without changing
        the semantic meaning. The original query always leads.
        """
        expansion_terms: Set[str] = set()
        query_lower = query.lower()

        # 1. Add task-type synonyms that aren't already in the query
        synonyms = _TASK_SYNONYMS.get(task_type, [])
        for syn in synonyms:
            if syn not in query_lower:
                expansion_terms.add(syn)

        # 2. Add entities from intent analysis (already extracted by LLM)
        for entity in entities[:5]:
            if entity.lower() not in query_lower:
                expansion_terms.add(entity)

        # 3. Mine doc_intelligence for matching topic keywords
        query_words = set(query_lower.split()) - _STOPWORDS
        for doc in doc_intelligence:
            intel = doc.get("intelligence") or {}
            topics: List[str] = intel.get("answerable_topics") or []
            for topic in topics:
                topic_words = set(topic.lower().split()) - _STOPWORDS
                overlap = query_words & topic_words
                if len(overlap) >= 2:
                    # This topic is relevant — add non-overlapping keywords
                    new_words = topic_words - query_words - _STOPWORDS
                    for w in list(new_words)[:3]:
                        if len(w) > 2:
                            expansion_terms.add(w)

            # 4. Add matching entity names from doc intelligence
            doc_entities = intel.get("entities") or []
            for ent in doc_entities[:10]:
                ent_name = ent.get("name", str(ent)) if isinstance(ent, dict) else str(ent)
                ent_lower = ent_name.lower()
                ent_words = set(ent_lower.split()) - _STOPWORDS
                if ent_words & query_words and ent_lower not in query_lower:
                    expansion_terms.add(ent_name)

        # Cap expansion to avoid noise
        expansion_list = list(expansion_terms)[:8]
        if not expansion_list:
            return query

        enhanced = f"{query} {' '.join(expansion_list)}"
        logger.debug("Query enhanced: '%s' -> '%s'", query[:60], enhanced[:120])
        return enhanced

    # ------------------------------------------------------------------
    # Document intelligence loader
    # ------------------------------------------------------------------

    def _load_doc_intelligence(self, subscription_id: str) -> List[Dict[str, Any]]:
        """Load document intelligence metadata from MongoDB."""
        try:
            cursor = self._mongodb.find(
                {
                    "$or": [
                        {"subscription_id": subscription_id},
                        {"subscription": subscription_id},
                        {"subscriptionId": subscription_id},
                    ],
                    "intelligence": {"$exists": True, "$ne": None},
                },
                {
                    "document_id": 1,
                    "profile_id": 1,
                    "profile": 1,
                    "profile_name": 1,
                    "intelligence.summary": 1,
                    "intelligence.entities": 1,
                    "intelligence.answerable_topics": 1,
                    "intelligence.key_facts": 1,
                    "intelligence.document_type": 1,
                },
            )
            results = []
            for doc in cursor:
                # Normalize field names for connector docs
                if "profile" in doc and "profile_id" not in doc:
                    doc["profile_id"] = doc["profile"]
                results.append(doc)
            return results
        except Exception:
            logger.exception("Failed to load doc intelligence for subscription=%s", subscription_id)
            return []

    # ------------------------------------------------------------------
    # Phase 4 — rich-mode prompt building (test seam + production path)
    # ------------------------------------------------------------------

    async def _resolve_profile_domain(
        self, subscription_id: str, profile_id: str
    ) -> str:
        """Resolve ``profile_domain`` for adapter lookup.

        The production path reads from the profile record in MongoDB (the
        Phase 1 ingest writes the domain there). Tests patch this method.
        Falls back to ``"generic"`` on any error so rich-mode wiring never
        blows up a request.
        """
        try:
            from src.api.document_status import get_profile_record
            rec = get_profile_record(subscription_id, profile_id)
            if rec:
                return rec.get("profile_domain") or "generic"
        except Exception:  # noqa: BLE001
            logger.debug(
                "profile_domain lookup failed for sub=%s prof=%s",
                subscription_id, profile_id, exc_info=True,
            )
        return "generic"

    @staticmethod
    def _build_compact_prompt(classified: Any, pack_summary: Any) -> str:
        """Legacy-compact prompt — a no-op stand-in for the pre-Phase-4 path.

        The real compact reasoner path is driven by
        :func:`src.generation.prompts.build_reason_prompt`; the Phase 4
        wiring test only needs a distinct return value so tests can prove
        the resolver chose the compact branch without invoking the full
        reasoner pipeline.
        """
        return f"COMPACT:{getattr(classified, 'query_text', '')}"

    async def _build_prompt_for_test(
        self,
        *,
        classified: Any,
        pack_summary: Any,
        subscription_id: str,
        profile_id: str,
    ) -> str:
        """Rich-mode wiring seam used by tests.

        Production's :meth:`handle` embeds the same resolver → adapter →
        template dispatch inline. This method exists so the wiring can be
        exercised in isolation without booting the full pipeline.

        Logic:

        1. Flag OFF or explicit compact override → compact prompt.
        2. Analytical / borderline intent + thin pack → honest-compact.
        3. Rich-shaped intent → load adapter, build persona, dispatch to
           the rich template for ``analyze`` / ``diagnose`` / ``recommend``.
           ``investigate`` / ``overview`` route through the analyze template
           per spec §8.
        """
        from src.generation.prompts import (
            AnalyzePromptInputs,
            DiagnosePromptInputs,
            RecommendPromptInputs,
            ResponseShape,
            build_analyze_rich_prompt,
            build_diagnose_rich_prompt,
            build_honest_compact_prompt,
            build_recommend_rich_prompt,
            persona_bundle_from_adapter,
            resolve_response_shape,
        )

        enable_rich = await _is_rich_mode_enabled(subscription_id)
        shape = resolve_response_shape(
            intent=classified.intent,
            format_hint=classified.format_hint,
            pack=pack_summary,
            enable_rich_mode=enable_rich,
        )

        if shape is ResponseShape.COMPACT:
            return self._build_compact_prompt(classified, pack_summary)

        domain = await self._resolve_profile_domain(subscription_id, profile_id)
        adapter = await _load_adapter(subscription_id, domain)

        if shape is ResponseShape.HONEST_COMPACT:
            return build_honest_compact_prompt(
                query_text=classified.query_text,
                pack_summary=pack_summary,
            )

        # Investigate / overview route through the analyze template (spec §8).
        template_intent = (
            "analyze"
            if classified.intent in ("investigate", "overview")
            else classified.intent
        )

        # Rich shape for a borderline intent (compare / summarize / etc.)
        # falls back to compact — they don't have dedicated rich templates.
        if template_intent not in ("analyze", "diagnose", "recommend"):
            return self._build_compact_prompt(classified, pack_summary)

        persona = persona_bundle_from_adapter(adapter, intent=template_intent)
        cap = getattr(adapter.output_caps, template_intent, 1200) or 1200
        pack_tokens = (
            adapter.retrieval_caps.max_pack_tokens.get(template_intent, 6000)
            if hasattr(adapter.retrieval_caps, "max_pack_tokens")
            else 6000
        )

        # Build evidence / insights / bank entries directly off PackSummary
        # per ERRATA §10 — no private helper methods required.
        evidence = []
        for item in pack_summary.evidence_items:
            if not item.provenance:
                continue
            doc_id, chunk_id = item.provenance[0]
            evidence.append(
                {"doc_id": doc_id, "chunk_id": chunk_id, "text": item.text}
            )
        insights = [
            {
                "type": (item.metadata or {}).get("insight_type", "insight"),
                "narrative": item.text,
            }
            for item in pack_summary.insights
        ]

        if template_intent == "analyze":
            return build_analyze_rich_prompt(
                AnalyzePromptInputs(
                    query_text=classified.query_text,
                    persona_role=persona.role,
                    persona_voice=persona.voice,
                    grounding_rules=persona.grounding_rules,
                    pack_tokens=pack_tokens,
                    output_cap_tokens=cap,
                    evidence_items=evidence,
                    insight_refs=insights,
                    domain=domain,
                )
            )
        if template_intent == "diagnose":
            hits = [
                {
                    "symptom": (item.metadata or {}).get(
                        "symptom", item.text[:120]
                    ),
                    "doc_id": item.provenance[0][0] if item.provenance else "",
                    "chunk_id": item.provenance[0][1] if item.provenance else "",
                    "rank": i + 1,
                }
                for i, item in enumerate(pack_summary.insights)
            ]
            return build_diagnose_rich_prompt(
                DiagnosePromptInputs(
                    query_text=classified.query_text,
                    persona_role=persona.role,
                    persona_voice=persona.voice,
                    grounding_rules=persona.grounding_rules,
                    pack_tokens=pack_tokens,
                    output_cap_tokens=cap,
                    evidence_items=evidence,
                    diagnostic_hits=hits,
                    domain=domain,
                )
            )
        if template_intent == "recommend":
            return build_recommend_rich_prompt(
                RecommendPromptInputs(
                    query_text=classified.query_text,
                    persona_role=persona.role,
                    persona_voice=persona.voice,
                    grounding_rules=persona.grounding_rules,
                    pack_tokens=pack_tokens,
                    output_cap_tokens=cap,
                    evidence_items=evidence,
                    bank_entries=list(pack_summary.bank_entries),
                    domain=domain,
                )
            )
        return self._build_compact_prompt(classified, pack_summary)

    async def _apply_recommend_grounding(
        self,
        *,
        response_text: str,
        classified: Any,
        shape: Any,
        pack_summary: Any,
    ) -> str:
        """Run the recommendation post-pass for recommend-intent rich output.

        Called from the production streaming / non-streaming response path
        once the LLM has produced final text. A no-op for any other intent
        or shape so the caller can invoke it unconditionally.
        """
        from src.generation.prompts import ResponseShape
        from src.generation.recommendation_grounding import (
            enforce_recommendation_grounding,
        )

        intent = getattr(classified, "intent", "")
        if intent != "recommend" or shape is not ResponseShape.RICH:
            return response_text
        rewritten, _report = enforce_recommendation_grounding(
            response_text,
            bank_entries=list(getattr(pack_summary, "bank_entries", ()) or ()),
        )
        return rewritten
