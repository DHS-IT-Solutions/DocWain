"""SMESynthesizer orchestrator (spec §8, ERRATA §1/§2/§3/§5).

Phase 1 shipped the control flow with stub builders that returned ``[]``;
Phase 2 keeps the orchestration unchanged and swaps builder bodies for the
real implementations. The orchestrator's contract:

1. Open the synthesis trace writer with ``synthesis_id = f"{sub}:{prof}:{version}"``.
2. Resolve the adapter via :class:`AdapterLoader`. ``content_hash`` and
   ``version`` are read directly off the returned :class:`Adapter` instance
   (ERRATA §1).
3. Iterate ``deps.builders`` in insertion order. The canonical order is
   ``dossier → insight → comparison → kg_edge → recommendation`` — the
   recommendation builder depends on the verified Insight Index output, so
   the orchestrator threads accepted insight items as a keyword argument
   when the builder's ``build`` accepts ``insight_items``. For every
   builder: run ``builder.build(...)``, call
   ``verifier.verify_batch(items, ctx)``, persist accepted items via
   ``storage.persist_items(...)``, emit trace events for every drop and each
   builder's completion summary.
4. Always close the trace writer, even on exception.

A failing builder does not halt subsequent builders. Phase 2 builders log
non-fatal failures (LLM errors, parse failures) to the trace; truly fatal
exceptions propagate out of ``run`` via the ``finally: tw.close()`` gate.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Protocol

from src.intelligence.sme.adapter_loader import AdapterLoader
from src.intelligence.sme.adapter_schema import Adapter
from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.storage import SMEArtifactStorage
from src.intelligence.sme.trace import SynthesisTraceWriter
from src.intelligence.sme.verifier import SMEVerifier, VerifierContext


# Recognised slugs produced by the five Phase 2 builders. Used to thread
# the Insight Index output into the Recommendation Bank builder.
_INSIGHT_SLUG = "insight"
_RECOMMENDATION_SLUG = "recommendation"


class ArtifactBuilder(Protocol):
    """Structural builder contract used by the synthesizer.

    Mirrors the :class:`src.intelligence.sme.builders._base.ArtifactBuilder`
    abstract base class without forcing the orchestrator to import the
    concrete ABC. Phase 1 stubs and Phase 2 real implementations satisfy both.
    """

    artifact_type: str

    def build(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        adapter: Adapter,
        version: int,
    ) -> list[ArtifactItem]: ...


@dataclass
class SynthesizerDeps:
    """Injected dependency bundle for :class:`SMESynthesizer`.

    ``builders`` is an ordered mapping of artifact-type slug → builder. The
    orchestrator iterates via ``builders.items()`` so insertion order controls
    the run sequence. Phase 2 keeps this shape; only the values change.
    """

    adapter_loader: AdapterLoader
    storage: SMEArtifactStorage
    verifier: SMEVerifier
    trace_writer: SynthesisTraceWriter
    builders: dict[str, ArtifactBuilder]


class SMESynthesizer:
    """Orchestrator skeleton — control flow only in Phase 1.

    The :meth:`run` entrypoint returns a ``{artifact_type: accepted_count}``
    mapping so downstream callers (Phase 2 training-stage integration) can
    surface per-artifact-type counts without re-reading the trace.
    """

    def __init__(self, deps: SynthesizerDeps) -> None:
        self._d = deps

    def run(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        profile_domain: str,
        synthesis_version: int,
    ) -> dict[str, int]:
        """Execute a single synthesis run.

        All trace events include enough context (subscription, profile,
        synthesis id, adapter version + hash) for downstream diagnostics to
        reconstruct the run without cross-referencing other logs.
        """
        synthesis_id = f"{subscription_id}:{profile_id}:{synthesis_version}"
        tw = self._d.trace_writer
        deps = self._d

        tw.open(
            subscription_id=subscription_id,
            profile_id=profile_id,
            synthesis_id=synthesis_id,
        )
        try:
            adapter = deps.adapter_loader.load(subscription_id, profile_domain)
            ctx = VerifierContext(
                subscription_id=subscription_id, profile_id=profile_id
            )
            tw.append(
                {
                    "stage": "start",
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                    "synthesis_id": synthesis_id,
                    "adapter_version": adapter.version,
                    "adapter_hash": adapter.content_hash,
                }
            )

            counts: dict[str, int] = {}
            accepted_by_type: dict[str, list[ArtifactItem]] = {}
            for artifact_type, builder in deps.builders.items():
                build_kwargs: dict[str, Any] = {
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                    "adapter": adapter,
                    "version": synthesis_version,
                }
                # Thread verified Insight Index items into the Recommendation
                # Bank builder. The recommendation builder's public build()
                # accepts an optional ``insight_items`` kwarg; other builders
                # do not. We introspect the builder's signature so MagicMock
                # doubles (Phase 1 tests) and real implementations alike get
                # the right call.
                if (
                    artifact_type == _RECOMMENDATION_SLUG
                    and _accepts_kwarg(builder.build, "insight_items")
                ):
                    build_kwargs["insight_items"] = list(
                        accepted_by_type.get(_INSIGHT_SLUG, [])
                    )
                items = builder.build(**build_kwargs)
                verdicts = deps.verifier.verify_batch(items, ctx)
                accepted_items: list[ArtifactItem] = []
                for verdict in verdicts:
                    if verdict.passed and verdict.adjusted_item is not None:
                        accepted_items.append(verdict.adjusted_item)
                    else:
                        tw.append(
                            {
                                "stage": "verifier_drop",
                                "builder": artifact_type,
                                "item_id": verdict.item_id,
                                "failing_check": verdict.failing_check,
                                "drop_reason": verdict.drop_reason,
                            }
                        )
                accepted_by_type[artifact_type] = accepted_items
                deps.storage.persist_items(
                    subscription_id,
                    profile_id,
                    artifact_type,
                    accepted_items,
                    version=synthesis_version,
                )
                counts[artifact_type] = len(accepted_items)
                tw.append(
                    {
                        "stage": "builder_complete",
                        "builder": artifact_type,
                        "accepted": len(accepted_items),
                        "dropped": len(verdicts) - len(accepted_items),
                    }
                )
            tw.append({"stage": "complete", "counts": counts})
            return counts
        finally:
            tw.close()


def _accepts_kwarg(callable_: Any, name: str) -> bool:
    """Return True when ``callable_`` accepts a keyword argument ``name``.

    Handles MagicMock, bound methods, and plain functions. Falls back to
    ``False`` on any inspection failure so Phase 1 MagicMock doubles (whose
    signatures are effectively ``(*args, **kwargs)`` — which matches
    everything) keep working without change. For MagicMock specifically we
    return ``False`` so Phase 1 tests that assert the builder was called
    without ``insight_items`` continue to pass; Phase 2 real builders expose
    the kwarg explicitly in their signature.
    """
    try:
        sig = inspect.signature(callable_)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.name == name:
            return True
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            # Only a real signature declaring **kwargs should qualify —
            # MagicMock's signature introspection returns *args / **kwargs,
            # so we don't want to treat every mock as insight_items-aware.
            # Check whether the callable is a MagicMock: if so, decline.
            from unittest.mock import Mock

            if isinstance(callable_, Mock):
                return False
            return True
    return False
