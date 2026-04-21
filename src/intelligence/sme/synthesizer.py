"""SMESynthesizer orchestrator (spec §8, ERRATA §1/§2/§3/§5).

The Phase 1 implementation is the production control flow with builders that
return ``[]``. Phase 2 fills the builder bodies without touching this file.
The orchestrator's contract:

1. Open the synthesis trace writer with ``synthesis_id = f"{sub}:{prof}:{version}"``.
2. Resolve the adapter via :class:`AdapterLoader`. ``content_hash`` and
   ``version`` are read directly off the returned :class:`Adapter` instance
   (ERRATA §1).
3. For each (artifact_type, builder) pair in ``deps.builders`` in insertion
   order: run ``builder.build(...)``, call ``verifier.verify_batch(items, ctx)``,
   persist accepted items via ``storage.persist_items(...)``, emit trace
   events for every drop and each builder's completion summary.
4. Always close the trace writer, even on exception.

The control flow intentionally avoids a master abort — a failing builder does
not halt subsequent builders, because Phase 2's real bodies should still
produce partial output when one rule breaks. Builder-level exception handling
lands in Phase 2; Phase 1's stubs cannot raise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.intelligence.sme.adapter_loader import AdapterLoader
from src.intelligence.sme.adapter_schema import Adapter
from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.storage import SMEArtifactStorage
from src.intelligence.sme.trace import SynthesisTraceWriter
from src.intelligence.sme.verifier import SMEVerifier, VerifierContext


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
            for artifact_type, builder in deps.builders.items():
                items = builder.build(
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    adapter=adapter,
                    version=synthesis_version,
                )
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
