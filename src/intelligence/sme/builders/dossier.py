"""SMEDossierBuilder — per-section LLM synthesis with fail-closed verification.

Phase 2 implementation. Phase 1 shipped the skeleton returning ``[]``; this
module now performs real LLM-driven per-section synthesis using the adapter's
persona, dossier section weights, and prompt template.

Contract:

* Builder consumes a :class:`BuilderContext` exposing ``iter_profile_chunks``.
* For each dossier section declared in ``adapter.dossier.section_weights`` we
  emit one :class:`ArtifactItem` carrying:

  * ``text`` = section narrative (ERRATA §3 — SMEVerifier operates on
    ``.text``; the builder never needs a separate ``.narrative`` attribute
    because the unified contract already stores the narrative there).
  * ``evidence`` = ≥1 :class:`EvidenceRef` cited by the LLM (check 1 +
    evidence-validity pre-pass).
  * ``confidence`` parsed from the LLM response.
  * ``metadata`` = ``{section, section_weight, entity_refs}`` so retrieval
    layers can filter by section.

* Drops are logged through the injected trace writer via a structured
  ``append({...})`` call (ERRATA §5).

LLM injection uses a ``LLMClient`` structural protocol so Phase 2 tests can
plug in a MagicMock; production wiring injects the existing DocWain gateway
wrapper which handles its own per-op safety timeout. No wall-clock timeout is
added at this layer (memory rule: no internal timeouts).
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Protocol

from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef
from src.intelligence.sme.builders._base import ArtifactBuilder


class LLMClient(Protocol):
    """Structural contract every Phase 2 builder uses for LLM calls.

    Implementations SHALL NOT impose a DocWain-layer timeout; the underlying
    transport (httpx / gateway) owns per-op safety. Returns the raw string
    body of the assistant message — parsing responsibility lives in the
    builder so per-builder JSON schema drift is contained.
    """

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        trace_tag: str,
        adapter_version: str,
    ) -> str: ...


class _TraceSink(Protocol):
    """Minimal trace surface used by Phase 2 builders (ERRATA §5)."""

    def append(self, event: dict[str, Any]) -> None: ...


class SMEDossierBuilder(ArtifactBuilder):
    """Builder for the per-profile domain-aware dossier.

    Construction takes the standard Phase 1 ``ctx`` (profile chunk reader) plus
    the Phase 2 dependencies: ``llm``, ``trace``. Tests that want to assert on
    every section's LLM call inject a MagicMock for ``llm`` and a MagicMock
    trace sink; the builder's unit tests do not reach the verifier or storage
    — the orchestrator owns that wiring (Task 7).
    """

    artifact_type = "dossier"

    def __init__(
        self,
        *,
        ctx,
        llm: LLMClient,
        trace: _TraceSink,
    ) -> None:
        super().__init__(ctx=ctx)
        self._llm = llm
        self._trace = trace

    def _synthesize(
        self, *, subscription_id: str, profile_id: str, adapter, version: int
    ) -> list[ArtifactItem]:
        chunks = list(self._ctx.iter_profile_chunks(subscription_id, profile_id))
        evidence_pack = "\n".join(
            f"[{c['doc_id']}#{c['chunk_id']}] {c.get('text', '')}" for c in chunks
        )
        persona = adapter.persona
        system_prompt = (
            f"You are a {persona.role}. Voice: {persona.voice}. "
            "Return ONLY a single JSON object with keys: section, narrative, "
            "evidence (list of {doc_id, chunk_id, quote?}), confidence "
            "(0.0-1.0), entity_refs (list of strings). Cite every factual "
            "claim with an evidence entry."
        )
        grounding_rules = "\n".join(f"- {r}" for r in persona.grounding_rules)
        items: list[ArtifactItem] = []
        for section_name, weight in adapter.dossier.section_weights.items():
            user_prompt = (
                f"Write the '{section_name}' section of the dossier for a "
                f"{persona.role} audience. Grounding rules:\n{grounding_rules}\n\n"
                f"Section weight: {weight:.2f}. Prompt template hint: "
                f"{adapter.dossier.prompt_template}.\n\nEVIDENCE:\n{evidence_pack}"
            )
            trace_tag = f"dossier:{subscription_id}:{profile_id}:{section_name}"
            try:
                raw = self._llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    trace_tag=trace_tag,
                    adapter_version=adapter.version,
                )
            except Exception as exc:  # noqa: BLE001 — LLM call failure is logged, section skipped
                self._trace.append(
                    {
                        "stage": "builder_llm_error",
                        "builder": self.artifact_type,
                        "section": section_name,
                        "error": str(exc),
                    }
                )
                continue

            parsed = self._parse_section(raw, section_name)
            if parsed is None:
                continue

            item = self._make_item(
                subscription_id=subscription_id,
                profile_id=profile_id,
                section=section_name,
                weight=weight,
                parsed=parsed,
            )
            if item is None:
                # Parsing succeeded but didn't yield ≥1 evidence — skip.
                self._trace.append(
                    {
                        "stage": "builder_no_evidence",
                        "builder": self.artifact_type,
                        "section": section_name,
                    }
                )
                continue
            items.append(item)
        return items

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_section(self, raw: str, section: str) -> dict[str, Any] | None:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            self._trace.append(
                {
                    "stage": "builder_parse_failure",
                    "builder": self.artifact_type,
                    "section": section,
                    "error": str(exc),
                }
            )
            return None
        if not isinstance(data, dict):
            self._trace.append(
                {
                    "stage": "builder_parse_failure",
                    "builder": self.artifact_type,
                    "section": section,
                    "error": "llm response not a JSON object",
                }
            )
            return None
        return data

    def _make_item(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        section: str,
        weight: float,
        parsed: dict[str, Any],
    ) -> ArtifactItem | None:
        narrative = str(parsed.get("narrative", "")).strip()
        if not narrative:
            return None
        raw_evidence = parsed.get("evidence") or []
        evidence: list[EvidenceRef] = []
        for ev in raw_evidence:
            if not isinstance(ev, dict):
                continue
            doc_id = str(ev.get("doc_id") or "")
            chunk_id = str(ev.get("chunk_id") or "")
            if not doc_id or not chunk_id:
                continue
            quote = ev.get("quote")
            evidence.append(
                EvidenceRef(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    quote=str(quote) if quote is not None else None,
                )
            )
        if not evidence:
            return None
        confidence = float(parsed.get("confidence", 0.6))
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0
        entity_refs = [
            str(e) for e in (parsed.get("entity_refs") or []) if isinstance(e, str)
        ]
        return ArtifactItem(
            item_id=f"dossier:{subscription_id}:{profile_id}:{section}:{uuid.uuid4().hex[:8]}",
            artifact_type="dossier",
            subscription_id=subscription_id,
            profile_id=profile_id,
            text=narrative,
            evidence=evidence,
            confidence=confidence,
            inference_path=[],
            domain_tags=[],
            metadata={
                "section": section,
                "section_weight": weight,
                "entity_refs": entity_refs,
            },
        )
