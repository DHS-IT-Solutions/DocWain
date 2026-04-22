"""InsightIndexBuilder — per-detector LLM synthesis of typed insights.

Phase 2 implementation. Phase 1 shipped the skeleton returning ``[]``; this
module now iterates the adapter's ``insight_detectors`` list and generates one
:class:`ArtifactItem` per detected insight. Detector rule kinds are passed
through to the LLM verbatim — the adapter YAML (spec §5) owns the vocabulary;
the builder's job is to let the model materialize the detection.

Each detector run is pooled into a single result list before the orchestrator
runs the verifier (verifier's check 5 — contradiction detection — needs the
full pool, so the builder deliberately returns a flat list without
per-detector batching).

LLM injection: builders depend on a structural ``LLMClient`` protocol defined
in :mod:`src.intelligence.sme.builders.dossier`; tests inject a MagicMock. No
wall-clock timeout at the DocWain layer.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef
from src.intelligence.sme.builders._base import ArtifactBuilder
from src.intelligence.sme.builders.dossier import LLMClient, _TraceSink


_INSIGHT_TYPES = frozenset(
    {"trend", "anomaly", "gap", "risk", "opportunity", "conflict"}
)


class InsightIndexBuilder(ArtifactBuilder):
    """Builder for per-profile typed insight items.

    ``domain_tags`` on each emitted item always includes the detector's
    insight type (``trend``/``anomaly``/…) so downstream retrieval (Phase 3)
    can filter by type without inspecting ``metadata``. The detector type is
    also mirrored in ``metadata["detector"]`` so the trace + debugging paths
    carry the originating adapter configuration.
    """

    artifact_type = "insight"

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
            "Return ONLY a JSON object with key 'items' whose value is an "
            "array of objects each with keys: type, narrative, evidence (list "
            "of {doc_id, chunk_id, quote?}), confidence (0.0-1.0), domain_tags "
            "(list), temporal_scope (optional string), entity_refs (list)."
        )
        pooled: list[ArtifactItem] = []
        for detector in adapter.insight_detectors:
            detector_type = detector.type
            rule = detector.rule
            params = dict(detector.params or {})
            user_prompt = (
                f"Detector: type={detector_type}, rule={rule}, params={params}. "
                "Enumerate insights from the evidence that satisfy this "
                "detector. Cite each claim with at least one evidence entry. "
                f"\n\nEVIDENCE:\n{evidence_pack}"
            )
            trace_tag = (
                f"insight_index:{subscription_id}:{profile_id}:{detector_type}:{rule}"
            )
            try:
                raw = self._llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    trace_tag=trace_tag,
                    adapter_version=adapter.version,
                )
            except Exception as exc:  # noqa: BLE001
                self._trace.append(
                    {
                        "stage": "builder_llm_error",
                        "builder": self.artifact_type,
                        "detector": detector_type,
                        "rule": rule,
                        "error": str(exc),
                    }
                )
                continue
            detector_items = self._parse_items(raw, detector_type, rule)
            for raw_item in detector_items:
                item = self._make_item(
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    detector_type=detector_type,
                    rule=rule,
                    raw=raw_item,
                )
                if item is not None:
                    pooled.append(item)
        return pooled

    # ------------------------------------------------------------------
    def _parse_items(
        self, raw: str, detector_type: str, rule: str
    ) -> list[dict[str, Any]]:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            self._trace.append(
                {
                    "stage": "builder_parse_failure",
                    "builder": self.artifact_type,
                    "detector": detector_type,
                    "rule": rule,
                    "error": str(exc),
                }
            )
            return []
        if not isinstance(data, dict):
            self._trace.append(
                {
                    "stage": "builder_parse_failure",
                    "builder": self.artifact_type,
                    "detector": detector_type,
                    "rule": rule,
                    "error": "llm response not a JSON object",
                }
            )
            return []
        items = data.get("items", [])
        if not isinstance(items, list):
            return []
        return [it for it in items if isinstance(it, dict)]

    def _make_item(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        detector_type: str,
        rule: str,
        raw: dict[str, Any],
    ) -> ArtifactItem | None:
        narrative = str(raw.get("narrative", "")).strip()
        if not narrative:
            return None
        # The detector's type wins when the LLM disagrees — adapter config is
        # authoritative. If the LLM returned a known type we record it in
        # domain_tags anyway so downstream retrieval sees both.
        llm_type = str(raw.get("type", detector_type)).strip() or detector_type
        resolved_type = detector_type if detector_type in _INSIGHT_TYPES else llm_type
        if resolved_type not in _INSIGHT_TYPES:
            # Drop items whose type is outside the spec §6 enum.
            self._trace.append(
                {
                    "stage": "builder_invalid_type",
                    "builder": self.artifact_type,
                    "detector": detector_type,
                    "rule": rule,
                    "got": resolved_type,
                }
            )
            return None
        evidence = _parse_evidence(raw.get("evidence") or [])
        if not evidence:
            return None
        confidence = _clip_confidence(raw.get("confidence", 0.6))
        domain_tags = [
            str(t) for t in (raw.get("domain_tags") or []) if isinstance(t, str)
        ]
        if resolved_type not in domain_tags:
            domain_tags.insert(0, resolved_type)
        temporal_scope = raw.get("temporal_scope")
        entity_refs = [
            str(e) for e in (raw.get("entity_refs") or []) if isinstance(e, str)
        ]
        return ArtifactItem(
            item_id=f"insight:{subscription_id}:{profile_id}:{resolved_type}:{uuid.uuid4().hex[:8]}",
            artifact_type="insight",
            subscription_id=subscription_id,
            profile_id=profile_id,
            text=narrative,
            evidence=evidence,
            confidence=confidence,
            inference_path=[],
            domain_tags=domain_tags,
            metadata={
                "detector": detector_type,
                "rule": rule,
                "temporal_scope": temporal_scope,
                "entity_refs": entity_refs,
            },
        )


def _parse_evidence(raw_list: list[Any]) -> list[EvidenceRef]:
    out: list[EvidenceRef] = []
    for ev in raw_list:
        if not isinstance(ev, dict):
            continue
        doc_id = str(ev.get("doc_id") or "")
        chunk_id = str(ev.get("chunk_id") or "")
        if not doc_id or not chunk_id:
            continue
        quote = ev.get("quote")
        out.append(
            EvidenceRef(
                doc_id=doc_id,
                chunk_id=chunk_id,
                quote=str(quote) if quote is not None else None,
            )
        )
    return out


def _clip_confidence(raw: Any) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.6
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val
