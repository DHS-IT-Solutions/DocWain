"""ComparativeRegisterBuilder — cross-doc comparisons on adapter-defined axes.

Phase 2 implementation. Phase 1 shipped the skeleton returning ``[]``; this
module now iterates the adapter's ``comparison_axes`` list and generates one
:class:`ArtifactItem` per delta / conflict / timeline / corroboration finding
the LLM extracts for each axis.

Each emitted item carries:

* ``text`` = the analysis narrative (ERRATA §3 — used by SMEVerifier).
* ``evidence`` = ≥2 :class:`EvidenceRef` entries (the verifier enforces ≥1;
  per spec §6 comparative items should cite ≥2 distinct documents, which the
  builder encourages through the prompt but does not hard-enforce — the
  orchestrator's verifier runs check 1 + 2 so under-cited items drop there).
* ``metadata`` = ``{comparison_type, axis, dimension, unit, compared_items,
  resolution}`` — ``compared_items`` is a list of ``doc_id`` strings pulled
  out of the evidence refs so retrieval can filter on participating docs.

LLM injection uses the shared ``LLMClient`` protocol; tests inject a
MagicMock. No wall-clock timeout at the DocWain layer.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef
from src.intelligence.sme.builders._base import ArtifactBuilder
from src.intelligence.sme.builders.dossier import LLMClient, _TraceSink
from src.intelligence.sme.builders.insight_index import (
    _clip_confidence,
    _parse_evidence,
)

_COMPARATIVE_TYPES = frozenset(
    {"delta", "conflict", "timeline", "corroboration"}
)


class ComparativeRegisterBuilder(ArtifactBuilder):
    """Builder for the comparative register artifact."""

    artifact_type = "comparison"

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
            "array of comparison findings. Each finding has keys: type "
            "(delta|conflict|timeline|corroboration), axis, analysis, "
            "resolution (optional), compared_items (list of doc_ids, "
            "ideally ≥2 distinct docs), evidence (list of {doc_id, "
            "chunk_id, quote?}), confidence (0.0-1.0)."
        )
        pooled: list[ArtifactItem] = []
        for axis in adapter.comparison_axes:
            axis_name = axis.name
            dimension = axis.dimension
            unit = axis.unit
            user_prompt = (
                f"Compare evidence along axis '{axis_name}' "
                f"(dimension={dimension}"
                f"{f', unit={unit}' if unit else ''}). Enumerate every "
                "material delta, conflict, temporal progression, or cross-"
                "document corroboration you observe. Prefer comparisons that "
                "span ≥2 distinct documents.\n\nEVIDENCE:\n" + evidence_pack
            )
            trace_tag = (
                f"comparative:{subscription_id}:{profile_id}:{axis_name}"
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
                        "axis": axis_name,
                        "error": str(exc),
                    }
                )
                continue
            parsed_items = self._parse_items(raw, axis_name)
            for raw_item in parsed_items:
                item = self._make_item(
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    axis_name=axis_name,
                    dimension=dimension,
                    unit=unit,
                    raw=raw_item,
                )
                if item is not None:
                    pooled.append(item)
        return pooled

    # ------------------------------------------------------------------
    def _parse_items(self, raw: str, axis_name: str) -> list[dict[str, Any]]:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            self._trace.append(
                {
                    "stage": "builder_parse_failure",
                    "builder": self.artifact_type,
                    "axis": axis_name,
                    "error": str(exc),
                }
            )
            return []
        if not isinstance(data, dict):
            self._trace.append(
                {
                    "stage": "builder_parse_failure",
                    "builder": self.artifact_type,
                    "axis": axis_name,
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
        axis_name: str,
        dimension: str,
        unit: str | None,
        raw: dict[str, Any],
    ) -> ArtifactItem | None:
        analysis = str(raw.get("analysis", "")).strip()
        if not analysis:
            return None
        comparison_type = str(raw.get("type", "")).strip()
        if comparison_type not in _COMPARATIVE_TYPES:
            self._trace.append(
                {
                    "stage": "builder_invalid_type",
                    "builder": self.artifact_type,
                    "axis": axis_name,
                    "got": comparison_type,
                }
            )
            return None
        evidence = _parse_evidence(raw.get("evidence") or [])
        if not evidence:
            return None
        raw_compared = raw.get("compared_items") or []
        compared_items = [
            str(d) for d in raw_compared if isinstance(d, (str, int, float))
        ]
        if not compared_items:
            compared_items = sorted({e.doc_id for e in evidence})
        resolution = raw.get("resolution")
        confidence = _clip_confidence(raw.get("confidence", 0.6))
        domain_tags = [comparison_type]
        if dimension and dimension not in domain_tags:
            domain_tags.append(dimension)
        return ArtifactItem(
            item_id=f"comparison:{subscription_id}:{profile_id}:{axis_name}:{uuid.uuid4().hex[:8]}",
            artifact_type="comparison",
            subscription_id=subscription_id,
            profile_id=profile_id,
            text=analysis,
            evidence=evidence,
            confidence=confidence,
            inference_path=[],
            domain_tags=domain_tags,
            metadata={
                "comparison_type": comparison_type,
                "axis": axis_name,
                "dimension": dimension,
                "unit": unit,
                "compared_items": compared_items,
                "resolution": str(resolution) if resolution else None,
            },
        )
