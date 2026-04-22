"""RecommendationBankBuilder — grounded recommendations linked to insights.

Phase 2 implementation. Phase 1 shipped the skeleton returning ``[]``. This
module now iterates the adapter's ``recommendation_frames`` and, for each
frame, filters the caller-provided Insight Index items by
``frame.requires.insight_types`` then LLM-synthesizes a
:class:`ArtifactItem` that cites ≥1 linked insight and ≥1 evidence chunk —
per the task brief these are HARD requirements at the builder layer; items
that fail either check drop before the verifier runs.

Phase 2 orchestrator (Task 7) threads the verified Insight Index items into
``build(..., insight_items=...)`` so recommendations can quote the
``item_id`` of every insight they stand on. The linkage is preserved in
``metadata["linked_insights"]``.

LLM injection reuses the dossier protocol; trace reuses the same structured
append surface.
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


class RecommendationBankBuilder(ArtifactBuilder):
    """Builder for the recommendation-bank artifact."""

    artifact_type = "recommendation"

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

    def build(  # type: ignore[override]
        self,
        *,
        subscription_id: str,
        profile_id: str,
        adapter,
        version: int,
        insight_items: list[ArtifactItem] | None = None,
    ) -> list[ArtifactItem]:
        """Extended public entrypoint — Phase 2 recommendation bank consumes
        the verified Insight Index items so every recommendation can cite
        ≥1 insight item_id (task contract).

        The base class :meth:`ArtifactBuilder.build` signature lacks the
        ``insight_items`` kwarg; the orchestrator calls this builder's
        ``build`` directly, so the override is safe. Other consumers that
        call the base signature get ``insight_items=None`` which disables
        frames whose ``requires.insight_types`` list is non-empty.
        """
        return self._synthesize_with_insights(
            subscription_id=subscription_id,
            profile_id=profile_id,
            adapter=adapter,
            version=version,
            insight_items=insight_items or [],
        )

    def _synthesize(
        self, *, subscription_id: str, profile_id: str, adapter, version: int
    ) -> list[ArtifactItem]:
        # Fallback path for the base ABC — returns [] when no insight_items
        # were threaded (e.g. a test that bypasses the public ``build``).
        return self._synthesize_with_insights(
            subscription_id=subscription_id,
            profile_id=profile_id,
            adapter=adapter,
            version=version,
            insight_items=[],
        )

    # ------------------------------------------------------------------
    def _synthesize_with_insights(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        adapter,
        version: int,
        insight_items: list[ArtifactItem],
    ) -> list[ArtifactItem]:
        chunks = list(self._ctx.iter_profile_chunks(subscription_id, profile_id))
        evidence_pack = "\n".join(
            f"[{c['doc_id']}#{c['chunk_id']}] {c.get('text', '')}" for c in chunks
        )
        persona = adapter.persona
        system_prompt = (
            f"You are a {persona.role}. Voice: {persona.voice}. "
            "Return ONLY a JSON object with key 'items' whose value is an "
            "array. Each item has keys: recommendation, rationale, "
            "linked_insights (list of insight item_ids you are standing on — "
            "MUST be a non-empty subset of those provided), "
            "estimated_impact (object, qualitative OK), assumptions (list), "
            "caveats (list), evidence (list of {doc_id, chunk_id, quote?}), "
            "confidence (0.0-1.0), domain_tags (list). Every recommendation "
            "MUST cite ≥1 linked_insights and ≥1 evidence entry."
        )
        pooled: list[ArtifactItem] = []
        for frame in adapter.recommendation_frames:
            frame_name = frame.frame
            template = frame.template
            requires = dict(frame.requires or {})
            required_types = [
                str(t) for t in (requires.get("insight_types") or [])
            ]
            eligible_insights = self._filter_insights(
                insight_items, required_types
            )
            if required_types and not eligible_insights:
                self._trace.append(
                    {
                        "stage": "builder_no_matching_insights",
                        "builder": self.artifact_type,
                        "frame": frame_name,
                        "required_types": required_types,
                    }
                )
                continue
            insight_pack = "\n".join(
                f"[{ins.item_id}] ({_insight_detector(ins)}): {ins.text}"
                for ins in eligible_insights
            )
            user_prompt = (
                f"Frame: {frame_name}. Template hint: {template}.\n"
                f"Required insight types: {required_types or 'any'}.\n\n"
                f"CANDIDATE INSIGHTS:\n{insight_pack}\n\n"
                f"EVIDENCE:\n{evidence_pack}"
            )
            trace_tag = (
                f"recommendation:{subscription_id}:{profile_id}:{frame_name}"
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
                        "frame": frame_name,
                        "error": str(exc),
                    }
                )
                continue
            parsed = self._parse_items(raw, frame_name)
            eligible_ids = {i.item_id for i in eligible_insights}
            for raw_item in parsed:
                item = self._make_item(
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    frame_name=frame_name,
                    template=template,
                    eligible_insight_ids=eligible_ids,
                    raw=raw_item,
                )
                if item is not None:
                    pooled.append(item)
        return pooled

    # ------------------------------------------------------------------
    def _filter_insights(
        self,
        insight_items: list[ArtifactItem],
        required_types: list[str],
    ) -> list[ArtifactItem]:
        if not required_types:
            return list(insight_items)
        required_set = set(required_types)
        out: list[ArtifactItem] = []
        for ins in insight_items:
            detector = _insight_detector(ins)
            if detector in required_set:
                out.append(ins)
                continue
            # fallback — adapter may lean on domain_tags when detector is
            # absent
            if any(t in required_set for t in ins.domain_tags):
                out.append(ins)
        return out

    def _parse_items(self, raw: str, frame_name: str) -> list[dict[str, Any]]:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            self._trace.append(
                {
                    "stage": "builder_parse_failure",
                    "builder": self.artifact_type,
                    "frame": frame_name,
                    "error": str(exc),
                }
            )
            return []
        if not isinstance(data, dict):
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
        frame_name: str,
        template: str,
        eligible_insight_ids: set[str],
        raw: dict[str, Any],
    ) -> ArtifactItem | None:
        recommendation = str(raw.get("recommendation", "")).strip()
        if not recommendation:
            return None
        rationale = str(raw.get("rationale", "")).strip()
        raw_links = raw.get("linked_insights") or []
        linked_insights: list[str] = [
            str(i) for i in raw_links if isinstance(i, str)
        ]
        # Hard task requirement: every recommendation cites ≥1 linked insight.
        if not linked_insights:
            self._trace.append(
                {
                    "stage": "builder_no_linked_insights",
                    "builder": self.artifact_type,
                    "frame": frame_name,
                }
            )
            return None
        # Filter to eligible ids when the frame had a restrictor — an LLM
        # hallucinating insight ids outside the allowlist is a verifier-style
        # failure and we drop here.
        if eligible_insight_ids:
            filtered = [i for i in linked_insights if i in eligible_insight_ids]
            if not filtered:
                self._trace.append(
                    {
                        "stage": "builder_linked_insights_out_of_scope",
                        "builder": self.artifact_type,
                        "frame": frame_name,
                        "linked_insights": linked_insights,
                    }
                )
                return None
            linked_insights = filtered
        evidence = _parse_evidence(raw.get("evidence") or [])
        if not evidence:
            self._trace.append(
                {
                    "stage": "builder_no_evidence",
                    "builder": self.artifact_type,
                    "frame": frame_name,
                }
            )
            return None
        estimated_impact = raw.get("estimated_impact") or {}
        if not isinstance(estimated_impact, dict):
            estimated_impact = {"value": estimated_impact}
        assumptions = [
            str(a) for a in (raw.get("assumptions") or []) if isinstance(a, str)
        ]
        caveats = [
            str(c) for c in (raw.get("caveats") or []) if isinstance(c, str)
        ]
        confidence = _clip_confidence(raw.get("confidence", 0.6))
        domain_tags = [
            str(t) for t in (raw.get("domain_tags") or []) if isinstance(t, str)
        ]
        return ArtifactItem(
            item_id=f"recommendation:{subscription_id}:{profile_id}:{frame_name}:{uuid.uuid4().hex[:8]}",
            artifact_type="recommendation",
            subscription_id=subscription_id,
            profile_id=profile_id,
            text=recommendation,
            evidence=evidence,
            confidence=confidence,
            inference_path=[],
            domain_tags=domain_tags,
            metadata={
                "frame": frame_name,
                "template": template,
                "rationale": rationale,
                "linked_insights": linked_insights,
                "estimated_impact": estimated_impact,
                "assumptions": assumptions,
                "caveats": caveats,
            },
        )


def _insight_detector(insight: ArtifactItem) -> str:
    """Return the insight's detector type for frame matching.

    The Insight Index builder stores the detector in both ``metadata["detector"]``
    and ``domain_tags[0]`` — we prefer the metadata entry for clarity.
    """
    detector = insight.metadata.get("detector") if insight.metadata else None
    if detector:
        return str(detector)
    return insight.domain_tags[0] if insight.domain_tags else ""
