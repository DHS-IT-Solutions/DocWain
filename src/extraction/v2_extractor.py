"""V2 extractor — Layer 2 semantic understanding over Layer 1 deterministic text.

Calls the unified DocWain model (served by vLLM on port 8100) to produce
semantic entity / field / confidence labels. Deterministic Layer 1 already
gave us the text and tables; V2's job here is interpretation, not
re-extraction from raw bytes.

Backend: ``src.serving.vllm_manager.VLLMManager`` — which adds automatic
fallback to Ollama Cloud and local Ollama when vLLM is unreachable, so
extraction stays resilient even during serving hiccups or training mode.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# JSON schema constraining the model's output. vLLM enforces this with
# guided_json so the response is valid JSON with the expected keys every
# time — no regex-from-text parsing needed.
_V2_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "think": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["text", "type"],
            },
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "headers": {"type": "array", "items": {"type": "string"}},
                    "rows": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
        "fields": {"type": "object"},
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
            },
        },
        "confidence": {"type": "number"},
    },
    "required": ["entities", "fields", "confidence"],
}


_SYSTEM_PROMPT = (
    "You are DocWain, an enterprise document intelligence model. Given the "
    "deterministically-extracted content of a document, produce structured "
    "semantic understanding: entities (people, organisations, amounts, "
    "dates, identifiers), key fields as a flat dictionary "
    "(vendor, total, invoice_number, date, etc.), relationships between "
    "entities, and an overall confidence score in [0, 1]. Do not invent "
    "content not present in the provided text. Return a JSON object "
    "matching the required schema."
)


def _build_user_prompt(
    *,
    text_content: str,
    doc_type: str,
    page_type: str,
    file_type: str,
) -> str:
    return (
        f"Document format: {file_type}\n"
        f"Document type hint: {doc_type}\n"
        f"Page role: {page_type}\n\n"
        f"--- DOCUMENT TEXT (deterministically extracted) ---\n"
        f"{text_content}\n"
        f"--- END DOCUMENT TEXT ---\n\n"
        "Extract entities, relationships, and key fields from the text above. "
        "Use exact values from the text — do not invent anything. For fields "
        "that are not present in the text, omit the key rather than guessing."
    )


def _default_empty_result(confidence: float = 0.0) -> Dict[str, Any]:
    return {
        "think": "",
        "entities": [],
        "tables": [],
        "fields": {},
        "relationships": [],
        "confidence": confidence,
    }


def _parse_response(raw_text: str) -> Dict[str, Any]:
    """Parse the model's JSON response defensively.

    vLLM guided_json should give us valid JSON, but we still handle the
    "wrapped in fences / extra prose" case from any fallback backend.
    """
    if not raw_text or not raw_text.strip():
        return _default_empty_result(confidence=0.0)

    candidate = raw_text.strip()
    # strip common markdown fence wrappers
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    # find the outermost JSON object
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        logger.warning("V2: model output contained no JSON object")
        return _default_empty_result(confidence=0.3)

    try:
        parsed = json.loads(candidate[start: end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("V2: JSON decode failed: %s", exc)
        return _default_empty_result(confidence=0.3)

    # Normalise entities to the shape the merger's Entity dataclass expects
    # (text / type / confidence). The prompt schema uses ``name`` for readability
    # but the downstream dataclass field is ``text``.
    entities = []
    for e in (parsed.get("entities") or []):
        if not isinstance(e, dict):
            continue
        text = str(e.get("text") or e.get("name") or e.get("value") or "").strip()
        if not text:
            continue
        entities.append({
            "text": text,
            "type": str(e.get("type") or "UNKNOWN").upper(),
            "confidence": float(e.get("confidence", 0.8) or 0.0),
        })

    # Normalise relationships to the shape the merger's Relationship dataclass
    # expects (subject/predicate/object/confidence/evidence).
    relationships = []
    for r in (parsed.get("relationships") or []):
        if not isinstance(r, dict):
            continue
        relationships.append({
            "subject": str(r.get("subject", "")),
            "predicate": str(r.get("predicate", "")),
            "object": str(r.get("object", "")),
            "confidence": float(r.get("confidence", 0.8) or 0.0),
            "evidence": str(r.get("evidence", "")),
        })

    return {
        "think": parsed.get("think", "") or "",
        "entities": entities,
        "tables": list(parsed.get("tables") or []),
        "fields": dict(parsed.get("fields") or {}),
        "relationships": relationships,
        "confidence": float(parsed.get("confidence", 0.3) or 0.0),
    }


class V2Extractor:
    """Semantic understanding layer backed by the unified DocWain vLLM model.

    Parameters
    ----------
    vllm_manager
        Optional pre-built :class:`VLLMManager`. If ``None``, a default
        manager is constructed using ``src.serving.config`` defaults on
        first call (lazy). Tests can inject a mock.
    ollama_host
        Legacy parameter kept for backward-compat with existing callers
        (ExtractionEngine passes it). Currently unused — all calls go
        through the vLLM manager. When the v1 extractor's Ollama path
        is fully retired this arg will be removed.
    max_text_chars
        Cap on document text length included in the prompt. Very long
        documents are truncated to protect the model context window; the
        deterministic extraction still carries the full text downstream.
    """

    def __init__(
        self,
        *,
        vllm_manager: Any = None,
        ollama_host: Optional[str] = None,  # noqa: ARG002 (kept for sig compat)
        max_text_chars: int = 24000,
    ) -> None:
        self._manager = vllm_manager
        self.max_text_chars = max_text_chars

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_manager(self):
        if self._manager is not None:
            return self._manager
        # Prefer the AppState-managed singleton when the API is running so
        # we reuse the configured VLLMManager (URL, model, gpu_mode_file).
        try:
            from src.api.rag_state import get_app_state
            state = get_app_state()
            if state and getattr(state, "vllm_manager", None):
                self._manager = state.vllm_manager
                return self._manager
        except Exception:  # noqa: BLE001
            pass
        # Fall back to a default instance for offline / test use.
        from src.serving.vllm_manager import VLLMManager
        from src.serving.config import GPU_MODE_FILE
        self._manager = VLLMManager(gpu_mode_file=GPU_MODE_FILE)
        return self._manager

    # ------------------------------------------------------------------
    # Public API — signature preserved for ExtractionEngine
    # ------------------------------------------------------------------

    def extract(
        self,
        document_bytes: bytes,  # noqa: ARG002 (kept for sig compat — not used in Layer 2)
        file_type: str,
        page_images: Optional[List[Any]] = None,  # noqa: ARG002 (V1 multi-modal, unused here)
        *,
        doc_type: str = "unknown",
        page_type: str = "body",
        text_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run V2 semantic extraction over the deterministic text.

        Returns a dict with keys: think, entities, tables, fields,
        relationships, confidence. The shape matches the previous Ollama
        implementation so ``ExtractionEngine`` / merger don't need changes.
        """
        text = (text_content or "").strip()
        if not text:
            logger.info("V2: no text_content provided; returning empty result")
            return _default_empty_result(confidence=0.0)

        if len(text) > self.max_text_chars:
            logger.info(
                "V2: truncating text_content %d -> %d chars for prompt",
                len(text), self.max_text_chars,
            )
            text = text[: self.max_text_chars]

        user_prompt = _build_user_prompt(
            text_content=text,
            doc_type=doc_type,
            page_type=page_type,
            file_type=file_type,
        )

        manager = self._get_manager()
        try:
            raw_text = manager.query(
                prompt=user_prompt,
                system_prompt=_SYSTEM_PROMPT,
                guided_json=_V2_RESPONSE_SCHEMA,
                max_tokens=4096,
                temperature=0.05,  # near-deterministic for extraction
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("V2: vLLM query failed: %s", exc)
            return _default_empty_result(confidence=0.0)

        result = _parse_response(raw_text)
        logger.info(
            "V2 extraction complete — fmt=%s doc_type=%s entities=%d tables=%d confidence=%.2f",
            file_type, doc_type,
            len(result["entities"]), len(result["tables"]),
            result["confidence"],
        )
        return result


__all__ = ["V2Extractor"]
