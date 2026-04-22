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


# ---------------------------------------------------------------------------
# Entity type handling — open vocabulary with light-touch normalisation.
#
# DocWain must support documents from any domain: invoices, resumes, medical
# reports, legal contracts, technical specs, academic papers, emails, and
# whatever comes next. Hardcoding a closed entity-type enum would force every
# domain-specific concept (MEDICATION, DIAGNOSIS, CLAUSE, CITATION,
# PROTOCOL_NAME, GENE, ...) to collapse into OTHER, throwing away useful
# structure.
#
# Design:
#   - Schema's ``type`` is a plain string — any label is allowed.
#   - Prompt lists common canonical labels as examples + tells the model to
#     invent domain-specific types when they fit the content better.
#   - A small alias table normalises well-known drift (ORG vs ORGANISATION;
#     POSTCODE vs POSTAL_CODE; AMOUNT vs MONEY; etc.) so KG doesn't
#     accumulate duplicate type nodes for the same concept.
# ---------------------------------------------------------------------------


# Common canonical types — listed for prompt guidance and alias normalisation.
# Domain-specific types (not in this list) flow through untouched.
_COMMON_CANONICAL_TYPES = [
    "PERSON", "ORGANIZATION", "LOCATION", "ADDRESS", "CITY", "COUNTRY",
    "POSTAL_CODE", "PHONE", "EMAIL", "URL",
    "DATE", "DURATION", "MONEY", "QUANTITY", "CURRENCY", "IDENTIFIER",
    "PRODUCT", "SERVICE", "DOCUMENT_TYPE", "EVENT", "OTHER",
]


# Alias map: map observed drift variants to the canonical form. Any type
# NOT in this map is preserved as-is (uppercased, whitespace-stripped),
# so domain-specific types like MEDICATION / CLAUSE / GENE flow through.
_TYPE_ALIASES: Dict[str, str] = {
    # ORGANIZATION variants
    "ORG": "ORGANIZATION",
    "ORGANISATION": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "VENDOR": "ORGANIZATION",
    "CUSTOMER": "ORGANIZATION",
    "INSTITUTION": "ORGANIZATION",
    # POSTAL_CODE variants
    "POSTCODE": "POSTAL_CODE",
    "POST_CODE": "POSTAL_CODE",
    "ZIP": "POSTAL_CODE",
    "ZIPCODE": "POSTAL_CODE",
    "ZIP_CODE": "POSTAL_CODE",
    # PHONE variants
    "PHONE_NUMBER": "PHONE",
    "TELEPHONE": "PHONE",
    "TEL": "PHONE",
    "MOBILE": "PHONE",
    # EMAIL variants
    "EMAIL_ADDRESS": "EMAIL",
    "E_MAIL": "EMAIL",
    # MONEY variants
    "AMOUNT": "MONEY",
    "PRICE": "MONEY",
    "CURRENCY_AMOUNT": "MONEY",
    "MONETARY_AMOUNT": "MONEY",
    # DATE variants
    "DATE_TIME": "DATE",
    "DATETIME": "DATE",
    "TIMESTAMP": "DATE",
    "CALENDAR_DATE": "DATE",
    # IDENTIFIER variants
    "ID": "IDENTIFIER",
    "REFERENCE": "IDENTIFIER",
    "REF": "IDENTIFIER",
    "CODE": "IDENTIFIER",
    "SKU": "IDENTIFIER",
    # ADDRESS variants
    "STREET_ADDRESS": "ADDRESS",
    "POSTAL_ADDRESS": "ADDRESS",
    "MAILING_ADDRESS": "ADDRESS",
    # PRODUCT / SERVICE kept commerce-specific but canonical
    "ITEM": "PRODUCT",
    "GOOD": "PRODUCT",
}


def _normalize_type(raw: Any) -> str:
    """Normalise a raw type label: uppercase, strip spaces, map known aliases.

    Unknown types (domain-specific or newly-coined) are returned in their
    canonical uppercase form but otherwise untouched — so a medical doc's
    ``MEDICATION`` or a legal doc's ``CLAUSE`` survive and flow to KG.
    """
    if not raw:
        return "OTHER"
    s = str(raw).strip().upper().replace(" ", "_").replace("-", "_")
    return _TYPE_ALIASES.get(s, s)


# JSON schema for vLLM guided_json. ``type`` is an open string — no enum.
# No ``think`` field: it's pure latency tax (hundreds of tokens of reasoning
# that never reaches the caller) and guided_json will emit it verbatim when
# included in the schema.
_V2_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
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
    "You are DocWain, a domain-agnostic document intelligence model. "
    "Given the deterministically-extracted content of a document of any "
    "kind (business, medical, legal, technical, academic, scientific, "
    "or otherwise), produce structured semantic understanding:\n"
    "  - entities: things worth naming, with a ``type`` label\n"
    "  - fields: a flat key-value dictionary of the document's key facts "
    "(keys should be descriptive snake_case labels drawn from the content — "
    "vendor, total, patient_name, diagnosis, party_a, effective_date, "
    "protocol_version, etc., whatever fits the document)\n"
    "  - relationships: factual relations between entities\n"
    "  - confidence: overall score in [0, 1]\n\n"
    "BE CONCISE. List each distinct entity exactly once — never repeat the "
    "same text/type pair. Extract only content-bearing entities. Skip page "
    "numbers, running headers/footers, boilerplate, and generic label words. "
    "Prefer ``fields`` over entity duplication when the information is a "
    "key-value fact (put vendor/total/date in fields, not as a separate "
    "ORGANIZATION/MONEY/DATE entity too).\n\n"
    "Do not invent content not present in the provided text. Use exact "
    "values from the text. Omit any field not stated in the text rather "
    "than guessing. Return a JSON object matching the schema.\n\n"
    "Entity type naming — use SCREAMING_SNAKE_CASE. These are common "
    "canonical types; use them when they fit:\n"
    "  PERSON ORGANIZATION LOCATION ADDRESS CITY COUNTRY POSTAL_CODE\n"
    "  PHONE EMAIL URL DATE DURATION MONEY QUANTITY CURRENCY\n"
    "  IDENTIFIER PRODUCT SERVICE DOCUMENT_TYPE EVENT OTHER\n\n"
    "If the document is from a specialised domain and none of the above "
    "fit, introduce an appropriate domain-specific type in the same style "
    "— MEDICATION, DIAGNOSIS, PROCEDURE, LAW_REFERENCE, CLAUSE, STATUTE, "
    "CITATION, GENE, PROTEIN, PROTOCOL, VERSION, COMPONENT, THEOREM, "
    "CONCEPT, etc. Prefer existing canonical types over inventing "
    "synonyms (use ORGANIZATION, not ORG or ORGANISATION; use "
    "POSTAL_CODE, not POSTCODE or ZIP)."
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
        "entities": [],
        "tables": [],
        "fields": {},
        "relationships": [],
        "confidence": confidence,
    }


def _repair_truncated_json(candidate: str) -> Optional[str]:
    """Best-effort close for JSON truncated by max_tokens.

    When the model hits ``max_tokens`` mid-array the output looks like::

        {"entities": [{"text":"a","type":"X"}, {"text":"b","type

    ``json.loads`` dies with ``Expecting ',' delimiter`` partway through.
    Re-running the 90-second call just to get a tidy closing brace is a
    non-starter, so we walk the string once, track structural depth, and
    remember the position of the last ``}``/``]`` that closed something
    at depth >= 1 (i.e. ended an inner element of an outer container).

    On truncation we slice to that position and append the closing
    characters needed to balance what's still open. That lets us keep
    the entities/fields that did make it out rather than discarding the
    whole response.
    """
    stack: List[str] = []
    in_string = False
    escape = False
    last_inner_close = -1
    stack_at_last: List[str] = []

    for i, ch in enumerate(candidate):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_string:
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}":
            if not stack or stack[-1] != "{":
                return None  # unbalanced — not a shape we can repair
            stack.pop()
            if stack:
                last_inner_close = i
                stack_at_last = list(stack)
        elif ch == "]":
            if not stack or stack[-1] != "[":
                return None
            stack.pop()
            if stack:
                last_inner_close = i
                stack_at_last = list(stack)

    if not stack:
        return candidate  # already balanced — caller will parse cleanly
    if last_inner_close < 0:
        return None  # nothing safe to keep

    head = candidate[: last_inner_close + 1].rstrip().rstrip(",")
    closers = "".join(
        "}" if c == "{" else "]" for c in reversed(stack_at_last)
    )
    return head + closers


def _parse_response(raw_text: str) -> Dict[str, Any]:
    """Parse the model's JSON response defensively.

    vLLM guided_json should give us valid JSON, but when generation is
    cut off by ``max_tokens`` the output is a truncated prefix. Instead
    of retrying the whole 90-second generation we attempt a one-shot
    repair (closing unbalanced braces/brackets at the last safe point)
    and only fall back to an empty result if that also fails.
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
    if start == -1:
        logger.warning("V2: model output contained no JSON object")
        return _default_empty_result(confidence=0.3)
    body = candidate[start: end + 1] if end > start else candidate[start:]

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        repaired = _repair_truncated_json(body)
        if repaired:
            try:
                parsed = json.loads(repaired)
                logger.info(
                    "V2: JSON repaired from %d→%d chars (was %s)",
                    len(body), len(repaired), exc.msg,
                )
            except json.JSONDecodeError as exc2:
                logger.warning(
                    "V2: JSON repair still invalid: %s (original: %s)",
                    exc2, exc,
                )
        if parsed is None:
            logger.warning("V2: JSON decode failed, unrepairable: %s", exc)
            return _default_empty_result(confidence=0.3)

    # Normalise entities to the shape the merger's Entity dataclass expects
    # (text / type / confidence). Entity ``type`` is also normalised via the
    # alias table so common drift variants (ORG/ORGANISATION, POSTCODE, etc.)
    # collapse to canonical labels, while domain-specific types pass through.
    # De-dup on (text, type) so a model that still lists the same entity
    # twice despite the prompt guidance doesn't double-bill downstream
    # consumers (KG, UI entity counts).
    entities = []
    seen: set = set()
    for e in (parsed.get("entities") or []):
        if not isinstance(e, dict):
            continue
        text = str(e.get("text") or e.get("name") or e.get("value") or "").strip()
        if not text:
            continue
        etype = _normalize_type(e.get("type"))
        key = (text.lower(), etype)
        if key in seen:
            continue
        seen.add(key)
        entities.append({
            "text": text,
            "type": etype,
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

        Returns a dict with keys: entities, tables, fields, relationships,
        confidence. The shape matches what ``ExtractionEngine`` / merger
        expect; callers that relied on a ``think`` field should reference
        ``extraction_confidence`` or the merger's ``understanding`` payload
        instead.
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

        # Size max_tokens proportional to input. The JSON encoding of a
        # well-structured extraction (entities + fields + relationships +
        # wrapping syntax) runs about 3x the input char count, which
        # translates to ~output_chars/3 tokens for JSON. Floor 1024 covers
        # tiny table-heavy docs; ceiling 6144 covers realistic long docs
        # without inviting runaway generation.
        #
        # The earlier 8192-floor setting was the root cause of minute-long
        # extractions: on short inputs the model would fill the budget with
        # duplicated entities and verbose "think" text, and — worse — often
        # hit the ceiling mid-JSON and produce corrupt output. Keep this
        # proportional sizing in lockstep with the "be concise" prompt
        # guidance above; loosening one without the other reintroduces the
        # runaway-generation pathology.
        max_tokens = max(1024, min(6144, int(len(text) * 3)))

        manager = self._get_manager()

        try:
            raw_text = manager.query(
                prompt=user_prompt,
                system_prompt=_SYSTEM_PROMPT,
                guided_json=_V2_RESPONSE_SCHEMA,
                max_tokens=max_tokens,
                temperature=0.05,
                require_vllm=True,  # unified DocWain model, no Ollama fallback
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("V2: vLLM query failed: %s", exc)
            return _default_empty_result(confidence=0.0)

        result = _parse_response(raw_text)

        logger.info(
            "V2 extraction complete — fmt=%s doc_type=%s entities=%d tables=%d confidence=%.2f max_tokens=%d",
            file_type, doc_type,
            len(result["entities"]), len(result["tables"]),
            result["confidence"], max_tokens,
        )
        return result


__all__ = ["V2Extractor"]
