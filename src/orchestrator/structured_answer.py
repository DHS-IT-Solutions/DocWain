from __future__ import annotations

import json
from src.utils.logging_utils import get_logger
import re
from typing import Any, Dict, List, Optional

from src.orchestrator.response_validator import validate_response_payload
from src.retrieval.profile_evidence import DocumentEvidence, ProfileEvidenceGraph

logger = get_logger(__name__)

def select_output_schema(intent: str) -> Dict[str, Any]:
    if intent == "extract":
        return {"schema": "extract", "required_fields": ["schema", "documents"]}
    if intent == "list":
        return {"schema": "list", "required_fields": ["schema", "items", "documents"]}
    if intent == "compare":
        return {"schema": "compare", "required_fields": ["schema", "comparisons", "documents"]}
    if intent == "rank":
        return {"schema": "rank", "required_fields": ["schema", "ranking", "documents"]}
    return {"schema": "answer", "required_fields": ["schema", "answer", "documents"]}

def generate_structured_answer(
    *,
    user_query: str,
    intent: str,
    retrieval_scope: str,
    target_document_ids: List[str],
    evidence_graph: ProfileEvidenceGraph,
    model_name: Optional[str],
    llm_client=None,
) -> str:
    schema = select_output_schema(intent)
    base_payload = build_payload(
        user_query=user_query,
        schema_name=schema["schema"],
        retrieval_scope=retrieval_scope,
        target_document_ids=target_document_ids,
        evidence_graph=evidence_graph,
    )

    if not model_name and llm_client is None:
        return json.dumps(base_payload, indent=2)

    prompt = _build_prompt(user_query, schema["schema"], base_payload)
    logger.info("structured_answer: prompt_len=%d schema=%s", len(prompt), schema["schema"])
    try:
        if llm_client is not None:
            text = llm_client.generate(prompt, max_tokens=_LLM_ANSWER_MAX_TOKENS)
        else:
            from src.llm.gateway import get_llm_gateway
            text = get_llm_gateway().generate(prompt, max_tokens=_LLM_ANSWER_MAX_TOKENS)
        text = (text or "").strip()
        logger.info(
            "structured_answer: llm returned len=%d preview=%r",
            len(text), text[:200] if text else "",
        )
        llm_payload = _extract_json(text)
        if llm_payload is None:
            logger.warning("structured_answer: json extract FAILED; returning base_payload")
        elif not isinstance(llm_payload, dict):
            logger.warning(
                "structured_answer: parsed payload not a dict (type=%s); returning base_payload",
                type(llm_payload).__name__,
            )
        else:
            # Graft the LLM's narrative/result fields onto base_payload.
            # The prompt tells the LLM NOT to echo `documents` (which would
            # overflow max_tokens on a 12+ doc profile) — the server supplies
            # that from base_payload. This way the LLM only produces the
            # schema-specific narrative field (~100 tokens vs 5K+ echoed).
            merged = dict(base_payload)
            for narrative_key in ("answer", "items", "comparisons", "ranking"):
                if narrative_key in llm_payload:
                    merged[narrative_key] = llm_payload[narrative_key]
            if _has_top_level_required(merged, schema):
                logger.info(
                    "structured_answer: LLM narrative grafted onto base_payload "
                    "(llm_keys=%s)",
                    list(llm_payload.keys()),
                )
                return json.dumps(merged, indent=2)
            logger.warning(
                "structured_answer: merged payload missing required fields; "
                "required=%s got=%s; returning base_payload",
                schema.get("required_fields"), list(merged.keys()),
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("structured_answer: LLM path raised — returning base_payload")
    return json.dumps(base_payload, indent=2)


# Structured-answer LLM calls are JSON-shaped and bounded. 2048 tokens is
# comfortably above any realistic schema-valid response for 10-25 docs
# (~700 tokens observed on 12-doc invoice profile). Keeping this well below
# Config.LLM.MAX_TOKENS (8192) avoids the case where Qwen3-14B generates
# filler up to the ceiling when response_format isn't enforced, which was
# producing 180s per-query latency before this change.
_LLM_ANSWER_MAX_TOKENS = 2048


def _has_top_level_required(payload: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    for field in schema.get("required_fields") or []:
        if field not in payload:
            return False
    return True

def build_payload(
    *,
    user_query: str,
    schema_name: str,
    retrieval_scope: str,
    target_document_ids: List[str],
    evidence_graph: ProfileEvidenceGraph,
) -> Dict[str, Any]:
    documents = [_document_payload(doc) for doc in evidence_graph.documents.values()]
    payload: Dict[str, Any] = {
        "schema": schema_name,
        "query": user_query,
        "retrieval_scope": retrieval_scope,
        "target_document_ids": target_document_ids,
        "documents": documents,
    }
    if _should_merge(user_query):
        payload["merged_entities"] = _merge_documents(evidence_graph)

    if schema_name == "extract":
        return payload
    if schema_name == "list":
        payload["items"] = _list_items_from_evidence(evidence_graph)
        return payload
    if schema_name == "compare":
        payload["comparisons"] = _compare_documents(evidence_graph)
        return payload
    if schema_name == "rank":
        payload["ranking"] = _rank_documents(evidence_graph)
        return payload
    payload["answer"] = _summarize_documents(evidence_graph)
    return payload

def _document_payload(doc: DocumentEvidence) -> Dict[str, Any]:
    contacts = doc.contacts
    return {
        "document_id": doc.document_id,
        "source_name": doc.source_name,
        "contacts": {
            "phones": _items_or_not_mentioned(contacts.get("phones", [])),
            "emails": _items_or_not_mentioned(contacts.get("emails", [])),
            "urls": _items_or_not_mentioned(contacts.get("urls", [])),
        },
        "dates": _items_or_not_mentioned(doc.dates),
        "identifiers": _items_or_not_mentioned(doc.identifiers),
        "entities": _items_or_not_mentioned(doc.entities),
        "sections": _items_or_not_mentioned(doc.sections),
        "tables": _items_or_not_mentioned(doc.tables),
    }

def _items_or_not_mentioned(items: List[Any]) -> Any:
    if not items:
        return "Not Mentioned"
    return [_item_payload(item) for item in items]

def _item_payload(item: Any) -> Dict[str, Any]:
    return {
        "value": item.value,
        "snippet": item.snippet,
        "document_id": item.document_id,
        "source_name": item.source_name,
        "chunk_id": item.chunk_id,
        "section_title": item.section_title,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "meta": item.meta,
    }

def _summarize_documents(graph: ProfileEvidenceGraph) -> str:
    parts: List[str] = []
    for doc in graph.documents.values():
        contact_count = sum(len(doc.contacts.get(key, [])) for key in ("phones", "emails", "urls"))
        identifier_count = len(doc.identifiers)
        date_count = len(doc.dates)
        parts.append(
            f"{doc.source_name or doc.document_id}: {contact_count} contacts, "
            f"{identifier_count} identifiers, {date_count} dates."
        )
    return " | ".join(parts) if parts else "Not Mentioned"

def _list_items_from_evidence(graph: ProfileEvidenceGraph) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for doc in graph.documents.values():
        for ident in doc.identifiers:
            items.append({"item": ident.value, "document_id": doc.document_id, "source_name": doc.source_name})
    return items

_MAX_COMPARE_PAIRS = 6


def _compare_documents(graph: ProfileEvidenceGraph) -> List[Dict[str, Any]]:
    # Cap at _MAX_COMPARE_PAIRS pairs — the full O(n^2) list for a 12-doc
    # profile produces 66 comparisons, which blows the LLM prompt budget
    # and the user-facing response size. The first few pairs are the
    # highest-signal anyway (most recently added docs).
    comparisons: List[Dict[str, Any]] = []
    doc_list = list(graph.documents.values())
    for idx, doc in enumerate(doc_list):
        for other in doc_list[idx + 1 :]:
            comparisons.append(
                {
                    "document_a": doc.document_id,
                    "document_b": other.document_id,
                    "similarities": _shared_values(doc, other),
                    "differences": _diff_values(doc, other),
                    "notes": "Not Mentioned" if not (doc.identifiers or other.identifiers) else "See identifiers.",
                }
            )
            if len(comparisons) >= _MAX_COMPARE_PAIRS:
                return comparisons
    return comparisons

def _rank_documents(graph: ProfileEvidenceGraph) -> List[Dict[str, Any]]:
    ranked = sorted(
        graph.documents.values(),
        key=lambda doc: len(doc.identifiers) + len(doc.dates) + sum(len(doc.contacts.get(k, [])) for k in ("phones", "emails", "urls")),
        reverse=True,
    )
    ranking: List[Dict[str, Any]] = []
    for idx, doc in enumerate(ranked, start=1):
        ranking.append(
            {
                "rank": idx,
                "document_id": doc.document_id,
                "source_name": doc.source_name,
                "reason": "Based on evidence density in extracted fields.",
            }
        )
    return ranking

def _shared_values(doc: DocumentEvidence, other: DocumentEvidence) -> List[str]:
    left = {item.value.lower() for item in doc.identifiers}
    right = {item.value.lower() for item in other.identifiers}
    return sorted(left.intersection(right))

def _diff_values(doc: DocumentEvidence, other: DocumentEvidence) -> List[str]:
    left = {item.value.lower() for item in doc.identifiers}
    right = {item.value.lower() for item in other.identifiers}
    return sorted(left.symmetric_difference(right))

_SCHEMA_OUTPUT_HINT: Dict[str, str] = {
    "answer": (
        '{"schema":"answer",'
        '"answer":"<natural-language answer to the user query, 1-3 sentences>"}'
    ),
    "extract": (
        '{"schema":"extract"}'
    ),
    "list": (
        '{"schema":"list",'
        '"items":[{"label":"<short label>","document_id":"<doc id>"}, ...]}'
    ),
    "compare": (
        '{"schema":"compare",'
        '"comparisons":[{"document_a":"<id>","document_b":"<id>",'
        '"similarities":["..."],"differences":["..."],"notes":"<short note>"}, ...]}'
    ),
    "rank": (
        '{"schema":"rank",'
        '"ranking":[{"rank":1,"document_id":"<id>","reason":"<short reason>"}, ...]}'
    ),
}


def _build_prompt(user_query: str, schema_name: str, base_payload: Dict[str, Any]) -> str:
    # Compact evidence before embedding: the raw base_payload carries per-item
    # snippet/chunk_id/meta/document_id/source_name that are redundant once
    # pulled into the prompt (document-level fields already scope them). For a
    # 12-doc profile the uncompacted payload is ~70K tokens, which overflows
    # DocWain's 32K context window and causes vLLM to reject with 400.
    # Compact form targets <=16K tokens for profiles up to ~25 docs.
    compact = _compact_payload_for_prompt(base_payload)
    schema_json = json.dumps(compact, separators=(",", ":"))
    hint = _SCHEMA_OUTPUT_HINT.get(schema_name, _SCHEMA_OUTPUT_HINT["answer"])
    answer_requirement = (
        '- The "answer" field MUST contain a natural-language answer '
        'to the user query (1-3 sentences), grounded in the evidence.\n'
        if schema_name == "answer" else ""
    )
    return (
        "You are DocWain's synthesis engine. Answer the user's query using ONLY "
        "the provided evidence. Do not invent facts.\n\n"
        f"User query: {user_query}\n\n"
        "Evidence (compact JSON):\n"
        f"{schema_json}\n\n"
        "Return ONLY valid JSON with EXACTLY this shape:\n"
        f"{hint}\n\n"
        "Requirements:\n"
        f'- The top-level "schema" field MUST be "{schema_name}".\n'
        '- Do NOT include a "documents" field in your response; the server '
        'will attach it from the evidence.\n'
        '- Do not add commentary, markdown, or text outside the JSON object.\n'
        f"{answer_requirement}"
    )


_MAX_ITEMS_PER_FIELD = 10
_MAX_SECTION_TITLE_LEN = 60


def _compact_payload_for_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a size-bounded copy of ``payload`` safe to embed in a prompt."""
    out: Dict[str, Any] = {
        k: v for k, v in payload.items() if k != "documents"
    }
    out["documents"] = [
        _compact_document(doc) for doc in payload.get("documents", [])
    ]
    if "merged_entities" in payload:
        out["merged_entities"] = payload["merged_entities"]
    return out


def _compact_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Strip redundant per-item fields; cap list lengths."""
    return {
        "id": doc.get("document_id"),
        "name": doc.get("source_name"),
        "contacts": _compact_contacts(doc.get("contacts")),
        "dates": _compact_items(doc.get("dates")),
        "identifiers": _compact_items(doc.get("identifiers")),
        "entities": _compact_items(doc.get("entities")),
        "sections": _compact_items(doc.get("sections")),
        "tables": _compact_items(doc.get("tables")),
    }


def _compact_contacts(contacts: Any) -> Dict[str, Any]:
    if not isinstance(contacts, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, items in contacts.items():
        compacted = _compact_items(items)
        if compacted:
            out[key] = compacted
    return out


def _compact_items(items: Any) -> List[Any]:
    if not isinstance(items, list) or not items:
        return []
    capped = items[:_MAX_ITEMS_PER_FIELD]
    return [_compact_item(it) for it in capped]


_MAX_SNIPPET_LEN = 180


def _compact_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    out: Dict[str, Any] = {}
    value = item.get("value")
    if value is not None:
        out["v"] = value
    snippet = item.get("snippet")
    # Snippets are the extracted surrounding text — essential for the LLM
    # to actually answer queries like "What is the total?" / "Which invoice
    # mentions X?". Without them the compact payload is just isolated
    # tokens and the model has nothing to reason over. Truncated so the
    # prompt stays under the 32K context budget.
    if snippet and snippet != value:
        out["snip"] = snippet[:_MAX_SNIPPET_LEN]
    p_start = item.get("page_start")
    p_end = item.get("page_end")
    if p_start is not None:
        out["p"] = str(p_start) if (p_end is None or p_end == p_start) else f"{p_start}-{p_end}"
    sec = item.get("section_title")
    if sec and sec != value:
        out["sec"] = sec[:_MAX_SECTION_TITLE_LEN]
    return out

def _should_merge(user_query: str) -> bool:
    lowered = (user_query or "").lower()
    return any(keyword in lowered for keyword in ("merge", "combine", "together"))

def _merge_documents(graph: ProfileEvidenceGraph) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    for doc in graph.documents.values():
        keys = _join_keys(doc)
        if not keys:
            continue
        entities.append(
            {
                "document_id": doc.document_id,
                "source_name": doc.source_name,
                "join_keys": sorted(keys),
                "contacts": _document_payload(doc)["contacts"],
            }
        )
    merged: List[Dict[str, Any]] = []
    used = set()
    for idx, entity in enumerate(entities):
        if idx in used:
            continue
        group = [entity]
        keys = set(entity["join_keys"])
        for jdx, other in enumerate(entities[idx + 1 :], start=idx + 1):
            if keys.intersection(other["join_keys"]):
                group.append(other)
                used.add(jdx)
        merged.append({"group": group, "merge_key": sorted(keys)})
    return merged

def _join_keys(doc: DocumentEvidence) -> set[str]:
    keys = set()
    for item in doc.contacts.get("emails", []):
        keys.add(item.value.lower())
    for item in doc.contacts.get("phones", []):
        keys.add(re.sub(r"[^\d+]", "", item.value))
    for item in doc.contacts.get("urls", []):
        url = item.value.lower()
        if "linkedin.com" in url:
            keys.add(url)
    return keys

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from an LLM response.

    DocWain's vLLM output is well-shaped JSON but frequently has:
      * a trailing ``.`` the model appends after the closing ``}``,
      * ``<think>...</think>`` blocks already stripped by vllm_manager
        but occasionally leaking a lone ``<think>`` when output hit
        max_tokens mid-reasoning,
      * markdown code fences (```` ```json ... ``` ````) when the
        model falls into "helpful" mode.
    ``json.loads(text)`` rejects any of these with "Extra data" /
    "Expecting value" — which used to send every LLM-enriched
    response straight to the deterministic fallback.
    """
    if not text:
        return None

    cleaned = text.strip()

    # Strip ``` code fences (```json ... ``` or ``` ... ```)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|JSON)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()

    # Strip leaked <think> artefacts.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = cleaned.lstrip("<think>").strip()

    # raw_decode returns the first valid JSON object and the index of
    # where it ended — tolerates any trailing garbage (period, prose,
    # another JSON block, etc.).
    decoder = json.JSONDecoder()
    # Try from the first ``{`` if there's text preceding.
    first_brace = cleaned.find("{")
    if first_brace != -1:
        try:
            obj, _ = decoder.raw_decode(cleaned[first_brace:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Final fallback: greedy ``{...}`` match.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None

__all__ = ["select_output_schema", "generate_structured_answer", "build_payload"]
