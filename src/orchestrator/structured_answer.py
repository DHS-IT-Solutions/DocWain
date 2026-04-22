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
    try:
        if llm_client is not None:
            text = llm_client.generate(prompt, max_tokens=_LLM_ANSWER_MAX_TOKENS)
        else:
            from src.llm.gateway import get_llm_gateway
            text = get_llm_gateway().generate(prompt, max_tokens=_LLM_ANSWER_MAX_TOKENS)
        text = (text or "").strip()
        payload = _extract_json(text)
        # Minimal validation: parseable JSON + required top-level fields.
        # The strict per-item consistency checks in validate_response_payload
        # were written for the deterministic-path output shape, which uses
        # document_id on every leaf item. Since _compact_payload_for_prompt
        # drops those leaf document_id fields to keep the prompt under the
        # 32K context limit, the LLM echoes back the compact shape, which
        # the strict validator rejects — sending every LLM-enriched
        # response back to the deterministic fallback.
        if payload and isinstance(payload, dict) and _has_top_level_required(payload, schema):
            return json.dumps(payload, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Structured answer generation failed: %s", exc)
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

def _compare_documents(graph: ProfileEvidenceGraph) -> List[Dict[str, Any]]:
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

def _build_prompt(user_query: str, schema_name: str, base_payload: Dict[str, Any]) -> str:
    # Compact evidence before embedding: the raw base_payload carries per-item
    # snippet/chunk_id/meta/document_id/source_name that are redundant once
    # pulled into the prompt (document-level fields already scope them). For a
    # 12-doc profile the uncompacted payload is ~70K tokens, which overflows
    # DocWain's 32K context window and causes vLLM to reject with 400.
    # Compact form targets <=16K tokens for profiles up to ~25 docs.
    compact = _compact_payload_for_prompt(base_payload)
    schema_json = json.dumps(compact, separators=(",", ":"))
    return (
        "You are an evidence-first synthesis engine. "
        "Use only the provided structured evidence. "
        "Do not add new facts. "
        "Return ONLY valid JSON following the given schema.\n\n"
        f"Schema name: {schema_name}\n"
        f"User query: {user_query}\n\n"
        "Structured evidence (fill in narrative fields as needed):\n"
        f"{schema_json}\n"
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


def _compact_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    out: Dict[str, Any] = {}
    if item.get("value") is not None:
        out["v"] = item["value"]
    p_start = item.get("page_start")
    p_end = item.get("page_end")
    if p_start is not None:
        out["p"] = str(p_start) if (p_end is None or p_end == p_start) else f"{p_start}-{p_end}"
    sec = item.get("section_title")
    if sec and sec != item.get("value"):
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
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

__all__ = ["select_output_schema", "generate_structured_answer", "build_payload"]
