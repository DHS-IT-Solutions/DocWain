"""V2 extraction engine — unified extraction via DocWain V2 model (Ollama)."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DEFAULT_RESULT = {
    "think": "",
    "entities": [],
    "tables": [],
    "fields": {},
    "confidence": 0.0,
}


def _build_extraction_prompt(doc_type: str = "unknown", page_type: str = "body") -> str:
    """Build the structured extraction prompt for the V2 model.

    The prompt tells the model to:
    - Reason first inside <think> tags
    - Extract entities, tables, and key-value fields
    - Flag any detected inconsistencies
    - Return a JSON object with the required keys
    """
    return (
        f"You are DocWain V2, an expert document extraction model.\n"
        f"Document type: {doc_type}\n"
        f"Page type: {page_type}\n\n"
        "Instructions:\n"
        "1. First, reason about the document inside <think> tags. Consider the document "
        "structure, content, and any potential issues before producing output.\n"
        "2. Extract all named entities. For each entity include: name, type, confidence.\n"
        "3. Extract all tables. For each table include: headers (list of strings), "
        "rows (list of lists).\n"
        "4. Extract key-value fields as a flat dictionary.\n"
        "5. Set a top-level confidence score (0.0–1.0) reflecting overall extraction quality.\n"
        "6. If you detect inconsistencies or anomalies, include an 'inconsistencies' list "
        "inside the JSON.\n\n"
        "Respond ONLY with a valid JSON object using exactly these keys:\n"
        "  think        - string: your reasoning from the <think> block\n"
        "  entities     - list of {name, type, confidence}\n"
        "  tables       - list of {headers, rows}\n"
        "  fields       - dict of key-value pairs\n"
        "  confidence   - float 0.0–1.0\n"
        "Do not include any text outside the JSON object."
    )


def _call_v2_model(
    prompt: str,
    images: list | None = None,
    *,
    ollama_host: str,
    model: str,
    timeout: int = 120,
) -> dict:
    """Call the Ollama chat API and parse the JSON response.

    Falls back to a default dict with confidence=0.3 if JSON parsing fails,
    or confidence=0.0 if an exception is raised.
    """
    default_parse_failure: dict[str, Any] = {
        "think": "",
        "entities": [],
        "tables": [],
        "fields": {},
        "confidence": 0.3,
    }

    try:
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = images

        payload = {
            "model": model,
            "messages": [message],
            "stream": False,
        }

        response = requests.post(
            f"{ollama_host}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()
        raw_content: str = data.get("message", {}).get("content", "")

        # Extract the JSON object from the response: find first { to last }
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start == -1 or end == -1 or end < start:
            logger.warning("V2 model response contained no valid JSON object; using defaults")
            return default_parse_failure

        json_str = raw_content[start : end + 1]
        parsed = json.loads(json_str)

        # Ensure all required keys are present
        result = {
            "think": parsed.get("think", ""),
            "entities": parsed.get("entities", []),
            "tables": parsed.get("tables", []),
            "fields": parsed.get("fields", {}),
            "confidence": float(parsed.get("confidence", 0.3)),
        }
        return result

    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse V2 model JSON response: %s", exc)
        return default_parse_failure
    except Exception as exc:  # noqa: BLE001
        logger.error("V2 model call failed: %s", exc)
        return {
            "think": "",
            "entities": [],
            "tables": [],
            "fields": {},
            "confidence": 0.0,
        }


class V2Extractor:
    """Unified extraction using the DocWain V2 model via Ollama.

    Combines reasoning, entity extraction, table detection, and field extraction
    in a single model call, with optional multi-modal image support.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "docwain:v2",
    ) -> None:
        self.ollama_host = ollama_host
        self.model = model

    def extract(
        self,
        document_bytes: bytes,
        file_type: str,
        page_images: list | None = None,
        *,
        doc_type: str = "unknown",
        page_type: str = "body",
        text_content: str | None = None,
    ) -> dict:
        """Run V2 extraction for a document page or full document.

        Args:
            document_bytes: Raw document bytes (used for context / future use).
            file_type: File extension or MIME type (e.g. "pdf", "png").
            page_images: Optional list of base-64-encoded page images.
            doc_type: High-level document type (e.g. "invoice", "contract").
            page_type: Page role within the document (e.g. "body", "cover", "table").
            text_content: Pre-extracted text to include in the prompt context.

        Returns:
            dict with keys: think, entities, tables, fields, confidence
        """
        prompt = _build_extraction_prompt(doc_type=doc_type, page_type=page_type)

        if text_content:
            prompt = f"{prompt}\n\nDocument text:\n{text_content}"

        result = _call_v2_model(
            prompt,
            images=page_images,
            ollama_host=self.ollama_host,
            model=self.model,
        )

        logger.info(
            "V2 extraction complete — doc_type=%s page_type=%s "
            "entities=%d tables=%d confidence=%.2f",
            doc_type,
            page_type,
            len(result.get("entities", [])),
            len(result.get("tables", [])),
            result.get("confidence", 0.0),
        )
        return result
