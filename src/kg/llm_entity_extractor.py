"""LLM-driven entity and relationship extraction via Ollama.

This module replaces the primary entity extraction path (regex+spaCy) with an
LLM-first approach.  The existing EntityExtractor is retained as a
cross-validation layer inside ``validate_entities``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests

from src.kg.ontology import get_domain_relationships

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas (documentation only — validated at runtime by callers)
# ---------------------------------------------------------------------------
# Entity:
#   {"name": str, "type": str, "aliases": list[str], "confidence": float}
#
# Relationship:
#   {"source": str, "target": str, "type": str, "evidence": str,
#    "confidence": float, "temporal_bounds": dict | None}

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "docwain:v2"

# Confidence threshold above which LLM entities bypass cross-validation
_HIGH_CONFIDENCE_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_extraction_prompt(text: str, domain: str = "generic") -> str:
    """Build the extraction prompt incorporating domain relationship types.

    Parameters
    ----------
    text:
        The document text to extract from.
    domain:
        One of the domains defined in src.kg.ontology (legal, financial, hr,
        medical, generic).

    Returns
    -------
    str
        A fully-formed prompt ready to send to the LLM.
    """
    rel_types = get_domain_relationships(domain)
    rel_list = ", ".join(rel_types) if rel_types else "related_to"

    prompt = (
        f"You are an expert information-extraction engine.\n"
        f"Domain: {domain}\n\n"
        f"Extract all named entities and their relationships from the text below.\n\n"
        f"ENTITY output schema (JSON array under key \"entities\"):\n"
        f"  {{\"name\": string, \"type\": string, \"aliases\": [string, ...], "
        f"\"confidence\": float 0-1}}\n\n"
        f"RELATIONSHIP output schema (JSON array under key \"relationships\"):\n"
        f"  {{\"source\": string, \"target\": string, \"type\": string, "
        f"\"evidence\": string, \"confidence\": float 0-1, "
        f"\"temporal_bounds\": {{\"start\": string, \"end\": string}} | null}}\n\n"
        f"Allowed relationship types for this domain: {rel_list}\n\n"
        f"Return ONLY valid JSON with exactly two top-level keys: "
        f"\"entities\" and \"relationships\".\n"
        f"Do not include any text outside the JSON object.\n\n"
        f"TEXT:\n{text}"
    )
    return prompt


def _call_llm(
    prompt: str,
    *,
    ollama_host: str = _DEFAULT_HOST,
    model: str = _DEFAULT_MODEL,
) -> dict:
    """Send *prompt* to Ollama and return the parsed JSON response.

    Parameters
    ----------
    prompt:
        The fully-formed extraction prompt.
    ollama_host:
        Base URL of the Ollama service.
    model:
        Model tag to use for generation.

    Returns
    -------
    dict
        Parsed JSON with ``entities`` and ``relationships`` keys.

    Raises
    ------
    requests.RequestException
        On HTTP-level failures.
    ValueError
        When the LLM response cannot be parsed as valid extraction JSON.
    """
    url = f"{ollama_host.rstrip('/')}/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()

    raw: str = response.json().get("response", "")

    # Attempt to extract a JSON object from the response even if there is
    # surrounding prose.
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON object found in LLM response: {raw!r}")

    parsed: dict = json.loads(json_match.group())

    if "entities" not in parsed:
        parsed["entities"] = []
    if "relationships" not in parsed:
        parsed["relationships"] = []

    return parsed


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LLMEntityExtractor:
    """Extract entities and relationships from text using an Ollama-hosted LLM.

    Parameters
    ----------
    ollama_host:
        Base URL of the Ollama REST API.
    model:
        Model tag to request from Ollama.
    """

    def __init__(
        self,
        ollama_host: str = _DEFAULT_HOST,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self.ollama_host = ollama_host
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str, domain: str = "generic") -> dict:
        """Extract entities and relationships from *text*.

        Parameters
        ----------
        text:
            Raw document text.
        domain:
            Ontology domain hint (legal, financial, hr, medical, generic).

        Returns
        -------
        dict
            ``{"entities": [...], "relationships": [...]}``
        """
        prompt = _build_extraction_prompt(text, domain=domain)
        result = _call_llm(prompt, ollama_host=self.ollama_host, model=self.model)
        return result

    def validate_entities(
        self,
        llm_entities: List[dict],
        original_text: str,
    ) -> List[dict]:
        """Cross-validate LLM-extracted entities against regex+spaCy.

        Rules
        -----
        * Entities confirmed by both LLM **and** regex/spaCy receive
          ``cross_validated=True``.
        * High-confidence LLM entities (``confidence >= 0.8``) pass through
          without cross-validation (``cross_validated=False``).
        * Low-confidence entities that lack regex/spaCy confirmation are
          rejected (excluded from the output).

        Parameters
        ----------
        llm_entities:
            List of entity dicts produced by :meth:`extract`.
        original_text:
            The original document text (used to run the baseline extractor).

        Returns
        -------
        list[dict]
            Validated entity dicts, each augmented with a ``cross_validated``
            boolean key.
        """
        from src.kg.entity_extractor import EntityExtractor

        baseline_extractor = EntityExtractor(use_nlp=True)
        baseline_entities: List = baseline_extractor.extract_with_metadata(original_text)

        # Build a set of normalised baseline names for fast lookup
        baseline_names: set[str] = {
            e.normalized_name for e in baseline_entities
        }
        # Also index by type-normalised pairs for slightly stricter matching
        baseline_type_names: set[tuple[str, str]] = {
            (e.type.upper(), e.normalized_name) for e in baseline_entities
        }

        validated: List[dict] = []

        for entity in llm_entities:
            name: str = (entity.get("name") or "").strip()
            entity_type: str = (entity.get("type") or "").upper()
            confidence: float = float(entity.get("confidence", 0.0))
            normalized_name: str = " ".join(name.lower().split())

            in_baseline = (
                normalized_name in baseline_names
                or (entity_type, normalized_name) in baseline_type_names
            )

            if in_baseline:
                # Confirmed by both sources
                validated.append({**entity, "cross_validated": True})
            elif confidence >= _HIGH_CONFIDENCE_THRESHOLD:
                # High-confidence LLM entity — pass without cross-validation
                validated.append({**entity, "cross_validated": False})
            else:
                # Low confidence and not confirmed — reject
                logger.debug(
                    "Rejected low-confidence unconfirmed entity: %r (confidence=%.2f)",
                    name,
                    confidence,
                )

        return validated
