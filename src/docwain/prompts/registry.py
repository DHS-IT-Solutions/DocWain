"""Prompt registry — canonical index of DocWain's task-specific prompts.

Prompts live in whichever module owns the capability (co-location with the
caller). This registry provides discoverability: `list_prompts()` enumerates
everything known; `get_prompt(name)` returns the system-prompt string. No
migration — this module is introspection only.

Spec: 2026-04-24-unified-docwain-engineering-layer-design.md §5.2
"""
from __future__ import annotations

from importlib import import_module
from typing import Dict, List, Tuple


# name -> (module dotted path, attribute name)
_PROMPT_INDEX: Dict[str, Tuple[str, str]] = {
    "entity_extraction": ("src.docwain.prompts.entity_extraction", "ENTITY_EXTRACTION_SYSTEM_PROMPT"),
    "researcher": ("src.docwain.prompts.researcher", "RESEARCHER_SYSTEM_PROMPT"),
    "chart_generation": ("src.docwain.prompts.chart_generation", "CHART_GENERATION_SYSTEM_PROMPT"),
    "docintel_classifier": ("src.extraction.vision.docintel", "CLASSIFIER_SYSTEM_PROMPT"),
    "docintel_coverage_verifier": ("src.extraction.vision.docintel", "COVERAGE_SYSTEM_PROMPT"),
    "vision_extractor": ("src.extraction.vision.extractor", "EXTRACTOR_SYSTEM_PROMPT"),
}


def list_prompts() -> List[str]:
    """Return the sorted list of registered prompt names."""
    return sorted(_PROMPT_INDEX.keys())


def get_prompt(name: str) -> str:
    """Return the system-prompt string for the given registered name.

    Raises KeyError if the name is not registered.
    Raises ImportError / AttributeError if the module or attribute is missing.
    """
    if name not in _PROMPT_INDEX:
        raise KeyError(f"unknown prompt name: {name!r}. Known: {list_prompts()}")
    module_path, attr = _PROMPT_INDEX[name]
    module = import_module(module_path)
    value = getattr(module, attr)
    if not isinstance(value, str):
        raise TypeError(f"registered prompt {name!r} is not a string (got {type(value).__name__})")
    return value


def register_prompt(name: str, module_path: str, attribute: str) -> None:
    """Register a new prompt under the given name."""
    _PROMPT_INDEX[name] = (module_path, attribute)
