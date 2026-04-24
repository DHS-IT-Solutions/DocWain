"""DomainAdapter dataclass — the shape of a domain YAML after parsing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DomainAdapter:
    domain: str = "generic"
    version: str = "v1"
    prompt_fragment: str = ""
    key_entities: List[str] = field(default_factory=list)
    analysis_hints: Dict[str, Any] = field(default_factory=dict)
    questions_to_ask: List[str] = field(default_factory=list)
