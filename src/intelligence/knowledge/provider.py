"""KnowledgeProvider — sanctioned KB lookup for adapters.

KBs are static JSON files. v1 ships bundled KBs; v2+ may load from Blob.
Per spec Section 8: KB augments interpretation, never adds claims to
insight bodies.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


class KbNotFound(FileNotFoundError):
    pass


class KnowledgeProvider(Protocol):
    @property
    def kb_id(self) -> str: ...
    def lookup(self, term: str) -> Optional[str]: ...
    def interpret(self, value: Any) -> Optional[str]: ...


@dataclass
class JsonKnowledgeProvider:
    kb_id: str
    version: str
    entries: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load_from_path(cls, path: str, *, kb_id: str) -> "JsonKnowledgeProvider":
        if not os.path.exists(path):
            raise KbNotFound(path)
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh) or {}
        return cls(
            kb_id=kb_id,
            version=str(raw.get("version") or "1.0"),
            entries=dict(raw.get("entries") or {}),
        )

    def lookup(self, term: str) -> Optional[str]:
        return self.entries.get(term.strip().lower()) or self.entries.get(term)

    def interpret(self, value: Any) -> Optional[str]:
        return self.lookup(str(value))
