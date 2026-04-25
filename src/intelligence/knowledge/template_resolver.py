"""Resolves {{kb.lookup('term')}} directives in researcher prompts.

Used by Researcher Agent v2 to inject KB-derived interpretation into
prompts before LLM call. The resolved text becomes part of the prompt;
the model sees facts + KB-augmented context together but is instructed
to keep KB content out of insight bodies (OQ1 separation rule).
"""
from __future__ import annotations

import re
from typing import Optional

from src.intelligence.knowledge.provider import KnowledgeProvider

_DIRECTIVE = re.compile(r"\{\{kb\.lookup\(['\"]([^'\"]+)['\"]\)\}\}")


def resolve_template(text: str, *, kb: Optional[KnowledgeProvider]) -> str:
    def _replace(match):
        term = match.group(1)
        if kb is None:
            return ""
        v = kb.lookup(term)
        return v if v is not None else ""
    return _DIRECTIVE.sub(_replace, text)
