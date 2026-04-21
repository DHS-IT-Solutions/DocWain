"""Prompt-path contract: user-visible response formatting instructions must
live in src/generation/prompts.py, never in src/intelligence/generator.py.

Rule source: user feedback memory feedback_prompt_paths.md.
"""
from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_intelligence_generator_has_no_format_instructions_dict():
    """`_FORMAT_INSTRUCTIONS` dict (table/bullets/sections/numbered/prose)
    must not live in intelligence/generator.py — it belongs in prompts.py.
    """
    path = REPO_ROOT / "src" / "intelligence" / "generator.py"
    if not path.exists():
        return  # whole file deleted — rule satisfied vacuously
    text = path.read_text(encoding="utf-8")
    # Look for the dict DEFINITION, not an import / re-export of the name.
    assert "_FORMAT_INSTRUCTIONS = {" not in text, (
        "src/intelligence/generator.py still defines _FORMAT_INSTRUCTIONS. "
        "Move the dict to src/generation/prompts.py (or re-export via a "
        "from-import)."
    )


def test_generation_prompts_has_format_instructions():
    """prompts.py must own the format templates that intelligence/generator
    used to own. This catches an accidental deletion that leaves no format
    templates anywhere.
    """
    path = REPO_ROOT / "src" / "generation" / "prompts.py"
    text = path.read_text(encoding="utf-8")
    for key in ("table", "bullets", "sections", "numbered", "prose"):
        assert key in text, (
            f"src/generation/prompts.py is missing format key {key!r}. "
            "After Task 11 all five format instructions should live here."
        )


def test_reasoning_engine_imports_prompts_not_intelligence_generator():
    """reasoning_engine.py should not construct IntelligentGenerator any
    more — it should use build_reason_prompt from generation.prompts (or
    the Reasoner from src.generation.reasoner).
    """
    path = REPO_ROOT / "src" / "intelligence" / "reasoning_engine.py"
    text = path.read_text(encoding="utf-8")
    assert "from src.intelligence.generator import IntelligentGenerator" not in text, (
        "reasoning_engine.py still imports IntelligentGenerator. "
        "Switch to src.generation.prompts.build_reason_prompt + "
        "src.generation.reasoner."
    )
