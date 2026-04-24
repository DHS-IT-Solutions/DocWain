"""Prompt registry exposes known DocWain prompts by name."""
import pytest


def test_registry_lists_known_prompts():
    from src.docwain.prompts.registry import list_prompts
    names = list_prompts()
    expected = {
        "entity_extraction",
        "researcher",
        "chart_generation",
        "docintel_classifier",
        "docintel_coverage_verifier",
        "vision_extractor",
    }
    missing = expected - set(names)
    assert not missing, f"registry missing prompts: {missing}"


def test_get_prompt_returns_non_empty_string():
    from src.docwain.prompts.registry import get_prompt
    text = get_prompt("entity_extraction")
    assert isinstance(text, str)
    assert len(text) > 50


def test_get_prompt_raises_on_unknown_name():
    from src.docwain.prompts.registry import get_prompt
    with pytest.raises(KeyError):
        get_prompt("nonexistent_prompt")


def test_all_registered_prompts_resolve():
    from src.docwain.prompts.registry import get_prompt, list_prompts
    for name in list_prompts():
        text = get_prompt(name)
        assert isinstance(text, str) and text, f"prompt {name!r} resolved to empty/non-string"
