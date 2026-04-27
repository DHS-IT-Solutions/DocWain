"""Tests for the prompt-clamp helpers introduced for UAT Issue #3.

Goal: prevent vLLM VLLMValidationError on prompts that don't fit the
model context window. Fail loud with PromptTooLargeError before sending
to vLLM; let the gateway fall back to a wider-context client.
"""
import pytest

from src.llm.clients import (
    PromptTooLargeError,
    estimate_prompt_tokens,
    clamp_max_tokens,
)


def test_estimate_returns_zero_on_empty():
    assert estimate_prompt_tokens("") == 0


def test_estimate_grows_with_text_length():
    short = estimate_prompt_tokens("hello")
    long = estimate_prompt_tokens("hello " * 1000)
    assert long > short
    assert short >= 1
    # Sanity: 1000 "hello "s should be a few thousand tokens
    assert long >= 500


def test_clamp_returns_requested_when_room_available():
    # Tiny prompt, big window — full requested value passes through
    n = clamp_max_tokens(prompt="short prompt", requested=2048, ctx_window=32768)
    assert n == 2048


def test_clamp_reduces_when_prompt_is_large():
    # Prompt close to ctx_window — clamped down
    big_prompt = "word " * 6000  # ~6000 tokens via heuristic
    n = clamp_max_tokens(prompt=big_prompt, requested=8192, ctx_window=8192)
    # Expect clamped to fit window minus prompt minus safety
    assert n < 8192
    assert n > 0


def test_clamp_raises_when_prompt_alone_overflows():
    # Prompt larger than the entire window — must raise
    huge = "word " * 50_000
    with pytest.raises(PromptTooLargeError) as exc:
        clamp_max_tokens(prompt=huge, requested=2048, ctx_window=8192)
    assert exc.value.ctx_window == 8192
    assert exc.value.prompt_tokens > 8192


def test_clamp_raises_when_min_output_cannot_fit():
    # Prompt + safety + min_output > window
    prompt = "word " * 7900  # ~7900 tokens
    with pytest.raises(PromptTooLargeError):
        clamp_max_tokens(
            prompt=prompt, requested=512, ctx_window=8192,
            safety=64, min_output=512,
        )


def test_clamp_reproduces_uat_overflow_scenario():
    """The 2026-04-27 production VLLMValidationError was: prompt 17409 +
    requested 15360 = 32769 > window 32768. Reproduce + confirm clamp."""
    # Build a prompt that estimates near 17K tokens
    prompt = "word " * 17000
    # Estimate is heuristic; just verify the clamp brings the total under window
    n = clamp_max_tokens(
        prompt=prompt, requested=15360, ctx_window=32768, safety=256,
    )
    assert n + estimate_prompt_tokens(prompt) + 256 <= 32768


def test_clamp_handles_16gb_window_gracefully():
    """On the 16 GB GPU profile, max-model-len drops to 8192. Confirm the
    clamp helps in that regime — moderate prompts get reduced max_tokens
    rather than failing."""
    prompt = "word " * 4000  # ~4000 tokens
    n = clamp_max_tokens(prompt=prompt, requested=4096, ctx_window=8192)
    # Should leave at least ~3000 tokens for output (8192 - 4000 - 256 safety)
    assert n > 0
    assert n <= 4096


def test_PromptTooLargeError_carries_diagnostic_data():
    huge = "word " * 100_000
    with pytest.raises(PromptTooLargeError) as exc:
        clamp_max_tokens(prompt=huge, requested=512, ctx_window=8192)
    assert exc.value.ctx_window == 8192
    assert exc.value.prompt_tokens > 0
    assert exc.value.min_output_tokens == 64  # default
    assert "prompt requires" in str(exc.value)
