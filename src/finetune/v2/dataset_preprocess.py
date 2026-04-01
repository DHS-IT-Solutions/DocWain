"""Chat-format converters for DocWain V2 SFT training.

Converts raw dataset rows into the multi-turn chat format expected by the
V2 training pipeline (vision SFT, tool-call SFT, plain SFT).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# System prompt constant
# ---------------------------------------------------------------------------

DOCWAIN_V2_SYSTEM: str = (
    "You are DocWain, an enterprise document intelligence assistant. "
    "You can analyse document images, extract tables, charts, and text, "
    "and answer questions grounded in visual evidence. "
    "When specialised tools are available you may invoke them via "
    "tool-call blocks; otherwise answer directly from the document content."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _system_message(tools_json: str | None = None) -> Dict[str, Any]:
    """Build the system message, optionally embedding available tools."""
    content = DOCWAIN_V2_SYSTEM
    if tools_json:
        content += f"\n\nAvailable tools:\n{tools_json}"
    return {"role": "system", "content": content}


def _tool_call_block(call: Dict[str, Any]) -> str:
    """Serialise a single tool call into a ``<tool_call>`` block."""
    return (
        "<tool_call>\n"
        + json.dumps({"name": call["name"], "arguments": call.get("arguments", {})}, indent=2)
        + "\n</tool_call>"
    )


def _tool_response_block(result: Any) -> str:
    """Serialise a tool result into a ``<tool_response>`` block."""
    return (
        "<tool_response>\n"
        + json.dumps(result, indent=2, default=str)
        + "\n</tool_response>"
    )


# ---------------------------------------------------------------------------
# Public format functions
# ---------------------------------------------------------------------------


def format_vision_sft(
    image_path: str,
    question: str,
    answer: str,
    tools_json: str | None = None,
) -> Dict[str, Any]:
    """Create a 3-message vision SFT training pair.

    The user turn includes an ``<image>`` token so the vision encoder can
    inject the image embedding at that position during training.

    Returns
    -------
    dict with a ``messages`` list of ``[system, user, assistant]``.
    """
    return {
        "messages": [
            _system_message(tools_json),
            {
                "role": "user",
                "content": f"<image>{image_path}</image>\n{question}",
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
    }


def format_tool_call_sft(
    query: str,
    tool_calls: List[Dict[str, Any]],
    tool_results: List[Any],
    final_answer: str,
    tools_json: str | None = None,
) -> Dict[str, Any]:
    """Create a multi-turn tool-call SFT training pair.

    Supports parallel tool calls — each call gets its own ``<tool_call>``
    block inside the assistant turn, and each result its own
    ``<tool_response>`` block inside the subsequent tool turn.

    Returns
    -------
    dict with a ``messages`` list:
      [system, user, assistant(tool_calls), tool(results), assistant(final)]
    """
    # Build the assistant turn with one or more tool-call blocks
    call_blocks = "\n".join(_tool_call_block(tc) for tc in tool_calls)

    # Build the tool-response turn
    response_blocks = "\n".join(
        _tool_response_block(tr) for tr in tool_results
    )

    return {
        "messages": [
            _system_message(tools_json),
            {
                "role": "user",
                "content": query,
            },
            {
                "role": "assistant",
                "content": call_blocks,
            },
            {
                "role": "tool",
                "content": response_blocks,
            },
            {
                "role": "assistant",
                "content": final_answer,
            },
        ],
    }


def format_no_tool_sft(
    query: str,
    answer: str,
    tools_json: str | None = None,
) -> Dict[str, Any]:
    """Create a simple 3-message SFT pair (no tool use).

    Returns
    -------
    dict with a ``messages`` list of ``[system, user, assistant]``.
    """
    return {
        "messages": [
            _system_message(tools_json),
            {
                "role": "user",
                "content": query,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
    }


def format_cot_sft(
    question: str,
    reasoning: str,
    answer: str,
    *,
    image_path: Optional[str] = None,
    tools_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a chain-of-thought SFT training pair.

    The assistant turn wraps the reasoning inside ``<think>...</think>`` tags
    followed by the final answer, enabling the model to learn explicit
    reasoning traces.

    Parameters
    ----------
    question:
        The user's question or instruction.
    reasoning:
        The chain-of-thought reasoning steps.
    answer:
        The final answer that follows the reasoning.
    image_path:
        Optional path/URL to a document image; if provided, an
        ``<image>`` token is prepended to the user content.
    tools_json:
        Optional JSON string of available tools to embed in the system prompt.

    Returns
    -------
    dict with a ``messages`` list of ``[system, user, assistant]``.
    """
    user_content = (
        f"<image>{image_path}</image>\n{question}" if image_path else question
    )
    assistant_content = f"<think>\n{reasoning}\n</think>\n\n{answer}"

    return {
        "messages": [
            _system_message(tools_json),
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def format_dpo_pair(
    question: str,
    chosen_reasoning: str,
    chosen_answer: str,
    rejected_reasoning: str,
    rejected_answer: str,
    *,
    image_path: Optional[str] = None,
    tools_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a DPO training pair with chosen and rejected completions.

    Both completions include ``<think>...</think>`` blocks so the model learns
    to distinguish high-quality from low-quality reasoning traces.

    Parameters
    ----------
    question:
        The user's question or instruction.
    chosen_reasoning:
        Reasoning trace for the preferred (chosen) completion.
    chosen_answer:
        Final answer for the preferred completion.
    rejected_reasoning:
        Reasoning trace for the dis-preferred (rejected) completion.
    rejected_answer:
        Final answer for the dis-preferred completion.
    image_path:
        Optional path/URL to a document image; prepended to user content
        as an ``<image>`` token.
    tools_json:
        Optional JSON string of available tools to embed in the system prompt.

    Returns
    -------
    dict with keys:
      - ``prompt``: list of ``[system, user]`` messages (the shared prompt)
      - ``chosen``: string containing the chosen completion with think block
      - ``rejected``: string containing the rejected completion with think block
    """
    user_content = (
        f"<image>{image_path}</image>\n{question}" if image_path else question
    )
    prompt = [
        _system_message(tools_json),
        {"role": "user", "content": user_content},
    ]
    chosen = f"<think>\n{chosen_reasoning}\n</think>\n\n{chosen_answer}"
    rejected = f"<think>\n{rejected_reasoning}\n</think>\n\n{rejected_answer}"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def format_insight_sft(
    question: str,
    reasoning: str,
    insight_category: str,
    insight_text: str,
    answer: str,
    *,
    viz_directive: Optional[str] = None,
    tools_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an insight-aware SFT training pair.

    The assistant turn includes a ``<think>`` block, a structured insight
    annotation, an optional visualisation directive, and the final answer.
    This format trains the model to surface actionable business insights
    alongside document answers.

    Parameters
    ----------
    question:
        The user's question or instruction.
    reasoning:
        Chain-of-thought reasoning steps.
    insight_category:
        Short label for the insight type (e.g. ``"anomaly"``, ``"trend"``).
    insight_text:
        Human-readable insight sentence or paragraph.
    answer:
        The final answer to the question.
    viz_directive:
        Optional visualisation instruction (e.g. ``"bar_chart: revenue by quarter"``).
        When provided, a ``<viz>`` block is inserted between the insight and answer.
    tools_json:
        Optional JSON string of available tools to embed in the system prompt.

    Returns
    -------
    dict with a ``messages`` list of ``[system, user, assistant]``.
    """
    viz_block = f"\n<viz>{viz_directive}</viz>" if viz_directive else ""
    assistant_content = (
        f"<think>\n{reasoning}\n</think>\n\n"
        f"<insight category=\"{insight_category}\">\n{insight_text}\n</insight>"
        f"{viz_block}\n\n{answer}"
    )

    return {
        "messages": [
            _system_message(tools_json),
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ],
    }
