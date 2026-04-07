"""Teams-specific card builders for the standalone pipeline.

Builds proper Bot Framework Activity objects (not dicts) so they work
with turn_context.send_activity / update_activity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from botbuilder.schema import Activity, ActivityTypes, Attachment

logger = logging.getLogger(__name__)

ADAPTIVE_CARD_TYPE = "application/vnd.microsoft.card.adaptive"


def _card_activity(card: Dict[str, Any]) -> Activity:
    """Build an Activity with an Adaptive Card attachment. No text — card only."""
    return Activity(
        type=ActivityTypes.message,
        attachments=[
            Attachment(content_type=ADAPTIVE_CARD_TYPE, content=card)
        ],
    )


def progress_card(
    filename: str,
    step: str,
    detail: str,
    progress_pct: int,
) -> Dict[str, Any]:
    """Build a progress card showing pipeline stage."""
    bar_filled = int(progress_pct / 100 * 14)
    bar = "█" * bar_filled + "░" * (14 - bar_filled)
    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {
                "type": "ColumnSet",
                "columns": [
                    {
                        "type": "Column",
                        "width": "auto",
                        "items": [{"type": "TextBlock", "text": step, "weight": "Bolder", "size": "Medium", "color": "Accent"}],
                    },
                    {
                        "type": "Column",
                        "width": "stretch",
                        "items": [
                            {"type": "TextBlock", "text": filename, "weight": "Bolder", "size": "Medium"},
                            {"type": "TextBlock", "text": detail, "wrap": True, "size": "Small", "isSubtle": True, "spacing": "None"},
                        ],
                    },
                ],
            },
            {"type": "TextBlock", "text": f"[{bar}] {progress_pct}%", "fontType": "Monospace", "size": "Small", "spacing": "Small"},
        ],
    }


def completion_card(
    filename: str,
    chunks_count: int,
    quality_grade: str,
    doc_type: str,
    actions: List[Dict[str, str]],
    questions: List[str],
) -> Dict[str, Any]:
    """Build a document-ready card with content-aware actions."""
    quality_labels = {"A": "Excellent", "B": "Good", "C": "Fair"}
    quality_text = quality_labels.get(quality_grade, quality_grade)

    q_text = "\n".join(f"- {q}" for q in questions[:3]) if questions else "- What is this document about?"

    # Build action buttons from domain actions — no duplicates
    card_actions = []
    seen_titles = set()
    for act in actions[:3]:
        title = act.get("title", "")
        if title and title not in seen_titles:
            seen_titles.add(title)
            card_actions.append({
                "type": "Action.Submit",
                "title": title,
                "data": {"action": "domain_query", "query": act.get("query", title)},
            })

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {"type": "TextBlock", "text": "✅ Document Ready", "weight": "Bolder", "size": "Medium", "color": "Good"},
            {"type": "TextBlock", "text": f"\"{filename}\" is now ready for questions.", "wrap": True, "spacing": "Small"},
            {
                "type": "ColumnSet",
                "spacing": "Small",
                "columns": [
                    {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": f"Type: {doc_type.title()}", "size": "Small", "isSubtle": True}]},
                    {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": f"Chunks: {chunks_count}", "size": "Small", "isSubtle": True}]},
                    {"type": "Column", "width": "auto", "items": [{"type": "TextBlock", "text": f"Quality: {quality_text}", "size": "Small", "isSubtle": True}]},
                ],
            },
            {
                "type": "Container",
                "separator": True,
                "spacing": "Medium",
                "items": [
                    {"type": "TextBlock", "text": "Try asking:", "weight": "Bolder", "size": "Small", "color": "Accent"},
                    {"type": "TextBlock", "text": q_text, "wrap": True, "size": "Small"},
                ],
            },
        ],
        "actions": card_actions,
    }


def error_card(filename: str, message: str) -> Dict[str, Any]:
    """Build an error card."""
    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": [
            {"type": "TextBlock", "text": "❌ Processing Failed", "weight": "Bolder", "color": "Attention"},
            {"type": "TextBlock", "text": message, "wrap": True, "size": "Small"},
        ],
    }


async def send_card(turn_context: Any, card: Dict[str, Any]) -> Optional[str]:
    """Send an Adaptive Card. Returns activity ID for in-place updates."""
    try:
        activity = _card_activity(card)
        resp = await turn_context.send_activity(activity)
        return getattr(resp, "id", None)
    except Exception as exc:
        logger.error("Failed to send card: %s", exc)
        return None


async def update_card(turn_context: Any, activity_id: Optional[str], card: Dict[str, Any]) -> Optional[str]:
    """Update a card in-place. Falls back to send if update fails."""
    if not activity_id:
        return await send_card(turn_context, card)

    try:
        activity = _card_activity(card)
        activity.id = activity_id
        await turn_context.update_activity(activity)
        return activity_id
    except Exception:
        pass

    # Fallback: delete + send
    try:
        await turn_context.delete_activity(activity_id)
    except Exception:
        pass
    return await send_card(turn_context, card)
