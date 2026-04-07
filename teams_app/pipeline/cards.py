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


def intelligence_card(
    filename: str,
    chunks_count: int,
    quality_grade: str,
    doc_type: str,
    summary: str,
    key_entities: List[str],
    questions: List[str],
) -> Dict[str, Any]:
    """Build a document intelligence card with LLM-generated insights and clickable questions.

    Shows: summary, doc type, entities, and 5 smart questions as action buttons.
    """
    quality_labels = {"A": "Excellent", "B": "Good", "C": "Fair"}
    quality_text = quality_labels.get(quality_grade, quality_grade)

    body: List[Dict[str, Any]] = [
        {"type": "TextBlock", "text": "DocWain Intelligence Report", "weight": "Bolder", "size": "Large", "color": "Accent"},
        {"type": "TextBlock", "text": f"\"{filename}\"", "weight": "Bolder", "size": "Medium", "spacing": "Small"},
        # Metadata row
        {
            "type": "ColumnSet",
            "spacing": "Small",
            "columns": [
                {"type": "Column", "width": "auto", "items": [
                    {"type": "TextBlock", "text": f"Type: {doc_type.replace('_', ' ').title()}", "size": "Small", "weight": "Bolder", "color": "Good"},
                ]},
                {"type": "Column", "width": "auto", "items": [
                    {"type": "TextBlock", "text": f"Indexed: {chunks_count} sections", "size": "Small", "isSubtle": True},
                ]},
                {"type": "Column", "width": "auto", "items": [
                    {"type": "TextBlock", "text": f"Quality: {quality_text}", "size": "Small", "isSubtle": True},
                ]},
            ],
        },
    ]

    # Summary section
    if summary:
        body.append({"type": "TextBlock", "text": "Summary", "weight": "Bolder", "size": "Small", "spacing": "Medium", "separator": True})
        body.append({"type": "TextBlock", "text": summary, "wrap": True, "size": "Small", "spacing": "None"})

    # Key entities
    if key_entities:
        entities_text = " | ".join(f"**{e}**" for e in key_entities[:5])
        body.append({"type": "TextBlock", "text": f"Key entities: {entities_text}", "wrap": True, "size": "Small", "spacing": "Small", "isSubtle": True})

    # Questions header
    if questions:
        body.append({
            "type": "TextBlock",
            "text": "Ask DocWain about this document:",
            "weight": "Bolder",
            "size": "Small",
            "color": "Accent",
            "spacing": "Medium",
            "separator": True,
        })

    # Build question buttons using Hero Card-style actions (messageBack)
    # These send the question as a regular user message — no Action.Submit invoke issues
    card_actions = []
    for i, q in enumerate(questions[:5]):
        if q and q.strip():
            title = q if len(q) <= 45 else q[:42] + "..."
            card_actions.append({
                "type": "Action.Submit",
                "title": title,
                "data": {
                    "msteams": {
                        "type": "messageBack",
                        "text": q,
                        "displayText": q,
                    }
                },
            })

    if not card_actions:
        for q in [
            "Provide a comprehensive summary of this document",
            "Extract all key names, dates, amounts, and important data points",
            "What are the most important findings and takeaways?",
        ]:
            card_actions.append({
                "type": "Action.Submit",
                "title": q[:42] + "..." if len(q) > 45 else q,
                "data": {
                    "msteams": {
                        "type": "messageBack",
                        "text": q,
                        "displayText": q,
                    }
                },
            })

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.5",
        "body": body,
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
