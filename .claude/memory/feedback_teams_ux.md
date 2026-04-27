---
name: Teams UX Rules
description: Teams bot UX lessons — card-only progress, messageBack buttons, plain text responses, auto-clear on upload
type: feedback
---

Several Teams UX rules learned through iteration:

1. **Card-only updates** — Never send both text AND card for the same message. Card only, updated in-place.
2. **messageBack buttons** — Use `msteams.messageBack` type for card action buttons, NOT `Action.Submit`. Action.Submit triggers an invoke that causes "Something went wrong" if the bot responds with text instead of a card update.
3. **Plain text responses** — Send query responses as plain text messages, not Adaptive Cards. Teams renders markdown natively (bold, headers, lists). Adaptive Card TextBlocks don't handle long Reasoner output.
4. **Tables → bullet lists** — Convert markdown tables to structured bullet lists for Teams display (`_format_for_teams()`).
5. **Auto-clear on upload** — Each new file upload should clear old embeddings. Responses must be grounded in the current document, not mixed with stale data from previous uploads.
6. **Show intelligence** — After embedding, generate an LLM-powered intelligence report with summary, entities, and 5 document-specific questions as clickable buttons. Showcases DocWain's intelligence.
7. **Clear command** — Provide "clear all" / "reset" command for manual cleanup.

**Why:** These were all discovered through production testing where the initial implementations caused user confusion, error messages, or incorrect responses.

**How to apply:** Follow these patterns whenever modifying `teams_app/bot/handler.py` or `teams_app/pipeline/cards.py`.
