"""Ten-query smoke test — the third Batch-0 exit criterion.

Runs ten canned queries (one per major intent) against the owner's
fixture profile, asserts each returns:
  * non-empty response
  * grounded=True when the intent requires evidence
  * DocWain persona present in greeting/identity responses
  * no "I'm having trouble" static fallback text (would indicate
    upstream failure masked by the fallback path)

Marked @pytest.mark.live and skipped unless DOCWAIN_SMOKE_PROFILE is
set in the environment, because it needs a live vLLM + qdrant + mongo.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live

SMOKE_PROFILE = os.environ.get("DOCWAIN_SMOKE_PROFILE")
SMOKE_SUB = os.environ.get("DOCWAIN_SMOKE_SUB")

skip_reason = "Set DOCWAIN_SMOKE_PROFILE and DOCWAIN_SMOKE_SUB to run."

CANNED_QUERIES = [
    ("greet",     "Hello",                        False, True),
    ("identity",  "Who are you?",                 False, True),
    ("lookup",    "What is the invoice total on INV-778899?", True,  False),
    ("list",      "List all invoices from Acme Corp.",        True,  False),
    ("count",     "How many documents do I have uploaded?",   True,  False),
    ("extract",   "Extract all line items from the latest invoice.", True, False),
    ("summarize", "Summarize the most recent document.",      True,  False),
    ("compare",   "Compare the two most recent invoices.",    True,  False),
    ("analyze",   "What does the data suggest about spending trends?", True, False),
    ("timeline",  "Show the timeline of invoices received.",  True,  False),
]


@pytest.mark.skipif(not (SMOKE_PROFILE and SMOKE_SUB), reason=skip_reason)
@pytest.mark.parametrize("intent,query,needs_grounding,needs_persona", CANNED_QUERIES)
def test_smoke_query(intent, query, needs_grounding, needs_persona):
    from src.query.pipeline import run_query_pipeline
    from src.api.app_lifespan import get_clients_for_smoke  # provided by app lifecycle

    clients = get_clients_for_smoke()
    result = run_query_pipeline(
        query=query,
        profile_id=SMOKE_PROFILE,
        subscription_id=SMOKE_SUB,
        clients=clients,
    )
    assert result.response, f"Empty response for intent={intent} query={query!r}"
    assert "I'm having trouble" not in result.response, (
        f"Static fallback surfaced for intent={intent} — upstream failure masked."
    )
    if needs_grounding:
        assert result.context_found, f"No context found for intent={intent} query={query!r}"
    if needs_persona:
        assert "DocWain" in result.response, (
            f"DocWain persona missing in intent={intent} response: {result.response[:200]}"
        )
