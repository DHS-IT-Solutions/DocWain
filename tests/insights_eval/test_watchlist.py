from datetime import datetime, timedelta, timezone

from src.intelligence.adapters.schema import (
    Adapter, AppliesWhen, ResearcherSection, KnowledgeConfig, Watchlist,
)
from src.intelligence.researcher_v2.watchlist import (
    evaluate_watchlists,
    WatchlistFiring,
)


def _adapter_with_watch(eval_expr: str) -> Adapter:
    return Adapter(
        name="generic", version="1.0", description="t",
        applies_when=AppliesWhen(),
        researcher=ResearcherSection(),
        knowledge=KnowledgeConfig(),
        watchlists=[Watchlist(
            id="renewal_due",
            description="renewal soon",
            eval=eval_expr,
            fires_insight_type="next_action",
        )],
    )


def test_eval_expr_renewal_due_fires():
    near = (datetime.now(tz=timezone.utc) + timedelta(days=10)).isoformat()
    docs = [{"document_id": "D1", "fields": {"policy_end_date": near}}]
    a = _adapter_with_watch("expr:doc.policy_end_date - now < 60d")
    fired = evaluate_watchlists(adapter=a, documents=docs)
    assert len(fired) == 1
    f = fired[0]
    assert isinstance(f, WatchlistFiring)
    assert f.watchlist_id == "renewal_due"
    assert f.document_id == "D1"


def test_eval_expr_does_not_fire_when_far_away():
    far = (datetime.now(tz=timezone.utc) + timedelta(days=120)).isoformat()
    docs = [{"document_id": "D1", "fields": {"policy_end_date": far}}]
    a = _adapter_with_watch("expr:doc.policy_end_date - now < 60d")
    fired = evaluate_watchlists(adapter=a, documents=docs)
    assert fired == []


def test_unsupported_expr_skipped():
    docs = [{"document_id": "D1", "fields": {}}]
    a = _adapter_with_watch("expr:bogus_function()")
    fired = evaluate_watchlists(adapter=a, documents=docs)
    assert fired == []
