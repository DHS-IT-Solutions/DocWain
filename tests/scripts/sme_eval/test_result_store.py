"""Tests for the result store."""
from datetime import datetime
from pathlib import Path

from scripts.sme_eval.result_store import JsonlResultStore
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid="finance_001") -> EvalResult:
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent="lookup",
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text="r",
        sources=[],
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_store_appends_and_reads(tmp_path: Path):
    store = JsonlResultStore(tmp_path / "results.jsonl")
    store.append(_result("a"))
    store.append(_result("b"))

    loaded = list(store.iter_all())
    assert len(loaded) == 2
    assert {r.query.query_id for r in loaded} == {"a", "b"}


def test_store_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "nested" / "deeply" / "results.jsonl"
    store = JsonlResultStore(path)
    store.append(_result())
    assert path.exists()


def test_store_is_append_only(tmp_path: Path):
    store = JsonlResultStore(tmp_path / "r.jsonl")
    store.append(_result("a"))
    store.append(_result("b"))

    # Open a new store instance against same path — should read existing
    store2 = JsonlResultStore(tmp_path / "r.jsonl")
    loaded = list(store2.iter_all())
    assert len(loaded) == 2


def test_store_filter_by_run_id(tmp_path: Path):
    store = JsonlResultStore(tmp_path / "r.jsonl")
    r1 = _result("a")
    r2 = _result("b")
    r2 = r2.model_copy(update={"run_id": "run_other"})
    store.append(r1)
    store.append(r2)

    loaded = list(store.iter_run("run_test"))
    assert len(loaded) == 1
    assert loaded[0].query.query_id == "a"
