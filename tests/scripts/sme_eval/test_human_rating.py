from datetime import datetime
from pathlib import Path

from scripts.sme_eval.human_rating import export_for_rating, import_ratings
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, domain="finance"):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text=f"query_{qid}",
            intent="analyze",
            profile_domain=domain,
            subscription_id="s",
            profile_id="p",
        ),
        response_text=f"response_{qid}",
        sources=[],
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_export_creates_csv(tmp_path: Path):
    out = tmp_path / "rate_me.csv"
    export_for_rating([_result("a"), _result("b")], out)
    assert out.exists()
    content = out.read_text()
    assert "query_id" in content
    assert "sme_score_1_to_5" in content
    assert "query_a" in content


def test_import_ratings_parses_csv(tmp_path: Path):
    csv_path = tmp_path / "rated.csv"
    csv_path.write_text(
        "query_id,profile_domain,query_text,response_text,sme_score_1_to_5,rater_notes\n"
        "query_a,finance,q,r,4,good voice\n"
        "query_b,finance,q,r,3,missing citations\n"
    )
    ratings = import_ratings(csv_path)
    assert ratings == {"query_a": 4, "query_b": 3}


def test_import_ratings_ignores_blank_scores(tmp_path: Path):
    csv_path = tmp_path / "rated.csv"
    csv_path.write_text(
        "query_id,profile_domain,query_text,response_text,sme_score_1_to_5,rater_notes\n"
        "query_a,finance,q,r,,no rating\n"
        "query_b,finance,q,r,5,excellent\n"
    )
    ratings = import_ratings(csv_path)
    assert ratings == {"query_b": 5}


def test_import_ratings_validates_score_range(tmp_path: Path):
    csv_path = tmp_path / "rated.csv"
    csv_path.write_text(
        "query_id,profile_domain,query_text,response_text,sme_score_1_to_5,rater_notes\n"
        "query_a,finance,q,r,7,out of range\n"
    )
    ratings = import_ratings(csv_path)
    # Out-of-range ratings dropped, not crashed
    assert ratings == {}
