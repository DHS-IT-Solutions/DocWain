"""Human rating CSV export/import.

Domain experts rate responses on a 1–5 SME scale in CSV. The import step
validates ratings and returns a {query_id: int} mapping that the baseline
snapshot merges into its human_rated_sme_score_avg.
"""
from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from scripts.sme_eval.schema import EvalResult

_COLUMNS = (
    "query_id",
    "profile_domain",
    "query_text",
    "response_text",
    "sme_score_1_to_5",
    "rater_notes",
)


def export_for_rating(results: Iterable[EvalResult], out_path: Path | str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "query_id": r.query.query_id,
                    "profile_domain": r.query.profile_domain,
                    "query_text": r.query.query_text,
                    "response_text": r.response_text,
                    "sme_score_1_to_5": "",
                    "rater_notes": "",
                }
            )


def import_ratings(csv_path: Path | str) -> dict[str, int]:
    """Return {query_id: rating}. Silently drops blanks and out-of-range scores."""
    path = Path(csv_path)
    out: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("sme_score_1_to_5") or "").strip()
            if not raw:
                continue
            try:
                score = int(raw)
            except ValueError:
                continue
            if score < 1 or score > 5:
                continue
            out[row["query_id"]] = score
    return out
