"""Tests for the sandbox runner wrapper (user Task 16).

The wrapper guards the live Phase 0 harness behind an API reachability
probe and falls back to a synthetic snapshot in dev environments. These
tests lock the contract:

* ``--dry-run`` emits a ``skipped`` snapshot and exits 0.
* Unreachable API emits a ``skipped`` snapshot and exits 0.
* A live snapshot with ``sme_artifact_hit_rate > 0`` exits 0.
* A live snapshot with ``sme_artifact_hit_rate == 0`` exits 1.
* A live snapshot with ``hallucination_rate > 0`` exits 1.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from scripts.sme_eval import run_sandbox


def test_dry_run_writes_skipped_snapshot(tmp_path: Path) -> None:
    out = tmp_path / "sandbox.json"
    rc = run_sandbox.main(["--dry-run", "--out", str(out)])
    assert rc == 0
    snapshot = json.loads(out.read_text(encoding="utf-8"))
    assert snapshot["status"] == "skipped"
    assert snapshot["reason"] == "dry-run"
    assert snapshot["metrics"]["sme_artifact_hit_rate"]["status"] == "skipped"


def test_unreachable_api_writes_skipped_snapshot(tmp_path: Path) -> None:
    out = tmp_path / "sandbox.json"
    with patch.object(run_sandbox, "_probe_api", return_value=False):
        rc = run_sandbox.main(
            ["--api-base-url", "http://127.0.0.1:99", "--out", str(out)]
        )
    assert rc == 0
    snapshot = json.loads(out.read_text(encoding="utf-8"))
    assert snapshot["status"] == "skipped"
    assert "api unreachable" in snapshot["reason"]


def _stub_baseline(out: Path, live_snapshot: dict):
    def _run(argv):
        out.write_text(json.dumps(live_snapshot), encoding="utf-8")
        return 0

    return _run


def test_live_run_with_positive_hit_rate_exits_ok(tmp_path: Path) -> None:
    out = tmp_path / "sandbox.json"
    live_snapshot = {
        "status": "complete",
        "metrics": {
            "sme_artifact_hit_rate": {"value": 0.82, "status": "ok"},
            "hallucination_rate": {"value": 0.0, "status": "ok"},
        },
    }
    with patch.object(run_sandbox, "_probe_api", return_value=True), patch.object(
        run_sandbox, "_baseline_main", _stub_baseline(out, live_snapshot)
    ):
        rc = run_sandbox.main(["--out", str(out)])
    assert rc == 0
    snapshot = json.loads(out.read_text(encoding="utf-8"))
    assert snapshot["status"] == "complete"
    assert snapshot["metrics"]["sme_artifact_hit_rate"]["value"] > 0


def test_live_run_zero_hit_rate_exits_nonzero(tmp_path: Path) -> None:
    out = tmp_path / "sandbox.json"
    live_snapshot = {
        "status": "complete",
        "metrics": {
            "sme_artifact_hit_rate": {"value": 0.0, "status": "ok"},
        },
    }
    with patch.object(run_sandbox, "_probe_api", return_value=True), patch.object(
        run_sandbox, "_baseline_main", _stub_baseline(out, live_snapshot)
    ):
        rc = run_sandbox.main(["--out", str(out)])
    assert rc == 1


def test_live_run_hallucination_regression_exits_nonzero(
    tmp_path: Path,
) -> None:
    out = tmp_path / "sandbox.json"
    live_snapshot = {
        "status": "complete",
        "metrics": {
            "sme_artifact_hit_rate": {"value": 0.9, "status": "ok"},
            "hallucination_rate": {"value": 0.08, "status": "ok"},
        },
    }
    with patch.object(run_sandbox, "_probe_api", return_value=True), patch.object(
        run_sandbox, "_baseline_main", _stub_baseline(out, live_snapshot)
    ):
        rc = run_sandbox.main(["--out", str(out)])
    assert rc == 1


def test_default_out_path_has_date_stamp() -> None:
    import re

    path = run_sandbox._default_out_path()
    assert re.match(
        r"tests/sme_metrics_sandbox_\d{4}-\d{2}-\d{2}\.json$", str(path)
    )
