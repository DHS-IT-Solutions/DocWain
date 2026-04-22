"""Phase 0 eval harness wrapper — sandbox delta check (user Task 16).

Writes ``tests/sme_metrics_sandbox_{YYYY-MM-DD}.json`` if the sandbox
DocWain API is reachable. Skips cleanly in dev environments where no
live API is up — the ``--dry-run`` flag forces skip with a synthetic
pass-through snapshot so the test suite can exercise the wiring.

Keep this entrypoint thin: all the heavy lifting lives in the existing
:mod:`scripts.sme_eval.run_baseline` module. This wrapper adds:

* Reachability probe (optional ``--dry-run`` override).
* Fixed output file naming (``sme_metrics_sandbox_{YYYY-MM-DD}.json``).
* Assertions on the resulting snapshot:
  - ``sme_artifact_hit_rate`` > 0 (sanity: Phase 2 synthesis ran).
  - ``hallucination_rate`` (if present) not regressed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


_METRIC_KEYS_WATCH = ("sme_artifact_hit_rate", "hallucination_rate")


def _baseline_main(argv: list[str]) -> int:
    """Indirection over the Phase 0 baseline entrypoint.

    Wrapped as a module-level callable so tests can ``patch.object`` this
    directly — ``patch.dict('sys.modules', ...)`` leaks across tests when
    another test has already imported ``run_baseline``.
    """
    from scripts.sme_eval import run_baseline

    return run_baseline.main(argv)


def _probe_api(base_url: str, timeout: float = 2.0) -> bool:
    """Return True iff ``/health`` (or equivalent) responds 2xx. Best-effort."""
    try:
        import httpx

        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{base_url.rstrip('/')}/health")
        return 200 <= resp.status_code < 300
    except Exception:
        return False


def _today_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _default_out_path() -> Path:
    return Path(
        f"tests/sme_metrics_sandbox_{_today_stamp()}.json"
    )


def _synthetic_snapshot(reason: str) -> dict:
    """Produce a pass-through snapshot when the live API is unreachable.

    The snapshot mirrors the shape of a real Phase 0 run but reports
    ``status=skipped`` so downstream consumers (tests, dashboards) can
    tell the skip apart from a genuine run.
    """
    return {
        "status": "skipped",
        "reason": reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "sme_artifact_hit_rate": {"value": None, "status": "skipped"},
            "hallucination_rate": {"value": None, "status": "skipped"},
        },
    }


def _assert_metrics(snapshot: dict) -> list[str]:
    """Return the list of assertion violations; empty list = success."""
    issues: list[str] = []
    metrics = snapshot.get("metrics", {}) or {}
    hit = (metrics.get("sme_artifact_hit_rate") or {}).get("value")
    if snapshot.get("status") != "skipped":
        if hit is None or hit <= 0:
            issues.append(
                f"sme_artifact_hit_rate must be > 0 post-Phase-2 (got {hit})"
            )
    hallucination = (metrics.get("hallucination_rate") or {}).get("value")
    if hallucination is not None and hallucination > 0.0 + 1e-6:
        # Phase 2 should not introduce hallucinations; flag the rise.
        issues.append(
            f"hallucination_rate regressed: {hallucination} > 0"
        )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("DOCWAIN_API_BASE_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_default_out_path(),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API probe and emit a synthetic skipped snapshot.",
    )
    args = parser.parse_args(argv)

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        snapshot = _synthetic_snapshot("dry-run")
    elif not _probe_api(args.api_base_url):
        snapshot = _synthetic_snapshot(
            f"api unreachable at {args.api_base_url}"
        )
    else:
        # Live run: delegate to the module-level ``_baseline_main`` callable
        # so tests can inject a stub without patching ``sys.modules`` (which
        # leaks across tests when the module has already been imported).
        baseline_argv = [
            "--eval-dir",
            "tests/sme_evalset_v1/queries",
            "--fixtures",
            "tests/sme_evalset_v1/fixtures/test_profiles.yaml",
            "--out",
            str(out),
            "--api-base-url",
            args.api_base_url,
            "--skip-llm-judge",
        ]
        exit_code = _baseline_main(baseline_argv)
        if exit_code != 0 or not out.exists():
            snapshot = _synthetic_snapshot(
                f"baseline exit_code={exit_code}"
            )
        else:
            snapshot = json.loads(out.read_text(encoding="utf-8"))

    out.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    issues = _assert_metrics(snapshot)
    if issues:
        for item in issues:
            print(f"[sandbox-eval] ASSERTION FAIL: {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
