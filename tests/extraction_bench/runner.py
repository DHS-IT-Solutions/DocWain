"""Extraction bench runner.

Iterates over tests/extraction_bench/cases/<doc_id>/ entries, runs the native
adapter on source.<ext>, compares against expected.json, emits a per-case and
aggregate report. Exits non-zero if any gate fails.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.extraction.adapters.dispatcher import dispatch_native  # noqa: E402
from tests.extraction_bench.scoring import score_extraction  # noqa: E402

BENCH_CASES = Path(__file__).parent / "cases"

NATIVE_COVERAGE_MIN = 1.0
NATIVE_FIDELITY_MIN = 0.98
NATIVE_STRUCTURE_MIN = 1.0
NATIVE_HALLUCINATION_MAX = 0.0


def _extraction_to_comparable(result) -> dict:
    data = asdict(result)
    for sheet in data.get("sheets", []) or []:
        new_cells = {}
        for k, v in (sheet.get("cells") or {}).items():
            new_cells[str(k)] = v
        sheet["cells"] = new_cells
    return data


def run_case(case_dir: Path) -> dict:
    source = next(case_dir.glob("source.*"))
    expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    file_bytes = source.read_bytes()
    result = dispatch_native(file_bytes, filename=source.name, doc_id=case_dir.name)
    actual = _extraction_to_comparable(result)
    scores = score_extraction(expected, actual)
    gate_passed = (
        scores["coverage"] >= NATIVE_COVERAGE_MIN
        and scores["fidelity"] >= NATIVE_FIDELITY_MIN
        and scores["structure"] >= NATIVE_STRUCTURE_MIN
        and scores["hallucination"] <= NATIVE_HALLUCINATION_MAX
    )
    return {"case": case_dir.name, "scores": scores, "gate_passed": gate_passed}


def main() -> int:
    if not BENCH_CASES.exists():
        print(f"ERROR: bench cases directory missing: {BENCH_CASES}", file=sys.stderr)
        return 2
    reports = []
    any_failed = False
    for case_dir in sorted(p for p in BENCH_CASES.iterdir() if p.is_dir()):
        report = run_case(case_dir)
        reports.append(report)
        status = "PASS" if report["gate_passed"] else "FAIL"
        print(
            f"[{status}] {report['case']}: "
            f"cov={report['scores']['coverage']:.3f} "
            f"fid={report['scores']['fidelity']:.3f} "
            f"struct={report['scores']['structure']:.3f} "
            f"hal={report['scores']['hallucination']:.3f} "
            f"weighted={report['scores']['weighted']:.3f}"
        )
        if not report["gate_passed"]:
            any_failed = True

    out = BENCH_CASES.parent / "bench_report.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
