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
from src.extraction.adapters.errors import NotNativePathError  # noqa: E402
from tests.extraction_bench.scoring import score_extraction  # noqa: E402

BENCH_CASES = Path(__file__).parent / "cases"

NATIVE_COVERAGE_MIN = 1.0
NATIVE_FIDELITY_MIN = 0.98
NATIVE_STRUCTURE_MIN = 1.0
NATIVE_HALLUCINATION_MAX = 0.0


def _stub_vision_client_calls():
    """Patch VisionClient.call for the bench so vision-path fixtures run offline."""
    from src.extraction.vision import client as _client_module
    from src.extraction.vision.client import VisionResponse

    def fake_call(self, *, system, user_text, image_bytes, image_mime="image/png",
                  max_tokens=1024, temperature=0.0):
        s = system.lower()
        if "routing decision" in s or "classifier" in s:
            return VisionResponse(
                text='{"format":"pdf_scanned","doc_type_hint":"unknown","layout_complexity":"simple",'
                     '"has_handwriting":false,"suggested_path":"vision","confidence":0.8}',
                prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="bench-stub",
            )
        if "coverage verifier" in s or "verifier stage" in s:
            return VisionResponse(
                text='{"complete":false,"missed_regions":[{"bbox":[0.0,0.0,1.0,1.0],"description":"full-page"}],'
                     '"low_confidence_regions":[]}',
                prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="bench-stub",
            )
        return VisionResponse(
            text='{"regions":[],"reading_order":[],"page_confidence":0.1}',
            prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="bench-stub",
        )

    _client_module.VisionClient.call = fake_call  # type: ignore


def _stub_fallback_responses(canned_text_for_scan: str, canned_text_for_image: str):
    """Patch run_fallback_ensemble to emit canned text based on image dimensions."""
    from src.extraction.vision.fallback import FallbackRegionResult
    import src.extraction.vision.orchestrator as _orch

    def fake(img, *, bbox):
        w, h = img.size
        if w == 400 and h == 200:
            return FallbackRegionResult(text=canned_text_for_image, agreement=1.0, engine_winner="both")
        return FallbackRegionResult(text=canned_text_for_scan, agreement=1.0, engine_winner="both")

    _orch.run_fallback_ensemble = fake


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
    try:
        result = dispatch_native(file_bytes, filename=source.name, doc_id=case_dir.name)
    except NotNativePathError:
        from src.extraction.vision.orchestrator import extract_via_vision
        hint = "pdf_scanned" if source.suffix.lower() == ".pdf" else "image"
        result = extract_via_vision(file_bytes, doc_id=case_dir.name, filename=source.name, format_hint=hint)
    actual = _extraction_to_comparable(result)
    scores = score_extraction(expected, actual)
    if result.path_taken == "native":
        gate_passed = (
            scores["coverage"] >= NATIVE_COVERAGE_MIN
            and scores["fidelity"] >= NATIVE_FIDELITY_MIN
            and scores["structure"] >= NATIVE_STRUCTURE_MIN
            and scores["hallucination"] <= NATIVE_HALLUCINATION_MAX
        )
    else:
        # Vision path gate per spec §8.4
        gate_passed = (
            scores["coverage"] >= 0.95
            and scores["fidelity"] >= 0.92
            and scores["structure"] >= 0.95
            and scores["hallucination"] <= 0.01
        )
    return {"case": case_dir.name, "scores": scores, "gate_passed": gate_passed, "path_taken": result.path_taken}


def main() -> int:
    if not BENCH_CASES.exists():
        print(f"ERROR: bench cases directory missing: {BENCH_CASES}", file=sys.stderr)
        return 2

    _stub_vision_client_calls()
    _stub_fallback_responses(
        canned_text_for_scan="scanned content via fallback",
        canned_text_for_image="image content via fallback",
    )

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
