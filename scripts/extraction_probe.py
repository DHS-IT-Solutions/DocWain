#!/usr/bin/env python3
"""Validator for the deterministic extraction layer.

Runs ``src.extraction.deterministic.extract`` over every file in a
directory and reports per-file validation results. Designed to be
re-run any time the deterministic extractor changes — it anchors the
quality bar.

Quality gates (per file):
  G1. text_full is non-empty (content was captured at all)
  G2. no unrecovered warnings that indicate fatal errors
  G3. format-specific structural checks (see below)

Format-specific gates:
  - PDF   : page_count > 0; at least one block of type paragraph
  - DOCX  : paragraph_count + table count > 0 (not a blank doc)
  - XLSX  : sheet_count > 0; every sheet has >= 1 non-empty cell
  - IMAGE : ocr_word_count > 10 (sanity — OCR found real text)
  - CSV   : row_count > 0
  - TXT   : text_char_count > 0

Cross-file advisory (not gated):
  - PDF multi-page tables: reports when cross_page stitching fired
  - DOCX watermarks: reports when detected
  - Image OCR mean confidence: reports, flags < 30

Exit code: 0 if every file passes all gates, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure src/ is importable when run directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extraction import deterministic  # noqa: E402


# ---------------------------------------------------------------------------
# Per-file validation — delegated to src.extraction.deterministic.validate
# ---------------------------------------------------------------------------


def _validate(raw: deterministic.RawExtraction) -> Tuple[bool, List[str], List[str]]:
    result = deterministic.validate(raw)
    return result["passed"], result["failed_checks"], result["advisories"]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _format_summary_line(fname: str, raw: deterministic.RawExtraction) -> str:
    fmt = raw.file_format
    meta_bits: List[str] = []

    if fmt == "pdf":
        meta_bits.append(f"pages={raw.metadata.get('page_count', 0)}")
        meta_bits.append(f"images_embedded={raw.metadata.get('embedded_image_count', 0)}")
    elif fmt == "docx":
        meta_bits.append(f"paragraphs={raw.metadata.get('paragraph_count', 0)}")
        meta_bits.append(f"images_inline={raw.metadata.get('inline_image_count', 0)}")
    elif fmt in ("xlsx", "xls"):
        meta_bits.append(f"sheets={raw.metadata.get('sheet_count', 0)}")
    elif fmt in ("png", "jpg", "jpeg", "tiff", "bmp"):
        meta_bits.append(f"dims={raw.metadata.get('pixel_dimensions', '?')}")
        meta_bits.append(f"ocr_words={raw.metadata.get('ocr_word_count', 0)}")

    meta_bits.append(f"text={len(raw.text_full)}ch")
    meta_bits.append(f"blocks={len(raw.blocks)}")
    meta_bits.append(f"tables={len(raw.tables)}")
    return f"{fname} [{fmt}]  " + "  ".join(meta_bits)


def run_validator(directory: Path, out_path: Path = None, verbose: bool = False) -> int:
    files = sorted(p for p in directory.iterdir() if p.is_file())
    if not files:
        print(f"No files in {directory}", file=sys.stderr)
        return 2

    results: Dict[str, Any] = {}
    overall_pass = True

    for p in files:
        try:
            content = p.read_bytes()
        except Exception as exc:  # noqa: BLE001
            results[p.name] = {"passed": False, "failed_gates": [f"read error: {exc}"]}
            overall_pass = False
            continue

        raw = deterministic.extract(content, p.name)
        passed, failed, advisories = _validate(raw)
        if not passed:
            overall_pass = False

        line = _format_summary_line(p.name, raw)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {line}")
        for f in failed:
            print(f"    - {f}")
        for a in advisories:
            print(f"    · {a}")
        if raw.warnings and verbose:
            for w in raw.warnings:
                print(f"    ! warning: {w}")

        results[p.name] = {
            "passed": passed,
            "failed_gates": failed,
            "advisories": advisories,
            "summary": line,
            "extraction": raw.to_dict() if verbose else {
                # keep file small in default mode
                "file_format": raw.file_format,
                "text_char_count": len(raw.text_full),
                "block_count": len(raw.blocks),
                "table_count": len(raw.tables),
                "metadata": raw.metadata,
                "warnings": raw.warnings,
            },
        }

    print()
    n_pass = sum(1 for r in results.values() if r["passed"])
    print(f"== Result: {n_pass}/{len(results)} files passed ==")

    if out_path:
        out_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str)
        )
        print(f"wrote {out_path}")

    return 0 if overall_pass else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path, help="Directory of documents to validate")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output file")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Include full extraction dump in --out JSON and show warnings")
    args = ap.parse_args()
    sys.exit(run_validator(args.directory, out_path=args.out, verbose=args.verbose))


if __name__ == "__main__":
    main()
