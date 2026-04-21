"""Grep-gate: serving-layer fast/smart split must not appear in live code.

This gate is NARROW: it only catches tokens that the Batch-0 plan tasks
3–7 will remove (docwain-fast aliases, FastPathHandler class, serving/
fast_path module, _is_fast_path / _handle_fast_path query helpers,
route_taken="smart"/"fast" labels, and split-specific wording in
docstrings). Legitimate uses of "14B" in training configs, "fast_path"
in agentic/execution/nlp keyword-shortcut helpers, and intent-metadata
"fast_path" keys are deliberately NOT caught.

Allowed locations for historical references (bypassed entirely):
 - docs/superpowers/specs/*     (design history)
 - docs/superpowers/plans/*     (implementation history)
 - eval_results/*               (snapshots)
 - scripts/batch0/*             (one-shot audit tooling)
 - tests/batch0/*               (this test itself)
 - Anything git-ignored (pycache, logs, .superpowers/, etc.)

Any hit in src/, deploy/, systemd/, or general tests/ is a fail.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

# Narrow: target only references that Tasks 3–7 of the Batch-0 plan will
# remove. Do NOT include bare 14B / 27B / fast_path / "smart" / FastPathHandler
# as standalone tokens — those catch legitimate training-layer, agentic
# keyword-shortcut, and intent-metadata usages that are unrelated to the
# fast/smart SERVING split we are collapsing.
FORBIDDEN_PATTERNS = [
    # Model-name aliases (removed in Task 5)
    r"docwain[-_]fast",
    r"docwain[-_]smart",
    # Serving-layer FastPathHandler class (removed in Task 4)
    r"class\s+FastPathHandler\b",
    r"FastPathHandler\b",
    r"src\.serving\.fast_path\b",
    r"serving/fast_path\.py\b",
    # Query-pipeline split helpers (removed in Task 6)
    r"_handle_fast_path\b",
    r"_is_fast_path\b",
    # Route-taken split labels (removed in Task 6)
    r"route_taken\s*=\s*[\"']smart[\"']",
    r"route_taken\s*=\s*[\"']fast[\"']",
    # Split-specific wording in docstrings/comments (scrubbed in Task 7)
    r"14B\s*[\"'](?:fast|smart)[\"']",
    r"27B\s*[\"'](?:fast|smart)[\"']",
    r"(?:\"fast\"|'fast')\s*(?:and|or|/|,)\s*(?:27B|\"smart\"|'smart')",
    r"docwain\s+(?:fast|smart)\s+(?:model|instance|path)",
]

ALLOWED_PREFIXES = (
    "docs/superpowers/specs/",
    "docs/superpowers/plans/",
    "eval_results/",
    "scripts/batch0/",
    "tests/batch0/",
)


def _tracked_files(repo_root: pathlib.Path) -> list[str]:
    out = subprocess.check_output(
        ["git", "ls-files"], cwd=repo_root, text=True,
    )
    return [p for p in out.splitlines() if p.strip()]


def test_grep_gate_no_fast_smart_refs(repo_root: pathlib.Path):
    combined = re.compile("|".join(FORBIDDEN_PATTERNS))
    offenders: list[str] = []
    for rel in _tracked_files(repo_root):
        if any(rel.startswith(p) for p in ALLOWED_PREFIXES):
            continue
        if not (rel.startswith("src/") or rel.startswith("deploy/")
                or rel.startswith("systemd/") or rel.startswith("tests/")):
            continue
        path = repo_root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if combined.search(line):
                offenders.append(f"{rel}:{line_no}: {line.strip()[:140]}")
    assert not offenders, (
        "Forbidden fast/smart tokens found in live code:\n  "
        + "\n  ".join(offenders[:50])
        + (f"\n  ...and {len(offenders) - 50} more" if len(offenders) > 50 else "")
    )
