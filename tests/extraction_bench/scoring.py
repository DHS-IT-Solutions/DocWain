"""Extraction accuracy scoring.

Per spec §8.3:
- Coverage (50%): every expected block/row/cell present in actual. Miss → 0 for that doc.
- Fidelity (30%): Levenshtein similarity per matched block.
- Structure (15%): tables match row × column; sheet / slide / page ordering preserved.
- Hallucination (5%): actual content not in expected is penalized.
"""
from __future__ import annotations

from typing import Any


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - (_levenshtein(a, b) / denom)


def _iter_expected_blocks(expected: dict):
    for p in expected.get("pages", []) or []:
        for b in p.get("blocks", []) or []:
            yield b.get("text", "")
    for s in expected.get("sheets", []) or []:
        for coord, cell in (s.get("cells") or {}).items():
            yield str(cell.get("value", ""))
    for sl in expected.get("slides", []) or []:
        for b in sl.get("elements", []) or []:
            yield b.get("text", "")


def _iter_expected_tables(expected: dict):
    for p in expected.get("pages", []) or []:
        for t in p.get("tables", []) or []:
            yield t
    for sl in expected.get("slides", []) or []:
        for t in sl.get("tables", []) or []:
            yield t


def compute_coverage(expected: dict, actual: dict) -> float:
    exp_blocks = list(_iter_expected_blocks(expected))
    act_blocks = set(_iter_expected_blocks(actual))
    exp_tables = list(_iter_expected_tables(expected))
    act_tables = list(_iter_expected_tables(actual))

    if not exp_blocks and not exp_tables:
        return 1.0

    for eb in exp_blocks:
        if eb.strip() and eb not in act_blocks:
            return 0.0
    if len(act_tables) < len(exp_tables):
        return 0.0
    return 1.0


def compute_fidelity(expected: dict, actual: dict) -> float:
    exp_blocks = [b for b in _iter_expected_blocks(expected) if b.strip()]
    act_blocks = [b for b in _iter_expected_blocks(actual) if b.strip()]
    if not exp_blocks:
        return 1.0
    scores = []
    act_pool = list(act_blocks)
    for eb in exp_blocks:
        if eb in act_pool:
            act_pool.remove(eb)
            scores.append(1.0)
            continue
        best = 0.0
        best_idx = -1
        for i, ab in enumerate(act_pool):
            s = _similarity(eb, ab)
            if s > best:
                best, best_idx = s, i
        if best_idx >= 0:
            act_pool.pop(best_idx)
        scores.append(best)
    return sum(scores) / len(scores)


def compute_structure(expected: dict, actual: dict) -> float:
    exp_tables = list(_iter_expected_tables(expected))
    act_tables = list(_iter_expected_tables(actual))
    if not exp_tables:
        return 1.0
    if len(exp_tables) != len(act_tables):
        return 0.0
    for e, a in zip(exp_tables, act_tables):
        if len(e.get("rows", [])) != len(a.get("rows", [])):
            return 0.0
        for er, ar in zip(e["rows"], a["rows"]):
            if len(er) != len(ar):
                return 0.0
    return 1.0


def compute_hallucination(expected: dict, actual: dict) -> float:
    exp_blocks = set(b for b in _iter_expected_blocks(expected) if b.strip())
    act_blocks = [b for b in _iter_expected_blocks(actual) if b.strip()]
    if not act_blocks:
        return 0.0
    extra = [b for b in act_blocks if b not in exp_blocks]
    return len(extra) / len(act_blocks)


def score_extraction(expected: dict, actual: dict) -> dict:
    c = compute_coverage(expected, actual)
    f = compute_fidelity(expected, actual)
    s = compute_structure(expected, actual)
    h = compute_hallucination(expected, actual)
    weighted = 0.50 * c + 0.30 * f + 0.15 * s + 0.05 * (1.0 - h)
    return {
        "coverage": c,
        "fidelity": f,
        "structure": s,
        "hallucination": h,
        "weighted": weighted,
    }
