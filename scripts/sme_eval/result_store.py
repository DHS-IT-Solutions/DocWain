"""JSONL-backed result store for eval runs.

One line per EvalResult. Append-only. Small enough to ship in-repo for
the 600-query baseline (~6 MB); large enough volumes later could move to
Blob following the storage-separation rule.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from scripts.sme_eval.schema import EvalResult


class JsonlResultStore:
    """Append-only JSONL store for EvalResult records."""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, result: EvalResult) -> None:
        line = result.model_dump_json()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def iter_all(self) -> Iterator[EvalResult]:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield EvalResult.model_validate_json(line)

    def iter_run(self, run_id: str) -> Iterator[EvalResult]:
        for r in self.iter_all():
            if r.run_id == run_id:
                yield r
