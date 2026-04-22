"""Thin top-level entry-point for monthly SME pattern mining.

All real logic lives in :mod:`scripts.sme_patterns.run`. Kept at top-level so
operators and systemd can invoke the pipeline by name.

Usage::

    python -m scripts.mine_sme_patterns --window-days 30 --out-dir analytics/
"""
from __future__ import annotations

import sys

from scripts.sme_patterns.run import main

if __name__ == "__main__":
    sys.exit(main())
