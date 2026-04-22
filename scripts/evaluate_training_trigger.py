"""Thin CLI delegate for the training-trigger evaluator.

Real logic lives in :mod:`scripts.sme_patterns.training_trigger`. Kept at
top-level so operators and systemd can invoke it by name.
"""
from __future__ import annotations

import sys

from scripts.sme_patterns.training_trigger import main

if __name__ == "__main__":
    sys.exit(main())
