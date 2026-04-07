"""Tiered file classification for express vs full pipeline."""

from __future__ import annotations

import os
from enum import Enum

from teams_app.config import EXPRESS_FILE_TYPES


class Pipeline(str, Enum):
    EXPRESS = "express"
    FULL = "full"


def classify_file(filename: str) -> Pipeline:
    """Classify a file as express or full pipeline based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in EXPRESS_FILE_TYPES:
        return Pipeline.EXPRESS
    return Pipeline.FULL


def should_escalate(extracted_text: str, min_chars: int = 50) -> bool:
    """Check if express extraction yielded too little content and should escalate to full."""
    return len(extracted_text.strip()) < min_chars
