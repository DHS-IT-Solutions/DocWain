"""/api/ask p95 latency assertion.

Asserts /api/ask p95 with INSIGHTS_PROACTIVE_INJECTION on equals p95
with it off, ±5%. Per spec Section 13.2.

NOTE: This test depends on SP-G (proactive injection) shipping. Until
then it skips with a clear reason. SP-G's final task re-enables it.
"""
from __future__ import annotations

import pytest


@pytest.mark.perf
def test_proactive_injection_does_not_regress_p95():
    pytest.skip(
        "Awaits SP-G — proactive injection helper not yet wired. "
        "Re-enable in SP-G final task."
    )
