"""Upload → screening-eligible time perf test.

Asserts time-from-upload-complete to HITL-screening-eligible is unchanged
after researcher v2 lands. Per spec Section 13.3.

NOTE: This test depends on the researcher_v2_queue being installed and
a synthetic upload harness existing. SP-C and SP-K.5 wire it up.
"""
from __future__ import annotations

import pytest


@pytest.mark.perf
def test_upload_to_screening_eligible_unchanged():
    pytest.skip(
        "Awaits SP-C researcher v2 + a synthetic upload harness. "
        "Re-enable in SP-K.5."
    )
