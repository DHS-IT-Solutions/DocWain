"""Ensure no hardcoded qwen3:14b model reference remains in production source."""
import subprocess
from pathlib import Path


def test_no_hardcoded_qwen3_14b_in_src():
    """qwen3:14b is not present as a hardcoded model name in production source.

    This regression test guards against re-introducing the stale reference that
    caused 45+ 'model not found' errors in production logs on 2026-04-24.
    """
    repo_root = Path(__file__).resolve().parents[3]
    src = repo_root / "src"
    assert src.exists(), f"src directory not found at {src}"
    # Use `grep -rn --include=*.py`; grep returns 1 if no matches (that's success for us).
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", "qwen3:14b", str(src)],
        capture_output=True, text=True,
    )
    # rc==1 means no matches (good); rc==0 means matches exist (bad); rc>=2 means grep error
    assert result.returncode == 1, (
        f"Stale qwen3:14b references found in src/:\n{result.stdout}"
    )
