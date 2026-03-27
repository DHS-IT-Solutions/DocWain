"""CLI tool: migrate secrets from a ``.env`` file into the local encrypted vault.

Usage::

    python -m src.vault.migrate --from-env .env
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Keys (or substrings) that indicate a value is sensitive.
_SENSITIVE_MARKERS = {"KEY", "PASSWORD", "SECRET", "TOKEN", "CONNECTION_STRING"}
# URI env vars that likely embed credentials.
_URI_PATTERN = re.compile(r"://[^:]+:[^@]+@", re.IGNORECASE)


def _is_sensitive(key: str, value: str) -> bool:
    upper = key.upper()
    for marker in _SENSITIVE_MARKERS:
        if marker in upper:
            return True
    if "URI" in upper and _URI_PATTERN.search(value):
        return True
    return False


def _parse_env_file(path: Path) -> dict[str, str]:
    """Return key=value pairs from a ``.env`` file, skipping comments/blanks."""
    pairs: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and value:
            pairs[key] = value
    return pairs


def migrate(env_path: Path) -> None:
    """Read *env_path*, filter sensitive keys, and write to the local vault."""
    from src.vault._local import LocalEncryptedBackend

    all_vars = _parse_env_file(env_path)
    sensitive = {k: v for k, v in all_vars.items() if _is_sensitive(k, v)}

    if not sensitive:
        print("No sensitive keys detected in the .env file.")
        return

    backend = LocalEncryptedBackend()

    print(f"Migrating {len(sensitive)} secret(s) from {env_path} ...")
    for key, value in sensitive.items():
        backend.set(key, value, metadata={"source": str(env_path)})
        print(f"  + {key}")

    # Verification pass
    errors = []
    for key, original in sensitive.items():
        stored = backend.get(key)
        if stored != original:
            errors.append(key)
    if errors:
        print(f"\nVerification FAILED for: {', '.join(errors)}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll {len(sensitive)} secrets migrated and verified successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate .env secrets to DocWain encrypted vault.")
    parser.add_argument("--from-env", required=True, type=Path, help="Path to the .env file")
    args = parser.parse_args()

    if not args.from_env.exists():
        print(f"File not found: {args.from_env}", file=sys.stderr)
        sys.exit(1)

    migrate(args.from_env)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
