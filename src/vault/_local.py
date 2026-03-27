"""Local encrypted-file backend using Fernet symmetric encryption."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from src.vault._base import SecretBackend

log = logging.getLogger(__name__)

_SALT = b"docwain-vault-v1"
_PBKDF2_ITERATIONS = 480_000


def _derive_fernet_key(master_key: str) -> bytes:
    """Derive a Fernet-compatible key from *master_key* via PBKDF2-HMAC-SHA256."""
    import base64

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_SALT,
        iterations=_PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(master_key.encode()))


class LocalEncryptedBackend(SecretBackend):
    """Encrypt secrets into a local JSON file protected by Fernet.

    The master key is read from the ``VAULT_MASTER_KEY`` environment variable.
    Secrets are stored at ``{project_root}/.vault/secrets.enc``.
    """

    def __init__(self, master_key: str | None = None, vault_path: Path | None = None) -> None:
        from cryptography.fernet import Fernet

        mk = master_key or os.getenv("VAULT_MASTER_KEY", "")
        if not mk:
            raise ValueError("VAULT_MASTER_KEY is required for LocalEncryptedBackend")

        self._fernet = Fernet(_derive_fernet_key(mk))
        self._lock = threading.Lock()

        if vault_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            vault_path = project_root / ".vault" / "secrets.enc"
        self._path = vault_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # -- internal helpers ------------------------------------------------

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            cipher_text = self._path.read_bytes()
            plain = self._fernet.decrypt(cipher_text)
            return json.loads(plain)
        except Exception:
            log.exception("Failed to decrypt vault file at %s", self._path)
            return {}

    def _save(self, data: dict) -> None:
        plain = json.dumps(data, sort_keys=True).encode()
        cipher_text = self._fernet.encrypt(plain)
        self._path.write_bytes(cipher_text)

    # -- SecretBackend interface -----------------------------------------

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            data = self._load()
        entry = data.get(key)
        if entry is None:
            return None
        # entries are stored as {"value": "...", "metadata": {...}}
        if isinstance(entry, dict):
            return entry.get("value")
        return str(entry)

    def set(self, key: str, value: str, metadata: dict | None = None) -> None:
        with self._lock:
            data = self._load()
            data[key] = {"value": value, "metadata": metadata or {}}
            self._save(data)

    def delete(self, key: str) -> None:
        with self._lock:
            data = self._load()
            data.pop(key, None)
            self._save(data)

    def list_keys(self) -> list[str]:
        with self._lock:
            data = self._load()
        return list(data.keys())

    def rotate(self, key: str, new_value: str) -> None:
        with self._lock:
            data = self._load()
            existing = data.get(key, {})
            meta = existing.get("metadata", {}) if isinstance(existing, dict) else {}
            data[key] = {"value": new_value, "metadata": meta}
            self._save(data)
        log.info("Rotated secret '%s' in local encrypted vault.", key)

    def health_check(self) -> bool:
        try:
            with self._lock:
                self._load()
            return True
        except Exception:
            return False
