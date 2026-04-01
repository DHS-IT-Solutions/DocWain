"""DocWain Secret Vault -- unified secret management with pluggable backends."""

from src.vault.vault import SecretVault, get_secret, get_vault

__all__ = ["SecretVault", "get_vault", "get_secret"]
