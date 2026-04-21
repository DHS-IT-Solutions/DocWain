"""SME feature flag resolution (spec §13, ERRATA §4 + §21).

All SME Phase 2+ behavior is guarded by one of eight boolean flags. Master
gate ``sme_redesign_enabled`` controls the dependent set (synthesis,
retrieval, KG synthesis, rich mode, URL-as-prompt). Two *independent* flags
(``enable_hybrid_retrieval``, ``enable_cross_encoder_rerank``) survive
master rollback per spec §13.5 — the retrieval infrastructure upgrades
should not regress even if a master disable is applied.

Backing store is MongoDB (control-plane only, per the storage separation
rule); the :class:`FlagStore` protocol lets tests inject a simple mock.

All flags default OFF in Phase 1 (spec §13.4). Per-subscription overrides
take precedence over the global default.
"""
from __future__ import annotations

from typing import Final, Protocol

# ---------------------------------------------------------------------------
# Exported flag-name constants (ERRATA §4 canonical)
# ---------------------------------------------------------------------------
SME_REDESIGN_ENABLED: Final[str] = "sme_redesign_enabled"
ENABLE_SME_SYNTHESIS: Final[str] = "enable_sme_synthesis"
ENABLE_SME_RETRIEVAL: Final[str] = "enable_sme_retrieval"
ENABLE_KG_SYNTHESIZED_EDGES: Final[str] = "enable_kg_synthesized_edges"
ENABLE_RICH_MODE: Final[str] = "enable_rich_mode"
ENABLE_URL_AS_PROMPT: Final[str] = "enable_url_as_prompt"
ENABLE_HYBRID_RETRIEVAL: Final[str] = "enable_hybrid_retrieval"
ENABLE_CROSS_ENCODER_RERANK: Final[str] = "enable_cross_encoder_rerank"


_MASTER: Final[str] = SME_REDESIGN_ENABLED
_DEPENDENT: Final[frozenset[str]] = frozenset(
    {
        ENABLE_SME_SYNTHESIS,
        ENABLE_SME_RETRIEVAL,
        ENABLE_KG_SYNTHESIZED_EDGES,
        ENABLE_RICH_MODE,
        ENABLE_URL_AS_PROMPT,
    }
)
_INDEPENDENT: Final[frozenset[str]] = frozenset(
    {ENABLE_HYBRID_RETRIEVAL, ENABLE_CROSS_ENCODER_RERANK}
)
_ALL: Final[frozenset[str]] = frozenset({_MASTER, *_DEPENDENT, *_INDEPENDENT})
_DEFAULTS: Final[dict[str, bool]] = {name: False for name in _ALL}


class FlagStore(Protocol):
    """Read-only flag override surface backed by MongoDB (control plane).

    Implementations return a ``dict[str, bool]`` whose keys are a subset of
    the canonical flag names. Missing keys fall back to the global default.
    """

    def get_subscription_overrides(
        self, subscription_id: str
    ) -> dict[str, bool]: ...


class MutableFlagStore(FlagStore, Protocol):
    """Extended ``FlagStore`` that supports per-subscription override writes.

    Phase 3 admin endpoints flip ``enable_sme_retrieval`` per opt-in
    subscription; the store must persist the write to MongoDB
    (control plane) so the next :meth:`SMEFeatureFlags.is_enabled` call
    sees the new override. Missing keys are still allowed — callers SHOULD
    pass through only the canonical 8 flag names (validation lives in the
    admin endpoint, not in the store).
    """

    def set_subscription_override(
        self, subscription_id: str, flag: str, value: bool
    ) -> None: ...


class SMEFeatureFlags:
    """Flag resolver with master-gate precedence.

    Instantiation is cheap; the resolver holds the store reference and reads
    overrides on every call. Callers that want per-request caching should
    wrap this in their own cache layer — this module stays stateless so
    admin mutations don't get silently masked.
    """

    def __init__(self, *, store: FlagStore) -> None:
        self._store = store

    def is_enabled(self, subscription_id: str, flag: str) -> bool:
        """Return True iff the flag is on for ``subscription_id``.

        Resolution order (spec §13):

        1. If ``flag`` is unknown, raise :class:`KeyError` — prevents typos
           from silently passing the default-off semantics.
        2. If ``flag`` is dependent and the master is off, return False
           regardless of the override.
        3. Apply per-subscription override; fall back to the global default.
        """
        if flag not in _ALL:
            raise KeyError(f"unknown feature flag {flag!r}")
        overrides = self._store.get_subscription_overrides(subscription_id)
        master_on = overrides.get(_MASTER, _DEFAULTS[_MASTER])
        if flag in _DEPENDENT and not master_on:
            return False
        return bool(overrides.get(flag, _DEFAULTS[flag]))


# ---------------------------------------------------------------------------
# Module-level singleton (ERRATA §4)
# ---------------------------------------------------------------------------
_flag_resolver_singleton: SMEFeatureFlags | None = None


def get_flag_resolver() -> SMEFeatureFlags:
    """Return the process-wide :class:`SMEFeatureFlags` instance.

    Non-FastAPI callers (synthesis pipeline, background workers, CLI tools)
    use this to avoid threading the resolver through every call site.
    FastAPI lifespan wires the same instance into ``app.state`` so admin,
    retrieval, and synthesis paths all see identical overrides.
    """
    global _flag_resolver_singleton
    if _flag_resolver_singleton is None:
        raise RuntimeError(
            "SMEFeatureFlags not initialized — call init_flag_resolver() at startup"
        )
    return _flag_resolver_singleton


def init_flag_resolver(*, store: FlagStore) -> SMEFeatureFlags:
    """Initialize the module-level singleton. Idempotent-replace: each call
    installs a fresh resolver bound to the provided store (tests exploit this
    to reset between cases)."""
    global _flag_resolver_singleton
    _flag_resolver_singleton = SMEFeatureFlags(store=store)
    return _flag_resolver_singleton


def get_flag_store() -> FlagStore:
    """Return the underlying :class:`FlagStore` the resolver is reading.

    Admin write helpers use this to get a handle on the process-wide store
    without threading it through every call site.
    """
    return get_flag_resolver()._store


def all_flag_names() -> frozenset[str]:
    """Return the full set of canonical flag names recognised by the resolver.

    Admin endpoints validate input flag names against this set so a typo
    never silently persists a non-resolvable override.
    """
    return _ALL


def set_subscription_override(
    subscription_id: str, flag: str, value: bool
) -> None:
    """Persist a per-subscription flag override via the shared store.

    Validates ``flag`` against the canonical 8-flag set and requires the
    backing store to implement :class:`MutableFlagStore`. Also bumps the
    monotonic flag-set version counter so any downstream cache keyed on
    ``flag_set_version`` (Phase 3 retrieval cache — Task 8) invalidates.

    Raises:
        KeyError: if ``flag`` is not one of the 8 canonical flag names.
        TypeError: if the backing store does not support override writes.
    """
    if flag not in _ALL:
        raise KeyError(f"unknown feature flag {flag!r}")
    store = get_flag_store()
    setter = getattr(store, "set_subscription_override", None)
    if not callable(setter):
        raise TypeError(
            "backing FlagStore does not implement MutableFlagStore — "
            "per-subscription override writes are not supported"
        )
    setter(subscription_id, flag, bool(value))
    bump_flag_set_version()


# ---------------------------------------------------------------------------
# Monotonic flag-set version counter (Phase 3 retrieval-cache invalidation)
# ---------------------------------------------------------------------------
#
# The Redis retrieval cache (Task 8) keys entries on
# ``(sub, prof, query_fp, flag_set_version)``. Any flag flip bumps this
# counter so cached packs rendered under the previous flag configuration
# are implicitly invalidated — no explicit scan-and-delete needed on the
# hot path.
_flag_set_version: int = 0


def bump_flag_set_version() -> int:
    """Atomically increment the monotonic flag-set version and return the
    new value. Admin write paths call this immediately after persisting a
    flag mutation so downstream caches that key on the version invalidate
    naturally."""
    global _flag_set_version
    _flag_set_version += 1
    return _flag_set_version


def get_flag_set_version() -> int:
    """Return the current monotonic flag-set version. Cache writers use
    this to tag entries with the active flag configuration."""
    return _flag_set_version


def reset_flag_set_version() -> None:
    """Reset the counter to zero. Tests exploit this for isolation."""
    global _flag_set_version
    _flag_set_version = 0
