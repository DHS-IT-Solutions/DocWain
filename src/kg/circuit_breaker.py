import threading
import time

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_STATE_CLOSED = "closed"
_STATE_OPEN = "open"
_STATE_HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for Neo4j connections.

    Tracks consecutive failures and trips open after *failure_threshold*
    consecutive errors.  While open, ``should_allow_request()`` returns
    ``False`` so callers can skip the downstream call entirely.  After
    *recovery_timeout_seconds* the breaker transitions to half-open,
    allowing a single probe request through.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 300,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout_seconds = recovery_timeout_seconds

        self._lock = threading.Lock()
        self._failure_count: int = 0
        self._state: str = _STATE_CLOSED
        self._opened_at: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def should_allow_request(self) -> bool:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state in (_STATE_CLOSED, _STATE_HALF_OPEN)

    def record_success(self) -> None:
        with self._lock:
            if self._state != _STATE_CLOSED:
                logger.info(
                    "Circuit breaker closed — Neo4j recovered (was %s)",
                    self._state,
                )
            self._failure_count = 0
            self._state = _STATE_CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold and self._state != _STATE_OPEN:
                self._state = _STATE_OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "Circuit breaker OPEN — Neo4j hit %d consecutive failures, "
                    "skipping KG for %ds",
                    self._failure_count,
                    self._recovery_timeout_seconds,
                )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        """Must be called while holding ``_lock``."""
        if (
            self._state == _STATE_OPEN
            and (time.monotonic() - self._opened_at) >= self._recovery_timeout_seconds
        ):
            self._state = _STATE_HALF_OPEN
            logger.info(
                "Circuit breaker half-open — allowing probe request to Neo4j"
            )


# Module-level singleton used across the application.
neo4j_breaker = CircuitBreaker()
