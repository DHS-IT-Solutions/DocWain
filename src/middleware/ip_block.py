"""
IP auto-block middleware.

Tracks 404 responses per client IP. When an IP exceeds the configured threshold
within a sliding window, it is permanently blocked. Blocked IPs receive 403
immediately with zero request processing.

Blocked IPs persist to a JSON file and survive restarts.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class IPBlockManager:
    """Tracks 404 rates and manages the permanent block list."""

    def __init__(
        self,
        threshold: int = 5,
        window_seconds: int = 60,
        whitelist: Optional[Set[str]] = None,
        persist_path: Optional[str] = None,
    ) -> None:
        self._threshold = threshold
        self._window = window_seconds
        self._whitelist: Set[str] = whitelist or {"127.0.0.1", "::1"}
        self._persist_path = persist_path

        self._blocked: Dict[str, str] = {}  # ip -> iso timestamp
        self._tracker: Dict[str, List[float]] = {}  # ip -> [monotonic timestamps]
        self._lock = threading.Lock()

        self._load()

    # ── Public API ──────────────────────────────────────────────

    def is_blocked(self, ip: str) -> bool:
        return ip in self._blocked

    def record_404(self, ip: str) -> bool:
        """Record a 404 for an IP. Returns True if the IP was just blocked."""
        if ip in self._whitelist or ip in self._blocked:
            return False

        now = time.monotonic()
        with self._lock:
            hits = self._tracker.get(ip)
            if hits is None:
                hits = []
                self._tracker[ip] = hits

            hits.append(now)

            # Trim to window
            cutoff = now - self._window
            self._tracker[ip] = [t for t in hits if t > cutoff]

            if len(self._tracker[ip]) >= self._threshold:
                self._blocked[ip] = datetime.now(timezone.utc).isoformat()
                del self._tracker[ip]
                self._save()
                logger.warning(
                    "[IP_BLOCK] Blocked %s — %d 404s in %ds",
                    ip, self._threshold, self._window,
                )
                return True

        return False

    def block(self, ip: str, reason: str = "manual") -> None:
        """Manually block an IP."""
        with self._lock:
            self._blocked[ip] = datetime.now(timezone.utc).isoformat()
            self._save()
        logger.info("[IP_BLOCK] Manually blocked %s (%s)", ip, reason)

    def unblock(self, ip: str) -> bool:
        """Remove an IP from the block list. Returns True if it was blocked."""
        with self._lock:
            if ip in self._blocked:
                del self._blocked[ip]
                self._save()
                logger.info("[IP_BLOCK] Unblocked %s", ip)
                return True
        return False

    def list_blocked(self) -> Dict[str, str]:
        """Return all blocked IPs with their block timestamps."""
        return dict(self._blocked)

    # ── Persistence ─────────────────────────────────────────────

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            p = Path(self._persist_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self._blocked, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("[IP_BLOCK] Failed to save block list: %s", exc)

    def _load(self) -> None:
        if not self._persist_path:
            return
        try:
            p = Path(self._persist_path)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._blocked = data
                    logger.info("[IP_BLOCK] Loaded %d blocked IPs from %s", len(data), self._persist_path)
        except Exception as exc:
            logger.debug("[IP_BLOCK] Failed to load block list: %s", exc)


# ── Module-level singleton ──────────────────────────────────────

_manager: Optional[IPBlockManager] = None


def get_block_manager() -> IPBlockManager:
    """Get or create the singleton IPBlockManager."""
    global _manager
    if _manager is None:
        from src.api.config import Config
        sec = getattr(Config, "Security", None)
        threshold = getattr(sec, "BLOCK_THRESHOLD", 5) if sec else 5
        window = getattr(sec, "BLOCK_WINDOW_SECONDS", 60) if sec else 60
        whitelist_str = getattr(sec, "WHITELIST_IPS", ["127.0.0.1", "::1"]) if sec else ["127.0.0.1", "::1"]
        persist = getattr(sec, "BLOCKED_IPS_FILE", "data/blocked_ips.json") if sec else "data/blocked_ips.json"

        if isinstance(whitelist_str, str):
            whitelist_str = [s.strip() for s in whitelist_str.split(",") if s.strip()]

        _manager = IPBlockManager(
            threshold=threshold,
            window_seconds=window,
            whitelist=set(whitelist_str),
            persist_path=persist,
        )
    return _manager


# ── ASGI Middleware ──────────────────────────────────────────────

_BLOCKED_BODY = json.dumps({"error": {"code": "IP_BLOCKED", "message": "Access denied"}}).encode("utf-8")


# Paths that no legitimate client would ever request.
# A single hit on any of these triggers an instant permanent block.
_HONEYPOT_PATTERNS = (
    "/.env", "/.git/", "/wp-config", "/wp-admin", "/wp-login",
    "/.aws/", "/credentials", "/master.key", "/secrets.",
    "/terraform.", "/actuator/", "/debug/vars",
    "/config.json", "/config.yaml", "/config.yml", "/config.toml",
    "/application.properties", "/application.yml", "/application.yaml",
    "/appsettings.", "/serverless.", "/docker-compose.",
    "/.streamlit/", "/.flaskenv", "/.secrets",
    "/backup/.env", "/backups/.env", "/old/.env", "/tmp/.env",
    "/judge",
)


class IPBlockMiddleware:
    """Pure ASGI middleware — blocks IPs with high 404 rates.

    Also instantly blocks IPs that probe known scanner honeypot paths.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._manager = get_block_manager()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        ip = self._get_client_ip(scope)

        # Fast reject for blocked IPs
        if self._manager.is_blocked(ip):
            await self._send_403(send)
            return

        # Instant block for honeypot paths — no legitimate client requests these
        path = scope.get("path", "")
        if not self._manager._whitelist or ip not in self._manager._whitelist:
            if any(pattern in path for pattern in _HONEYPOT_PATTERNS):
                self._manager.block(ip, reason=f"honeypot: {path}")
                await self._send_403(send)
                return

        # Intercept response to detect 404s
        response_status = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message.get("status", 0)
            await send(message)

        await self.app(scope, receive, send_wrapper)

        # After response sent, check if it was a 404
        if response_status == 404:
            self._manager.record_404(ip)

    def _get_client_ip(self, scope: Scope) -> str:
        """Extract client IP from X-Forwarded-For or ASGI scope."""
        headers = dict(scope.get("headers", []))
        xff = headers.get(b"x-forwarded-for")
        if xff:
            # First IP in X-Forwarded-For is the original client
            return xff.decode("utf-8", errors="replace").split(",")[0].strip()
        client = scope.get("client")
        if client:
            return client[0]
        return "unknown"

    async def _send_403(self, send: Send) -> None:
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(_BLOCKED_BODY)).encode()],
            ],
        })
        await send({
            "type": "http.response.body",
            "body": _BLOCKED_BODY,
        })
