"""Tests for IP auto-block middleware."""
import json
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def manager():
    from src.middleware.ip_block import IPBlockManager
    return IPBlockManager(threshold=3, window_seconds=60, whitelist={"127.0.0.1"})


@pytest.fixture
def persistent_manager(tmp_path):
    from src.middleware.ip_block import IPBlockManager
    path = str(tmp_path / "blocked.json")
    return IPBlockManager(threshold=3, window_seconds=60, persist_path=path), path


def test_not_blocked_initially(manager):
    assert manager.is_blocked("1.2.3.4") is False


def test_record_404_below_threshold(manager):
    assert manager.record_404("1.2.3.4") is False
    assert manager.record_404("1.2.3.4") is False
    assert manager.is_blocked("1.2.3.4") is False


def test_record_404_triggers_block_at_threshold(manager):
    manager.record_404("1.2.3.4")
    manager.record_404("1.2.3.4")
    result = manager.record_404("1.2.3.4")  # 3rd = threshold
    assert result is True
    assert manager.is_blocked("1.2.3.4") is True


def test_whitelisted_ip_never_blocked(manager):
    for _ in range(10):
        manager.record_404("127.0.0.1")
    assert manager.is_blocked("127.0.0.1") is False


def test_different_ips_tracked_independently(manager):
    manager.record_404("1.1.1.1")
    manager.record_404("1.1.1.1")
    manager.record_404("2.2.2.2")
    manager.record_404("2.2.2.2")
    assert manager.is_blocked("1.1.1.1") is False
    assert manager.is_blocked("2.2.2.2") is False


def test_manual_block(manager):
    manager.block("5.5.5.5", reason="test")
    assert manager.is_blocked("5.5.5.5") is True


def test_unblock(manager):
    manager.block("5.5.5.5")
    assert manager.unblock("5.5.5.5") is True
    assert manager.is_blocked("5.5.5.5") is False


def test_unblock_returns_false_if_not_blocked(manager):
    assert manager.unblock("9.9.9.9") is False


def test_list_blocked(manager):
    manager.block("1.1.1.1")
    manager.block("2.2.2.2")
    blocked = manager.list_blocked()
    assert "1.1.1.1" in blocked
    assert "2.2.2.2" in blocked
    assert len(blocked) == 2


def test_persistence_save_and_load(persistent_manager):
    mgr, path = persistent_manager
    mgr.block("10.0.0.1")
    mgr.block("10.0.0.2")

    # Verify file exists
    data = json.loads(Path(path).read_text())
    assert "10.0.0.1" in data
    assert "10.0.0.2" in data

    # Create new manager from same file
    from src.middleware.ip_block import IPBlockManager
    mgr2 = IPBlockManager(threshold=3, window_seconds=60, persist_path=path)
    assert mgr2.is_blocked("10.0.0.1") is True
    assert mgr2.is_blocked("10.0.0.2") is True


def test_window_expiry(manager):
    """Timestamps outside the window should not count."""
    import time
    # Use a manager with very short window
    from src.middleware.ip_block import IPBlockManager
    mgr = IPBlockManager(threshold=3, window_seconds=0.1)
    mgr.record_404("1.2.3.4")
    mgr.record_404("1.2.3.4")
    time.sleep(0.15)  # Wait for window to expire
    mgr.record_404("1.2.3.4")  # Only 1 in current window
    assert mgr.is_blocked("1.2.3.4") is False


def test_already_blocked_ip_does_not_retrack(manager):
    """Once blocked, record_404 returns False (no double-tracking)."""
    manager.record_404("1.2.3.4")
    manager.record_404("1.2.3.4")
    manager.record_404("1.2.3.4")  # blocked
    assert manager.is_blocked("1.2.3.4") is True
    result = manager.record_404("1.2.3.4")
    assert result is False


# ── ASGI middleware tests ───────────────────────────────────────

@pytest.fixture
def block_middleware():
    """Create middleware wrapping a dummy app that returns 404."""
    from src.middleware.ip_block import IPBlockMiddleware, IPBlockManager

    mgr = IPBlockManager(threshold=2, window_seconds=60, whitelist=set())

    async def dummy_404_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b"Not Found"})

    mw = IPBlockMiddleware(dummy_404_app)
    mw._manager = mgr
    return mw, mgr


@pytest.mark.asyncio
async def test_middleware_blocks_after_threshold(block_middleware):
    mw, mgr = block_middleware

    scope = {"type": "http", "headers": [], "client": ("9.9.9.9", 12345)}
    responses = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(msg):
        responses.append(msg)

    # First request — 404, not blocked yet
    await mw(scope, receive, send)
    assert responses[0]["status"] == 404

    responses.clear()
    # Second request — 404, triggers block
    await mw(scope, receive, send)
    assert responses[0]["status"] == 404
    assert mgr.is_blocked("9.9.9.9") is True

    responses.clear()
    # Third request — should be 403 (blocked)
    await mw(scope, receive, send)
    assert responses[0]["status"] == 403
    body = responses[1]["body"]
    assert b"IP_BLOCKED" in body


@pytest.mark.asyncio
async def test_middleware_passes_non_http():
    """Non-HTTP scopes pass through untouched."""
    from src.middleware.ip_block import IPBlockMiddleware, IPBlockManager

    called = False

    async def dummy_app(scope, receive, send):
        nonlocal called
        called = True

    mw = IPBlockMiddleware(dummy_app)
    mw._manager = IPBlockManager(threshold=2, window_seconds=60)

    await mw({"type": "websocket"}, None, None)
    assert called is True


@pytest.mark.asyncio
async def test_middleware_extracts_xff():
    """X-Forwarded-For header should be used for client IP."""
    from src.middleware.ip_block import IPBlockMiddleware, IPBlockManager

    mgr = IPBlockManager(threshold=1, window_seconds=60, whitelist=set())

    async def dummy_404_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw = IPBlockMiddleware(dummy_404_app)
    mw._manager = mgr

    scope = {
        "type": "http",
        "headers": [(b"x-forwarded-for", b"77.77.77.77, 10.0.0.1")],
        "client": ("10.0.0.1", 80),
    }

    async def receive():
        return {"type": "http.request", "body": b""}

    responses = []
    async def send(msg):
        responses.append(msg)

    await mw(scope, receive, send)
    assert mgr.is_blocked("77.77.77.77") is True  # threshold=1
    assert mgr.is_blocked("10.0.0.1") is False
