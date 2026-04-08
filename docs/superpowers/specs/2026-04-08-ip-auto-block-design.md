# IP Auto-Block Middleware — Design Spec

**Date:** 2026-04-08
**Status:** Approved

## Overview

ASGI middleware that tracks 404 responses per client IP. When an IP exceeds 5 404s within a 60-second sliding window, it is permanently blocked. Blocked IPs receive 403 Forbidden immediately with zero request processing. Blocks persist to disk and survive restarts. A CLI script manages the block list.

## Configuration (Config.Security)

```python
class Security:
    BLOCK_THRESHOLD = int(os.getenv("SECURITY_BLOCK_THRESHOLD", "5"))
    BLOCK_WINDOW_SECONDS = int(os.getenv("SECURITY_BLOCK_WINDOW", "60"))
    WHITELIST_IPS = os.getenv("SECURITY_WHITELIST_IPS", "127.0.0.1,::1").split(",")
    BLOCKED_IPS_FILE = os.getenv("SECURITY_BLOCKED_IPS_FILE", "data/blocked_ips.json")
```

## Middleware (src/middleware/ip_block.py)

### IPBlockManager

In-memory state with file persistence:

- `_blocked: Dict[str, str]` — `{ip: blocked_at_iso}` permanent block list
- `_tracker: Dict[str, List[float]]` — `{ip: [timestamps]}` sliding window tracker
- `_whitelist: Set[str]` — IPs that are never blocked

Methods:
- `is_blocked(ip) -> bool`
- `record_404(ip) -> bool` — records a 404, returns True if IP was just blocked
- `block(ip, reason="") -> None` — manually add to block list
- `unblock(ip) -> bool` — remove from block list
- `list_blocked() -> Dict[str, str]` — return all blocked IPs with timestamps
- `_save()` / `_load()` — persist to/from JSON file

### IPBlockMiddleware

Pure ASGI middleware (not Starlette BaseHTTPMiddleware — lower overhead):

1. Extract client IP from `X-Forwarded-For` header (first entry) or `client.host`
2. If IP in blocked set → return 403 JSON immediately, log it
3. Pass request through to app
4. If response status is 404 → call `record_404(ip)`
5. If record_404 returns True (just blocked) → log block event

403 response body:
```json
{"error": {"code": "IP_BLOCKED", "message": "Access denied"}}
```

## Files

| File | Purpose |
|------|---------|
| `src/middleware/ip_block.py` | IPBlockManager + IPBlockMiddleware |
| `scripts/manage_blocked_ips.py` | CLI: list, block, unblock IPs |
| `src/api/config.py` | Add Config.Security section |
| `src/main.py` | Add middleware (outermost, before CORS) |

## CLI (scripts/manage_blocked_ips.py)

```
python scripts/manage_blocked_ips.py list
python scripts/manage_blocked_ips.py block --ip 1.2.3.4
python scripts/manage_blocked_ips.py unblock --ip 1.2.3.4
```
