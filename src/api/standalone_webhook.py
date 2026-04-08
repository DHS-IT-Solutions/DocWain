"""Webhook delivery with HMAC-SHA256 signing and retry logic."""
import hashlib
import hmac
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


def compute_signature(body: bytes, secret: str) -> str:
    """Return HMAC-SHA256 hex digest of body using secret."""
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def deliver_webhook(
    url: str,
    payload: Dict[str, Any],
    request_id: str,
    key_hash: str,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> bool:
    """POST JSON payload to url with HMAC signing, retrying on non-2xx responses.

    Args:
        url: Destination URL.
        payload: Dict to serialise as JSON body.
        request_id: Value for X-DocWain-Request-Id header.
        key_hash: Secret used for HMAC signature.
        max_retries: Maximum delivery attempts (default 3).
        backoff_base: Base seconds for exponential backoff (default 1.0).

    Returns:
        True on any successful (2xx) delivery, False after exhausting retries.
    """
    body = json.dumps(payload).encode()
    signature = compute_signature(body, key_hash)
    headers = {
        "Content-Type": "application/json",
        "X-DocWain-Request-Id": request_id,
        "X-DocWain-Signature": signature,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, data=body, headers=headers)
            if 200 <= response.status_code < 300:
                logger.debug(
                    "Webhook delivered to %s on attempt %d (status %d)",
                    url, attempt + 1, response.status_code,
                )
                return True
            logger.warning(
                "Webhook to %s returned non-2xx status %d on attempt %d",
                url, response.status_code, attempt + 1,
            )
        except Exception as exc:
            logger.warning(
                "Webhook to %s raised exception on attempt %d: %s",
                url, attempt + 1, exc,
            )

        if attempt < max_retries - 1:
            sleep_seconds = backoff_base * (5 ** attempt)
            time.sleep(sleep_seconds)

    logger.error("Webhook delivery to %s failed after %d attempts", url, max_retries)
    return False


def deliver_webhook_async(
    url: str,
    payload: Dict[str, Any],
    request_id: str,
    key_hash: str,
    on_complete: Optional[Callable[[bool], None]] = None,
) -> None:
    """Submit webhook delivery to thread pool.

    Args:
        url: Destination URL.
        payload: Dict to serialise as JSON body.
        request_id: Value for X-DocWain-Request-Id header.
        key_hash: Secret used for HMAC signature.
        on_complete: Optional callback invoked with True/False result when done.
    """
    def _run() -> None:
        result = deliver_webhook(url, payload, request_id, key_hash)
        if on_complete is not None:
            on_complete(result)

    _executor.submit(_run)
