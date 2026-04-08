"""Tests for standalone webhook delivery with HMAC signing and retries."""
import hashlib
import hmac
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# compute_signature
# ---------------------------------------------------------------------------

def test_compute_signature():
    from src.api.standalone_webhook import compute_signature

    body = b'{"status": "complete"}'
    secret = "mysecret"

    result = compute_signature(body, secret)

    assert isinstance(result, str), "signature must be a string"
    assert len(result) == 64, "HMAC-SHA256 hex digest must be 64 characters"

    # Verify deterministic
    assert compute_signature(body, secret) == result


def test_compute_signature_different_secrets():
    from src.api.standalone_webhook import compute_signature

    body = b'{"status": "complete"}'
    sig1 = compute_signature(body, "secret_one")
    sig2 = compute_signature(body, "secret_two")

    assert sig1 != sig2, "different secrets must produce different signatures"


# ---------------------------------------------------------------------------
# deliver_webhook
# ---------------------------------------------------------------------------

def test_deliver_webhook_success():
    from src.api.standalone_webhook import deliver_webhook

    mock_response = MagicMock()
    mock_response.status_code = 200

    payload = {"request_id": "req-001", "status": "complete"}

    with patch("src.api.standalone_webhook.requests.post", return_value=mock_response) as mock_post:
        result = deliver_webhook(
            url="https://example.com/webhook",
            payload=payload,
            request_id="req-001",
            key_hash="somekeyhash",
        )

    assert result is True
    mock_post.assert_called_once()

    call_kwargs = mock_post.call_args
    headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {}
    # Unpack more robustly
    _, kwargs = call_kwargs
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert kwargs["headers"]["X-DocWain-Request-Id"] == "req-001"
    assert "X-DocWain-Signature" in kwargs["headers"]
    assert len(kwargs["headers"]["X-DocWain-Signature"]) == 64


def test_deliver_webhook_retries_on_failure():
    from src.api.standalone_webhook import deliver_webhook

    mock_fail = MagicMock()
    mock_fail.status_code = 500

    mock_ok = MagicMock()
    mock_ok.status_code = 200

    payload = {"request_id": "req-002", "status": "complete"}

    with patch("src.api.standalone_webhook.requests.post", side_effect=[mock_fail, mock_ok]) as mock_post:
        result = deliver_webhook(
            url="https://example.com/webhook",
            payload=payload,
            request_id="req-002",
            key_hash="somekeyhash",
            backoff_base=0.01,
        )

    assert result is True
    assert mock_post.call_count == 2, "should have retried once after initial failure"
