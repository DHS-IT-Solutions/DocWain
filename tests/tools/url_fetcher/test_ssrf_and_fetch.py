"""Lean SSRF-safety tests for url_fetcher.

Focus: non-negotiable SSRF coverage + one happy-path fetch. This is the
Phase-5 minimum — the full SSRF matrix is deferred to ops hardening.
"""
from __future__ import annotations

import socket
from unittest.mock import patch

import httpx
import pytest

from src.tools.url_fetcher import (
    DomainBlockedError,
    DomainPolicy,
    FetcherConfig,
    SizeCapExceededError,
    SsrfError,
    UrlFetcherError,
    check_domain_policy,
    fetch,
    is_blocked_ip,
    parse_and_classify,
    stream_with_cap,
)


# ---------------------------------------------------------------------------
# Test 1: private IP + scheme block at parse time
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "http://127.0.0.1/",
    "http://10.0.0.1/",
    "http://172.16.0.1/",
    "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/",
    "http://localhost/",
    "http://metadata.google.internal/",
    "file:///etc/passwd",
    "ftp://example.com/",
    "gopher://example.com/",
])
def test_private_or_non_http_blocked_at_parse(url):
    with pytest.raises(SsrfError):
        parse_and_classify(url)


def test_public_ip_not_blocked():
    assert is_blocked_ip("8.8.8.8") is False
    assert is_blocked_ip("1.1.1.1") is False


# ---------------------------------------------------------------------------
# Test 2: DNS resolution to private IP is blocked at connect-validate time
# ---------------------------------------------------------------------------
def test_dns_resolves_to_private_ip_blocked(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.5", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    def _handler(request):  # pragma: no cover — should never be called
        return httpx.Response(200, content=b"x")

    with pytest.raises(SsrfError):
        fetch("https://evil.example/", _transport=httpx.MockTransport(_handler))


# ---------------------------------------------------------------------------
# Test 3: redirect to private IP is blocked
# ---------------------------------------------------------------------------
def test_redirect_to_private_ip_blocked(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        # entry.example resolves public; loopback literal is blocked at parse.
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    def handler(request):
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    with pytest.raises(SsrfError):
        fetch("https://entry.example/", _transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Test 4: oversize body aborts
# ---------------------------------------------------------------------------
def test_stream_with_cap_raises_on_oversize():
    cfg = FetcherConfig(max_size=100)
    with pytest.raises(SizeCapExceededError):
        stream_with_cap([b"x" * 60, b"y" * 60], cfg)


# ---------------------------------------------------------------------------
# Test 5: happy-path public fetch with mocked transport
# ---------------------------------------------------------------------------
def test_happy_path_public_fetch(monkeypatch):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html>hello</html>",
        )

    result = fetch(
        "https://example.com/",
        _transport=httpx.MockTransport(handler),
    )
    assert result.status == 200
    assert b"hello" in result.body
    assert result.content_type.startswith("text/html")
    assert result.resolved_ip == "93.184.216.34"
    assert result.source_url == "https://example.com/"


# ---------------------------------------------------------------------------
# Test 6: domain blocklist
# ---------------------------------------------------------------------------
def test_domain_blocklist_rejects_match():
    policy = DomainPolicy(blocked_domains=("badguy.com", ".blacklisted.net"))
    with pytest.raises(DomainBlockedError):
        check_domain_policy("badguy.com", policy)
    with pytest.raises(DomainBlockedError):
        check_domain_policy("sub.blacklisted.net", policy)


def test_domain_allowlist_rejects_unlisted():
    policy = DomainPolicy(allowed_domains=("docs.company.com",))
    # Allowed host passes.
    check_domain_policy("docs.company.com", policy)
    # Unlisted host fails.
    with pytest.raises(DomainBlockedError):
        check_domain_policy("randomblog.net", policy)
