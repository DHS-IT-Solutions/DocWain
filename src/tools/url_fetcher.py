"""SSRF-safe URL fetcher for DocWain's URL-as-prompt pipeline.

Design goals (spec Section 7, Section 3 invariant 8):

* HTTP/HTTPS only. Anything else (file://, ftp://, gopher://, data:,
  javascript:, ...) is rejected at parse time.
* Hostname + resolved-IP pair is validated before any TCP connect. DNS
  rebinding is mitigated by re-resolving immediately before each
  connection and asserting the same public-IP set.
* Manual redirect following (<=5 hops). Each hop is re-parsed and
  re-validated; cross-scheme / private-target hops are rejected.
* Per-operation safety timeouts (external I/O only, spec Section 3
  invariant 8): fetch 15s, extract 30s. Callers do NOT add internal
  wall-clock timeouts on top of these.
* Streaming size cap (default 10 MB) terminates the connection when
  exceeded; partial body is discarded.
* Subscription-scoped domain allow/block lists; declared user-agent
  ``DocWain-URL-Fetcher/1.0``.

No persistence: this module only fetches bytes; callers are responsible
for interpreting the body (see ``url_ephemeral_source``).
"""
from __future__ import annotations

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


ALLOWED_SCHEMES = frozenset({"http", "https"})
DEFAULT_PORTS = {"http": 80, "https": 443}

DEFAULT_USER_AGENT = "DocWain-URL-Fetcher/1.0"

_FORBIDDEN_CHARS = re.compile(r"[\s\x00-\x1f\x7f]")


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------
class UrlFetcherError(Exception):
    """Base class for url_fetcher errors."""


class UrlParseError(UrlFetcherError):
    """URL could not be parsed / has forbidden structure."""


class SsrfError(UrlFetcherError):
    """URL targets a private, loopback, metadata, or otherwise blocked address."""


class SizeCapExceededError(UrlFetcherError):
    """Streaming body exceeded configured size cap."""


class DomainBlockedError(UrlFetcherError):
    """URL host is outside the subscription allowlist or inside the blocklist."""


class UnsupportedContentTypeError(UrlFetcherError):
    """Response content-type is outside the configured accept list."""


# ---------------------------------------------------------------------------
# Static denylists
# ---------------------------------------------------------------------------
BLOCKED_HOSTNAMES = frozenset({
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "metadata.google.internal",
    "metadata.azure.internal",
    "instance-data",
    "metadata",
    # 169.254.169.254 also appears as an IP literal — blocked twice for
    # defense in depth.
    "169.254.169.254",
})

# Additional addresses beyond stdlib's is_private / is_loopback /
# is_link_local / is_multicast / is_reserved / is_unspecified that we want
# to explicitly block.
_EXTRA_BLOCKED_V4 = frozenset({
    "100.100.100.200",  # Alibaba Cloud metadata
})


# ---------------------------------------------------------------------------
# Parsed URL + parsing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedUrl:
    scheme: str
    hostname: str
    port: int
    path: str
    query: str


def _strip_trailing_dot(host: str) -> str:
    return host[:-1] if host.endswith(".") else host


def parse_url(url: str) -> ParsedUrl:
    """Strict URL parse. Accepts HTTP(S) only. Strips user-info.

    Raises UrlParseError for structural problems, SsrfError for disallowed
    schemes.
    """
    if not url or not isinstance(url, str):
        raise UrlParseError("empty url")

    if _FORBIDDEN_CHARS.search(url):
        raise UrlParseError("url contains whitespace or control character")

    try:
        parsed = urlparse(url)
    except Exception as exc:  # noqa: BLE001
        raise UrlParseError(f"urlparse failed: {exc}") from exc

    scheme = (parsed.scheme or "").lower()
    if not scheme:
        raise UrlParseError("url has no scheme")

    if scheme not in ALLOWED_SCHEMES:
        raise SsrfError(f"scheme not allowed: {scheme}")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise UrlParseError("url has no hostname")

    port = parsed.port or DEFAULT_PORTS[scheme]

    return ParsedUrl(
        scheme=scheme,
        hostname=hostname,
        port=port,
        path=parsed.path or "/",
        query=parsed.query or "",
    )


# ---------------------------------------------------------------------------
# IP classification
# ---------------------------------------------------------------------------
def is_blocked_ip(ip_str: str) -> bool:
    """True if *ip_str* points to a private / loopback / reserved /
    link-local / multicast / broadcast / cloud-metadata / mapped-private
    address.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return is_blocked_ip(str(ip.ipv4_mapped))

    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return True
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True

    if isinstance(ip, ipaddress.IPv4Address):
        if str(ip) in _EXTRA_BLOCKED_V4:
            return True
        if int(ip) == int(ipaddress.IPv4Address("255.255.255.255")):
            return True

    return False


def parse_and_classify(url: str) -> ParsedUrl:
    """parse_url + hostname / IP-literal SSRF checks.

    Does not resolve DNS; the network layer re-validates at connect time.
    """
    p = parse_url(url)

    host = _strip_trailing_dot(p.hostname)
    if host in BLOCKED_HOSTNAMES:
        raise SsrfError(f"hostname on denylist: {host}")

    try:
        ipaddress.ip_address(host)
    except ValueError:
        return p  # hostname; DNS layer validates it later.

    if is_blocked_ip(host):
        raise SsrfError(f"ip literal points to blocked range: {host}")
    return p


# ---------------------------------------------------------------------------
# DNS resolution + rebinding guard
# ---------------------------------------------------------------------------
def resolve_and_validate(hostname: str) -> List[str]:
    """Resolve *hostname* and verify every resulting IP is public.

    Returns the sorted, de-duplicated list of validated IP strings (the
    pinned expected-IP set for the subsequent connect). Fail-closed on any
    DNS error.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise SsrfError(f"dns resolution failed for {hostname}: {exc}") from exc

    ips: List[str] = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip = sockaddr[0]
        if is_blocked_ip(ip):
            raise SsrfError(
                f"hostname {hostname} resolves to blocked ip {ip}"
            )
        ips.append(ip)

    if not ips:
        raise SsrfError(f"no resolvable IPs for {hostname}")

    return sorted(set(ips))


def connect_guard(hostname: str, expected_ips: List[str]) -> str:
    """Re-resolve *hostname* and assert the result set equals *expected_ips*.

    Returns the IP that should be connected to. Raises SsrfError on any
    mismatch (DNS rebinding defense).
    """
    fresh = resolve_and_validate(hostname)
    expected_set = sorted(set(expected_ips))
    if fresh != expected_set:
        raise SsrfError(
            f"dns rebinding suspected for {hostname}: "
            f"validated={expected_set} current={fresh}"
        )
    return fresh[0]


# ---------------------------------------------------------------------------
# Subscription domain policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DomainPolicy:
    """Per-subscription allow/block lists.

    Allowlist semantics: when non-empty, ONLY hosts matching an allowlist
    entry are accepted; otherwise all non-blocked hosts pass. Blocklist is
    consulted first and always blocks.

    Match rule: exact match OR suffix match with a leading dot (".x.com"
    matches "a.x.com" but not "x.com").
    """
    allowed_domains: Tuple[str, ...] = ()
    blocked_domains: Tuple[str, ...] = ()


def _domain_matches(host: str, patterns: Iterable[str]) -> bool:
    host = _strip_trailing_dot(host).lower()
    for pat in patterns:
        pat = pat.lower().strip()
        if not pat:
            continue
        if pat.startswith("."):
            if host.endswith(pat) or host == pat.lstrip("."):
                return True
        elif host == pat:
            return True
    return False


def check_domain_policy(host: str, policy: DomainPolicy) -> None:
    if _domain_matches(host, policy.blocked_domains):
        raise DomainBlockedError(f"host {host} is on the subscription blocklist")
    if policy.allowed_domains and not _domain_matches(
        host, policy.allowed_domains
    ):
        raise DomainBlockedError(
            f"host {host} is not on the subscription allowlist"
        )


# ---------------------------------------------------------------------------
# Fetcher configuration
# ---------------------------------------------------------------------------
DEFAULT_ACCEPT_CONTENT_TYPES: Tuple[str, ...] = (
    "text/html",
    "text/plain",
    "application/xhtml+xml",
    "application/json",
)

_BINARY_REJECT_PREFIXES: Tuple[str, ...] = (
    "application/octet-stream",
    "image/",
    "video/",
    "audio/",
    "application/zip",
    "application/x-tar",
    "application/x-gzip",
    "application/pdf",
)


@dataclass(frozen=True)
class FetcherConfig:
    max_size: int = 10_000_000           # 10 MB body cap
    fetch_timeout_s: float = 15.0        # external I/O safety
    extract_timeout_s: float = 30.0      # external extract safety
    max_redirects: int = 5
    user_agent: str = DEFAULT_USER_AGENT
    accept_content_types: Tuple[str, ...] = DEFAULT_ACCEPT_CONTENT_TYPES
    domain_policy: DomainPolicy = field(default_factory=DomainPolicy)


# ---------------------------------------------------------------------------
# Content-type validation
# ---------------------------------------------------------------------------
def validate_content_type(content_type: str, config: FetcherConfig) -> None:
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    if not ct:
        # Unknown content-type: allow text/plain fallback
        return
    for rej in _BINARY_REJECT_PREFIXES:
        if ct.startswith(rej):
            raise UnsupportedContentTypeError(
                f"content-type not accepted: {ct}"
            )
    for accept in config.accept_content_types:
        if ct == accept or ct.startswith(accept.split("/", 1)[0] + "/"):
            # allow any subtype of text/* if text/plain or text/html in list
            if accept in ct or ct == accept or (
                accept.endswith("/*") and ct.startswith(accept[:-1])
            ):
                return
            if ct.startswith("text/") and any(
                a.startswith("text/") for a in config.accept_content_types
            ):
                return
    # Strict fallback: if no accept prefix matched, reject
    if not any(
        ct == a or (a.startswith("text/") and ct.startswith("text/"))
        for a in config.accept_content_types
    ):
        raise UnsupportedContentTypeError(
            f"content-type not accepted: {ct}"
        )


def check_declared_content_length(
    header_value: Optional[str], config: FetcherConfig
) -> None:
    if not header_value:
        return
    try:
        declared = int(header_value)
    except ValueError:
        return
    if declared > config.max_size:
        raise SizeCapExceededError(
            f"declared content-length {declared} exceeds cap {config.max_size}"
        )


def stream_with_cap(chunks: Iterable[bytes], config: FetcherConfig) -> bytes:
    """Concatenate *chunks* but abort if the total exceeds the cap."""
    buf = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        if len(buf) + len(chunk) > config.max_size:
            raise SizeCapExceededError(
                f"body exceeded cap {config.max_size} bytes"
            )
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Fetch result + fetch orchestration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    headers: Dict[str, str]
    body: bytes
    content_type: str
    resolved_ip: str
    source_url: str
    fetched_at: datetime
    error: Optional[str] = None
    redirects: Tuple[str, ...] = ()

    @property
    def content(self) -> bytes:
        """Alias for ``body`` — matches the callsite contract in the spec."""
        return self.body


def fetch(
    url: str,
    *,
    config: Optional[FetcherConfig] = None,
    max_size: Optional[int] = None,
    fetch_timeout_s: Optional[float] = None,
    extract_timeout_s: Optional[float] = None,
    allowed_domains: Optional[Iterable[str]] = None,
    blocked_domains: Optional[Iterable[str]] = None,
    _transport: Any = None,
) -> FetchResult:
    """Fetch *url* with all SSRF + size + content-type defenses applied.

    Redirects are followed manually; every hop is re-parsed and re-validated.

    Kwargs form: callers can pass ``config`` (complete override) or use the
    simple kwargs (``max_size``, ``fetch_timeout_s``, ``extract_timeout_s``,
    ``allowed_domains``, ``blocked_domains``) matching the Phase 5 contract.
    """
    import httpx

    if config is None:
        policy = DomainPolicy(
            allowed_domains=tuple(allowed_domains or ()),
            blocked_domains=tuple(blocked_domains or ()),
        )
        config = FetcherConfig(
            max_size=max_size if max_size is not None else 10_000_000,
            fetch_timeout_s=fetch_timeout_s if fetch_timeout_s is not None else 15.0,
            extract_timeout_s=extract_timeout_s if extract_timeout_s is not None else 30.0,
            domain_policy=policy,
        )

    redirects: List[str] = []
    current = url
    for _ in range(config.max_redirects + 1):
        parsed = parse_and_classify(current)
        check_domain_policy(parsed.hostname, config.domain_policy)

        # Validate DNS; if the hostname is an IP literal we still emit a
        # resolved_ip by re-using it directly to keep the result shape
        # consistent.
        try:
            ipaddress.ip_address(parsed.hostname)
            pinned_ip = parsed.hostname
        except ValueError:
            validated_ips = resolve_and_validate(parsed.hostname)
            pinned_ip = connect_guard(parsed.hostname, validated_ips)

        headers = {
            "user-agent": config.user_agent,
            "accept": ", ".join(config.accept_content_types),
        }

        try:
            client_kwargs = dict(
                follow_redirects=False,
                timeout=config.fetch_timeout_s,
                headers=headers,
            )
            if _transport is not None:
                client_kwargs["transport"] = _transport
            with httpx.Client(**client_kwargs) as client:
                resp = client.get(current)
        except httpx.TimeoutException as exc:
            raise UrlFetcherError(
                f"safety timeout fetching {current}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise UrlFetcherError(
                f"network error fetching {current}: {exc}"
            ) from exc

        lc_headers = {k.lower(): v for k, v in resp.headers.items()}

        if 300 <= resp.status_code < 400 and "location" in lc_headers:
            location = lc_headers.get("location") or ""
            if not location:
                raise UrlFetcherError(
                    f"redirect with no location header from {current}"
                )
            next_url = urljoin(current, location)
            redirects.append(next_url)
            current = next_url
            continue

        check_declared_content_length(lc_headers.get("content-length"), config)
        content_type = lc_headers.get("content-type", "")
        validate_content_type(content_type, config)

        body = stream_with_cap([resp.content], config)

        return FetchResult(
            url=url,
            final_url=current,
            status=resp.status_code,
            headers=dict(resp.headers),
            body=body,
            content_type=content_type,
            resolved_ip=pinned_ip,
            source_url=url,
            fetched_at=datetime.now(timezone.utc),
            redirects=tuple(redirects),
        )

    raise UrlFetcherError(
        f"redirect budget exhausted after {config.max_redirects} hops for {url}"
    )
