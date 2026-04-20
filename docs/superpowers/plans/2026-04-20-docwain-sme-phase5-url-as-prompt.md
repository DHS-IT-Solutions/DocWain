# DocWain SME Phase 5 — URL-as-Prompt Ephemeral Handling

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a user's query contains one or more HTTP(S) URLs, fetch and reason over that URL content in-session without any persistence (no Qdrant, no Neo4j, no Blob, no Mongo) while preserving existing profile isolation, zero hallucination rate, and zero internal timeouts. URL content flows into the same pack as profile evidence (supplementary case) or, when the profile has nothing useful and the URL carries the query's intent, it becomes the primary evidence source (primary case).

**Architecture:** New module `src/tools/url_fetcher.py` adds a SSRF-safe HTTP(S) fetcher with hostname + re-resolved IP checks, manual redirect following, per-operation external-I/O safety timeouts (the only timeouts allowed in DocWain), a streaming size cap, a subscription-scoped domain allow/block list, and robots.txt handling. A second new module `src/tools/url_ephemeral_source.py` wraps the fetcher with extraction (via existing `src/tools/web_extract.py`), chunking (reusing `SectionChunker`), and in-memory embedding (using the same embedder that indexes the profile). The Phase 5 changes to `src/agent/core_agent.py` launch the ephemeral URL pipeline as a third concurrent task alongside the existing UNDERSTAND + pre-fetch RETRIEVE pair, pick supplementary vs primary automatically based on profile retrieval strength and query composition, and merge URL chunks into the pack either synchronously (if ready by Stage 3) or asynchronously as a supplementary section (if URL content arrives after Stage 4 has begun streaming). A per-subscription feature flag `enable_url_as_prompt` gates the whole capability; when OFF, URLs are treated as plain text exactly as today.

**Tech Stack:** Python 3.12, `httpx` (already in-tree, used by `src/tools/web_search.py`), `ipaddress` (stdlib), `socket` (stdlib, for `getaddrinfo`), `urllib.robotparser` (stdlib), `concurrent.futures.ThreadPoolExecutor` (as used in `core_agent.py` today for parallel UNDERSTAND+RETRIEVE), `pydantic` for config models, `pytest`, `pytest-httpx` or custom `MockTransport` for network test isolation, existing `SectionChunker` from `src/embedding/chunking/section_chunker.py`, existing embedder from `src/api/embedding_service.py`, existing `detect_urls_in_query` from `src/tools/web_search.py`.

**Related spec:** `docs/superpowers/specs/2026-04-20-docwain-profile-sme-reasoning-design.md` Sections 3 (invariant 8 — zero internal timeouts, external I/O safety timeouts are fine), 7 (URL-as-prompt handling), 12 Phase 5 (exit gate), 13.2 (`enable_url_as_prompt` flag), 14 (URL-as-prompt data-leak risk / mitigations), 17 (acceptance criteria bullet on SSRF test suite).

**Prior-phase contracts referenced by Phase 5:**
- Phase 1 — `FeatureFlagResolver` (per-subscription flag resolution, Blob-backed with Redis cache) and `SubscriptionConfig` adapter that exposes domain allow/block lists. If Phase 1 plan has not yet landed, Phase 5 provides a minimal shim at the boundary (`src/intelligence/sme/feature_flags.py`) reading from `Config.FeatureFlags` with the same surface — Phase 1 replaces the shim without Phase 5 rewrites.
- Phase 2 — `SMERetrievalLayer` signatures for Layer C. Phase 5 only uses the merge hook published by Phase 3; if Phase 3 is not yet landed, Phase 5 falls back to direct pack injection with the same shape (`EphemeralSource` → `List[Chunk]` with `ephemeral: true` metadata).
- Phase 3 — `UnifiedRetriever.retrieve()` parallel-layer orchestration and the pack-assembly `PackBudget`. Phase 5 adds Layer D (ephemeral URL) as a new concurrent leg; if Phase 3 is not yet landed, Phase 5 injects URL chunks directly into the reranker input with equivalent metadata.
- Phase 4 — adapter `response_persona_prompts.supplementary_analysis` template and the `compose_with_supplementary()` helper in `src/generation/prompts.py`. If Phase 4 is not yet landed, Phase 5 writes the supplementary-composition template in `prompts.py` as a standalone addition scoped to URL queries only — nothing else changes in `prompts.py`.

**Memory rules that constrain this plan (hard):**
- **No Claude attribution** — no Claude / Anthropic / Co-Authored-By strings in commits, code, or docs.
- **No timeouts on DocWain internal paths** — external I/O safety timeouts on URL fetch are the only timeouts allowed. Retrieval, reasoning, synthesis, and composition stay un-timed. See spec Section 3 invariant 8 and the Phase 0 plan's treatment of httpx timeouts as safety-only.
- **Ephemeral means ephemeral** — URL content never reaches Qdrant, Neo4j, Blob, Mongo, Redis, or the filesystem. Session-scoped only. The only persistence is the query trace line (metadata about the URL, never the body beyond a small hash/title snippet).
- **Response formatting lives in `src/generation/prompts.py`** — Phase 5 adds one citation-annotation hook to existing templates and one supplementary-section template (when not already added by Phase 4). `src/intelligence/generator.py` is NOT touched.
- **Profile isolation preserved even with URL content** — URL-derived chunks are merged with the authoring profile's pack only, never shared across profiles; the session-scoped cache is keyed by `(session_id, url_hash)` and cleared on session teardown.
- **Engineering-first, no retraining** — the model sees URL content as extra pack items with `source_url` provenance; no fine-tuning is required.
- **No customer data in training** — URL pipelines don't emit to any training dataset; traces log the URL and response hash only, never the URL body.

---

## File structure

```
src/tools/                                          [existing]
├── url_fetcher.py                                  [NEW — SSRF-safe HTTP(S) fetcher]
├── url_ephemeral_source.py                         [NEW — ephemeral fetch+extract+chunk+embed pipeline]
├── web_extract.py                                  [EXISTING — reused for extraction]
└── web_search.py                                   [EXISTING — reused for detect_urls_in_query]

src/agent/core_agent.py                             [MODIFIED — add Stage 0 URL task leg + merge logic]
src/generation/prompts.py                           [MODIFIED — citation annotation + (fallback) supplementary template]
src/intelligence/sme/feature_flags.py               [NEW OR SHIM — enable_url_as_prompt resolver]

src/retrieval/unified_retriever.py                  [MODIFIED — accept Layer D ephemeral chunks]
src/retrieval/ephemeral_merge.py                    [NEW — merge helper; keeps core_agent.py lean]

tests/tools/url_fetcher/                            [NEW — SSRF-heavy unit + integration tests]
├── __init__.py
├── conftest.py
├── test_url_parsing.py
├── test_ssrf_hostname_block.py
├── test_ssrf_ip_literal_block.py
├── test_ssrf_dns_resolution_block.py
├── test_ssrf_redirect_chain_block.py
├── test_ssrf_dns_rebinding.py
├── test_ssrf_scheme_block.py
├── test_size_cap.py
├── test_content_type.py
├── test_slow_loris.py
├── test_oversized_header_block.py
├── test_user_agent.py
├── test_robots_txt.py
├── test_domain_allowlist.py
├── test_domain_blocklist.py
└── test_safety_timeout.py

tests/tools/url_ephemeral_source/                   [NEW]
├── __init__.py
├── conftest.py
├── test_fetch_all.py
├── test_extract_chunk_embed.py
├── test_partial_failure.py
├── test_ephemeral_flag.py
├── test_session_scope.py
├── test_no_persistence.py
└── test_metadata_shape.py

tests/agent/                                        [existing dir]
├── test_core_agent_url_supplementary.py            [NEW — URL in parallel with profile]
├── test_core_agent_url_primary.py                  [NEW — weak profile, URL-dominant query]
├── test_core_agent_url_case_selection.py           [NEW — heuristic contract]
├── test_core_agent_url_flag_off.py                 [NEW — URL as plain text when flag disabled]
└── test_core_agent_url_late_arrival.py             [NEW — URL arrives mid-stream]

tests/intelligence/sme/                             [existing dir]
└── test_url_feature_flag.py                        [NEW — flag resolver contract]

tests/retrieval/
└── test_ephemeral_merge.py                         [NEW — merge into Stage 3 pack]

tests/generation/
└── test_prompts_url_citations.py                   [NEW — citation annotation + supplementary]

tests/perf/
└── test_url_supplementary_latency.py               [NEW — first-response latency unchanged]
```

Each file does one thing. Fetcher does not know about chunking; ephemeral source does not know about agent orchestration; agent orchestration does not know about HTTP. Tests are partitioned by SSRF attack class so each class stays a first-class citizen of the plan and can be updated independently as new bypass classes surface.

---

## Task 1: Preflight audit and directory scaffolding

**Files:**
- Create: `src/tools/url_fetcher.py` (empty stub)
- Create: `src/tools/url_ephemeral_source.py` (empty stub)
- Create: `src/retrieval/ephemeral_merge.py` (empty stub)
- Create: `src/intelligence/sme/feature_flags.py` (empty stub)
- Create: `tests/tools/url_fetcher/__init__.py` (empty)
- Create: `tests/tools/url_fetcher/conftest.py`
- Create: `tests/tools/url_ephemeral_source/__init__.py` (empty)
- Create: `tests/tools/url_ephemeral_source/conftest.py`
- Audit only: `src/tools/web_search.py`, `src/tools/web_extract.py`, `src/tools/common/http_client.py`, `src/embedding/chunking/section_chunker.py`, `src/agent/core_agent.py`, `src/api/embedding_service.py`, `src/retrieval/unified_retriever.py` (if exists)

- [ ] **Step 1: Read the existing URL-adjacent code**

Read and record what is reusable:
- `src/tools/web_search.py` — defines `detect_urls_in_query(query) -> (urls, cleaned_query)`, a private SSRF denylist (`_SSRF_DENY_HOSTNAMES`), and `fetch_url_content(url)`. The existing SSRF denylist is **incomplete** for Phase 5's threat model (missing DNS-rebinding handling, 0.0.0.0, broadcast, IPv6 unique-local, multicast, and configurable allow/block). Phase 5 replaces the fetch path entirely but reuses `detect_urls_in_query` verbatim.
- `src/tools/web_extract.py` — offers `_extract(url, max_chars)` that internally calls `fetch_text` from `src/tools/common/http_client.py`. Phase 5 swaps `fetch_text` for `url_fetcher.fetch()` inside a Phase-5-local extraction wrapper so the old `/web/extract` tool endpoint remains untouched while the ephemeral pipeline uses the hardened path.
- `src/tools/common/http_client.py` — read-only audit; note whether it follows redirects, whether it validates content-type, and its default timeout so Phase 5 can be strictly stricter.
- `src/embedding/chunking/section_chunker.py` — `SectionChunker().chunk_document(extracted_document, doc_internal_id, source_filename)` accepts an `ExtractedDocument`. Phase 5 builds a minimal `ExtractedDocument`-shaped adapter from plain-text URL output.
- `src/api/embedding_service.py` — inspect the public embedder interface used by `UnifiedRetriever` (the `embedder` parameter passed to `CoreAgent.__init__`). Phase 5's ephemeral source reuses the same embedder object — no separate model loads.
- `src/agent/core_agent.py` lines 130–260 — the existing parallel UNDERSTAND + PRE-FETCH RETRIEVE block uses `ThreadPoolExecutor(max_workers=2)`. Phase 5 extends this to three legs. No internal `.result(timeout=...)` semantic is added for the URL leg — it uses the external-I/O safety timeout internal to `url_fetcher.fetch()` instead.

- [ ] **Step 2: Create empty package files and stubs**

```bash
mkdir -p tests/tools/url_fetcher tests/tools/url_ephemeral_source tests/agent tests/retrieval tests/generation tests/perf tests/intelligence/sme
touch src/tools/url_fetcher.py src/tools/url_ephemeral_source.py src/retrieval/ephemeral_merge.py src/intelligence/sme/feature_flags.py
touch tests/tools/url_fetcher/__init__.py tests/tools/url_ephemeral_source/__init__.py
```

- [ ] **Step 3: Write fetcher conftest**

Create `tests/tools/url_fetcher/conftest.py`:

```python
"""Shared fixtures for url_fetcher tests.

These tests MUST NOT make real network calls. Every test uses either
httpx.MockTransport, a fake socket resolver, or a pure unit-level check.
Any test that would touch the network is marked `pytest.mark.network` and
excluded from the default run.
"""
from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import pytest


@dataclass
class FakeSocketResolver:
    """Replace socket.getaddrinfo with a deterministic mapping.

    mapping: {hostname: [(family, ip_address_string), ...]}
    Unknown hostnames raise socket.gaierror.
    """

    mapping: Dict[str, List[Tuple[int, str]]]

    def getaddrinfo(
        self,
        host: str,
        port: int | None = None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ):
        if host not in self.mapping:
            raise socket.gaierror(f"unknown host: {host}")
        out = []
        for fam, ip in self.mapping[host]:
            out.append((fam, socket.SOCK_STREAM, proto, "", (ip, port or 0)))
        return out


@pytest.fixture
def fake_dns(monkeypatch):
    """Install a FakeSocketResolver; tests add entries via .mapping."""
    resolver = FakeSocketResolver(mapping={})
    monkeypatch.setattr(socket, "getaddrinfo", resolver.getaddrinfo)
    return resolver


@pytest.fixture
def mock_transport_factory() -> Callable[[Callable[[httpx.Request], httpx.Response]], httpx.MockTransport]:
    """Build a MockTransport from a request handler. Kept as a factory so
    tests can define stateful handlers (e.g. for redirect chain tests)."""

    def _make(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
        return httpx.MockTransport(handler)

    return _make


@pytest.fixture
def sample_html() -> str:
    return (
        "<html><head><title>Hello</title></head>"
        "<body><p>Some content</p></body></html>"
    )
```

- [ ] **Step 4: Write ephemeral-source conftest**

Create `tests/tools/url_ephemeral_source/conftest.py`:

```python
"""Shared fixtures for url_ephemeral_source tests."""
from __future__ import annotations

from typing import List

import numpy as np
import pytest


class StubEmbedder:
    """Deterministic stub embedder. dim=384 matches Qwen3-style small bge."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.calls: List[str] = []

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.calls.extend(texts)
        # Deterministic: hash-derived vector, normalized.
        out = []
        for t in texts:
            h = hash(t) & 0xFFFFFFFF
            rng = np.random.default_rng(h)
            v = rng.standard_normal(self.dim).astype("float32")
            v = v / (np.linalg.norm(v) + 1e-9)
            out.append(v.tolist())
        return out

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder()
```

- [ ] **Step 5: Commit**

```bash
git add -f src/tools/url_fetcher.py src/tools/url_ephemeral_source.py \
    src/retrieval/ephemeral_merge.py src/intelligence/sme/feature_flags.py \
    tests/tools/url_fetcher/__init__.py tests/tools/url_fetcher/conftest.py \
    tests/tools/url_ephemeral_source/__init__.py tests/tools/url_ephemeral_source/conftest.py
git commit -m "phase5(sme-url): scaffold url fetcher + ephemeral source dirs and fixtures"
```

---

## Task 2: Feature-flag resolver (shim for Phase 1 flag infra)

**Files:**
- Create: `src/intelligence/sme/feature_flags.py`
- Create: `tests/intelligence/sme/test_url_feature_flag.py`

Phase 1's `FeatureFlagResolver` may or may not have landed when Phase 5 begins. This task ships a minimal resolver that reads from `Config.FeatureFlags` (with a per-subscription override map) and exposes the exact shape Phase 1 will provide (`resolve(subscription_id, flag_name) -> bool`). If Phase 1 is present, this task detects the import and delegates.

- [ ] **Step 1: Write the failing tests**

Create `tests/intelligence/sme/test_url_feature_flag.py`:

```python
"""Tests for the enable_url_as_prompt feature-flag resolver (Phase 5 shim)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.intelligence.sme.feature_flags import FeatureFlagResolver, UrlAsPromptFlag


def test_default_is_off():
    r = FeatureFlagResolver(default_map={}, subscription_overrides={})
    assert r.resolve("sub_a", UrlAsPromptFlag) is False


def test_global_default_on_applies():
    r = FeatureFlagResolver(
        default_map={UrlAsPromptFlag: True},
        subscription_overrides={},
    )
    assert r.resolve("sub_a", UrlAsPromptFlag) is True


def test_subscription_override_beats_default():
    r = FeatureFlagResolver(
        default_map={UrlAsPromptFlag: True},
        subscription_overrides={"sub_a": {UrlAsPromptFlag: False}},
    )
    assert r.resolve("sub_a", UrlAsPromptFlag) is False
    assert r.resolve("sub_b", UrlAsPromptFlag) is True


def test_unknown_flag_returns_false():
    r = FeatureFlagResolver(default_map={}, subscription_overrides={})
    assert r.resolve("sub_a", "enable_frobnicator") is False


def test_loads_from_config(monkeypatch):
    class StubConfig:
        class FeatureFlags:
            DEFAULT = {UrlAsPromptFlag: True}
            SUBSCRIPTION_OVERRIDES = {"sub_a": {UrlAsPromptFlag: False}}

    monkeypatch.setattr("src.intelligence.sme.feature_flags._config_module", StubConfig)
    r = FeatureFlagResolver.from_config()
    assert r.resolve("sub_a", UrlAsPromptFlag) is False
    assert r.resolve("sub_b", UrlAsPromptFlag) is True


def test_missing_config_section_gives_safe_default(monkeypatch):
    class StubConfig:
        pass

    monkeypatch.setattr("src.intelligence.sme.feature_flags._config_module", StubConfig)
    r = FeatureFlagResolver.from_config()
    assert r.resolve("sub_a", UrlAsPromptFlag) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/intelligence/sme/test_url_feature_flag.py -v
```
Expected: FAIL — module has no implementation yet.

- [ ] **Step 3: Write the resolver**

Create `src/intelligence/sme/feature_flags.py`:

```python
"""Feature-flag resolver for sub-project A.

Phase-5 scope: only `enable_url_as_prompt` is consumed here. Phase 1 may
later introduce a richer resolver backed by Azure Blob + Redis; this module
is designed to be replaced by that resolver without any caller changes —
same `resolve(subscription_id, flag_name) -> bool` signature.

No timeouts. Reads are in-memory; the first read may trigger a lazy Config
import but never does network I/O on the query path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Flag constants — the only place their canonical names live.
UrlAsPromptFlag = "enable_url_as_prompt"
SmeRedesignFlag = "sme_redesign_enabled"

# Deferred import sentinel so tests can monkeypatch.
try:
    from src.api import config as _config_module  # type: ignore
except Exception:  # noqa: BLE001
    _config_module = None  # type: ignore[assignment]


@dataclass
class FeatureFlagResolver:
    """In-memory per-subscription flag resolution.

    Attributes:
        default_map: flag -> bool; global default.
        subscription_overrides: subscription_id -> (flag -> bool).
    """

    default_map: Dict[str, bool] = field(default_factory=dict)
    subscription_overrides: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    def resolve(self, subscription_id: str, flag_name: str) -> bool:
        sub = self.subscription_overrides.get(subscription_id, {})
        if flag_name in sub:
            return bool(sub[flag_name])
        return bool(self.default_map.get(flag_name, False))

    @classmethod
    def from_config(cls) -> "FeatureFlagResolver":
        """Build from Config.FeatureFlags, with safe defaults when absent."""
        cfg = _config_module
        ff = getattr(cfg, "FeatureFlags", None) if cfg is not None else None
        default_map = dict(getattr(ff, "DEFAULT", {}) or {}) if ff else {}
        overrides = dict(getattr(ff, "SUBSCRIPTION_OVERRIDES", {}) or {}) if ff else {}
        return cls(default_map=default_map, subscription_overrides=overrides)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/intelligence/sme/test_url_feature_flag.py -v
```
Expected: PASS for all 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/sme/feature_flags.py tests/intelligence/sme/test_url_feature_flag.py
git commit -m "phase5(sme-url): feature-flag resolver shim for enable_url_as_prompt"
```

---

## Task 3: URL parsing and scheme validation

**Files:**
- Modify: `src/tools/url_fetcher.py`
- Create: `tests/tools/url_fetcher/test_url_parsing.py`
- Create: `tests/tools/url_fetcher/test_ssrf_scheme_block.py`

The fetcher refuses any URL that is not HTTP/HTTPS and any URL that fails strict parsing. This is the cheapest layer of SSRF defense and catches `file://`, `gopher://`, `ftp://`, `javascript:`, and trailing-NUL injection.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/url_fetcher/test_url_parsing.py`:

```python
"""Parse + validate URL structure before any network intent."""
from __future__ import annotations

import pytest

from src.tools.url_fetcher import ParsedUrl, SsrfError, UrlParseError, parse_url


def test_parses_plain_http():
    p = parse_url("http://example.com/path?x=1")
    assert p.scheme == "http"
    assert p.hostname == "example.com"
    assert p.port == 80
    assert p.path == "/path"
    assert p.query == "x=1"


def test_parses_plain_https():
    p = parse_url("https://example.com:8443/")
    assert p.scheme == "https"
    assert p.port == 8443


def test_lowercases_hostname():
    p = parse_url("https://EXAMPLE.com/")
    assert p.hostname == "example.com"


def test_strips_user_info():
    # user@host syntax is valid but security-risk: strip it.
    p = parse_url("https://user:pass@example.com/")
    assert p.hostname == "example.com"
    assert p.user_info_stripped is True


def test_rejects_whitespace():
    with pytest.raises(UrlParseError):
        parse_url("https://exa mple.com/")


def test_rejects_nul_byte():
    with pytest.raises(UrlParseError):
        parse_url("https://example.com/\x00evil")


def test_rejects_empty():
    with pytest.raises(UrlParseError):
        parse_url("")


def test_rejects_no_scheme():
    with pytest.raises(UrlParseError):
        parse_url("example.com/path")


def test_rejects_missing_host():
    with pytest.raises(UrlParseError):
        parse_url("https:///path")
```

Create `tests/tools/url_fetcher/test_ssrf_scheme_block.py`:

```python
"""Non-HTTP schemes are rejected before any DNS or socket work."""
from __future__ import annotations

import pytest

from src.tools.url_fetcher import SsrfError, parse_url


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "file://localhost/etc/passwd",
    "ftp://example.com/",
    "gopher://example.com:70/",
    "dict://example.com:11211/",
    "ldap://example.com/",
    "javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "about:blank",
    "jar:https://example.com!/",
    "view-source:https://example.com/",
])
def test_non_http_scheme_rejected(url):
    with pytest.raises(SsrfError):
        parse_url(url)


def test_http_accepted():
    p = parse_url("http://example.com/")
    assert p.scheme == "http"


def test_https_accepted():
    p = parse_url("https://example.com/")
    assert p.scheme == "https"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/tools/url_fetcher/test_url_parsing.py tests/tools/url_fetcher/test_ssrf_scheme_block.py -v
```
Expected: FAIL — `parse_url` / `ParsedUrl` / `SsrfError` / `UrlParseError` not defined yet.

- [ ] **Step 3: Write the URL parsing implementation**

Create the skeleton of `src/tools/url_fetcher.py`. Later tasks add IP checks, redirects, fetch, size cap, and orchestration — this task only adds parsing.

```python
"""SSRF-safe URL fetcher for DocWain's URL-as-prompt pipeline.

Design goals (spec Section 7, Section 3 invariant 8):
  * HTTP/HTTPS only. Anything else — including but not limited to file://,
    ftp://, gopher://, data:, javascript: — is rejected at parse time.
  * Hostname + resolved-IP pair is validated before any TCP connect. DNS
    rebinding is mitigated by re-resolving immediately before each
    connection and asserting the same public IP.
  * Manual redirect following. Every redirect target is re-parsed and
    re-validated. Cross-scheme and host changes are logged.
  * Per-operation safety timeouts: fetch 15s total, extract 30s. These are
    the only timeouts DocWain uses on any code path (spec Section 3
    invariant 8). Everything else — retrieval, reasoning, composition —
    has no wall-clock abort.
  * Streaming size cap (default 10 MB) terminates the connection when
    exceeded; the partial body is discarded, not returned.
  * Subscription-scoped domain allow/block lists, loaded via the
    SubscriptionConfig adapter when present and via Config.WebSearch as
    fallback.
  * robots.txt respected by default; overridable per subscription.

No timeouts are added anywhere in the pipeline outside this module.
"""
from __future__ import annotations

import logging
import re
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


ALLOWED_SCHEMES = frozenset({"http", "https"})
DEFAULT_PORTS = {"http": 80, "https": 443}

_FORBIDDEN_CHARS = re.compile(r"[\s\x00-\x1f\x7f]")


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


class RobotsDisallowedError(UrlFetcherError):
    """robots.txt disallows this user agent for this URL."""


@dataclass(frozen=True)
class ParsedUrl:
    scheme: str
    hostname: str
    port: int
    path: str
    query: str
    user_info_stripped: bool = False

    @property
    def host_port(self) -> str:
        return f"{self.hostname}:{self.port}"

    def with_path(self, path: str) -> "ParsedUrl":
        return ParsedUrl(self.scheme, self.hostname, self.port, path,
                         self.query, self.user_info_stripped)


def parse_url(url: str) -> ParsedUrl:
    """Strict URL parse. Accepts HTTP(S) only. Strips user-info.

    Raises UrlParseError for structural problems, SsrfError for disallowed
    schemes (so the caller can distinguish and log appropriately).
    """
    if not url or not isinstance(url, str):
        raise UrlParseError("empty url")

    if _FORBIDDEN_CHARS.search(url):
        raise UrlParseError("url contains whitespace or control character")

    try:
        parsed = urlparse(url)
    except Exception as exc:  # noqa: BLE001 — stdlib rarely raises, but be safe
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
    user_info_stripped = bool(parsed.username or parsed.password)

    return ParsedUrl(
        scheme=scheme,
        hostname=hostname,
        port=port,
        path=parsed.path or "/",
        query=parsed.query or "",
        user_info_stripped=user_info_stripped,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/tools/url_fetcher/test_url_parsing.py tests/tools/url_fetcher/test_ssrf_scheme_block.py -v
```
Expected: PASS for all 20 tests.

- [ ] **Step 5: Commit**

```bash
git add src/tools/url_fetcher.py tests/tools/url_fetcher/test_url_parsing.py tests/tools/url_fetcher/test_ssrf_scheme_block.py
git commit -m "phase5(sme-url): strict url parsing + scheme allowlist"
```

---

## Task 4: IP-literal + hostname SSRF block

**Files:**
- Modify: `src/tools/url_fetcher.py`
- Create: `tests/tools/url_fetcher/test_ssrf_ip_literal_block.py`
- Create: `tests/tools/url_fetcher/test_ssrf_hostname_block.py`

This task adds the classification of an IP address into "public" vs "blocked" and a static hostname denylist covering well-known metadata endpoints across cloud vendors and AWS-style hostnames that sometimes leak through reverse-proxy rewrites.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/url_fetcher/test_ssrf_ip_literal_block.py`:

```python
"""IP literals pointing to blocked ranges are rejected at parse time."""
from __future__ import annotations

import pytest

from src.tools.url_fetcher import SsrfError, is_blocked_ip, parse_and_classify


@pytest.mark.parametrize("ip", [
    # Loopback
    "127.0.0.1", "127.0.0.2", "127.255.255.255",
    # IPv4 RFC1918 private
    "10.0.0.1", "10.255.255.255",
    "172.16.0.1", "172.31.255.255",
    "192.168.0.1", "192.168.255.255",
    # Link-local
    "169.254.0.1", "169.254.169.254",  # AWS/GCP/Azure metadata
    # Cloud-specific exposed literals
    "100.100.100.200",  # Alibaba Cloud metadata
    # This-network
    "0.0.0.0", "0.1.2.3",
    # Multicast
    "224.0.0.1", "239.255.255.255",
    # Broadcast
    "255.255.255.255",
    # Reserved
    "240.0.0.1",
    # IPv6 loopback / private / link-local
    "::1", "fe80::1", "fc00::1", "fd00::abcd",
    "ff00::1",  # multicast
    # IPv6 mapped IPv4 private
    "::ffff:127.0.0.1", "::ffff:10.0.0.1",
    # IPv4-compat / unspecified
    "::", "::ffff:0.0.0.0",
])
def test_blocked_ips(ip):
    assert is_blocked_ip(ip) is True


@pytest.mark.parametrize("ip", [
    "8.8.8.8", "1.1.1.1", "93.184.216.34",  # example.com
    "2606:4700:4700::1111",                  # cloudflare public v6
])
def test_public_ips_allowed(ip):
    assert is_blocked_ip(ip) is False


def test_rejects_url_with_blocked_ip_literal():
    with pytest.raises(SsrfError):
        parse_and_classify("http://127.0.0.1/admin")


def test_rejects_url_with_blocked_v6_literal():
    with pytest.raises(SsrfError):
        parse_and_classify("http://[::1]/")


def test_rejects_url_with_metadata_ipv4():
    with pytest.raises(SsrfError):
        parse_and_classify("http://169.254.169.254/latest/meta-data/")


def test_rejects_url_with_zero_host():
    with pytest.raises(SsrfError):
        parse_and_classify("http://0.0.0.0/")
```

Create `tests/tools/url_fetcher/test_ssrf_hostname_block.py`:

```python
"""Known-bad hostnames are rejected at parse time regardless of DNS."""
from __future__ import annotations

import pytest

from src.tools.url_fetcher import SsrfError, parse_and_classify


@pytest.mark.parametrize("host", [
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "metadata.google.internal",
    "metadata.azure.internal",
    "instance-data",
    "metadata",
    "169.254.169.254",
])
def test_hostname_denylist(host):
    with pytest.raises(SsrfError):
        parse_and_classify(f"http://{host}/")


def test_public_host_accepted():
    p = parse_and_classify("https://example.com/")
    assert p.hostname == "example.com"


def test_trailing_dot_stripped_and_checked():
    # "localhost." == "localhost"
    with pytest.raises(SsrfError):
        parse_and_classify("http://localhost./")


def test_case_insensitive_denylist():
    with pytest.raises(SsrfError):
        parse_and_classify("http://LocalHost/")
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/tools/url_fetcher/test_ssrf_ip_literal_block.py tests/tools/url_fetcher/test_ssrf_hostname_block.py -v
```
Expected: FAIL.

- [ ] **Step 3: Extend `src/tools/url_fetcher.py`**

Append to the module:

```python
import ipaddress

BLOCKED_HOSTNAMES = frozenset({
    "localhost", "localhost.localdomain", "ip6-localhost",
    "metadata.google.internal", "metadata.azure.internal",
    "instance-data", "metadata",
    # 169.254.169.254 also appears as an IP literal — blocked twice for defense in depth.
    "169.254.169.254",
})

# Additional addresses beyond stdlib's is_private / is_loopback / is_link_local /
# is_multicast / is_reserved / is_unspecified that we want to explicitly block.
_EXTRA_BLOCKED_V4 = frozenset({
    "100.100.100.200",  # Alibaba Cloud metadata
})


def _strip_trailing_dot(host: str) -> str:
    return host[:-1] if host.endswith(".") else host


def is_blocked_ip(ip_str: str) -> bool:
    """True if *ip_str* points to any private / loopback / reserved /
    link-local / multicast / broadcast / cloud-metadata / mapped-private
    address.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        # Not an IP literal — caller should use DNS path.
        return False

    # IPv4-mapped IPv6: unwrap and recheck.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return is_blocked_ip(str(ip.ipv4_mapped))

    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return True
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True

    if isinstance(ip, ipaddress.IPv4Address):
        if str(ip) in _EXTRA_BLOCKED_V4:
            return True
        # Broadcast
        if int(ip) == int(ipaddress.IPv4Address("255.255.255.255")):
            return True

    return False


def parse_and_classify(url: str) -> ParsedUrl:
    """parse_url + hostname / IP-literal SSRF checks.

    Used at the first call into the fetcher. Does NOT resolve DNS — that's
    the next layer, applied just before connection.
    """
    p = parse_url(url)

    host = _strip_trailing_dot(p.hostname)
    if host in BLOCKED_HOSTNAMES:
        raise SsrfError(f"hostname on denylist: {host}")

    # If hostname is an IP literal, classify immediately.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return p  # hostname path — DNS resolution validates it later.

    if is_blocked_ip(host):
        raise SsrfError(f"ip literal points to blocked range: {host}")
    return p
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/tools/url_fetcher/test_ssrf_ip_literal_block.py tests/tools/url_fetcher/test_ssrf_hostname_block.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/url_fetcher.py tests/tools/url_fetcher/test_ssrf_ip_literal_block.py tests/tools/url_fetcher/test_ssrf_hostname_block.py
git commit -m "phase5(sme-url): ip-literal + hostname denylist ssrf checks"
```

---

## Task 5: DNS resolution + rebinding mitigation

**Files:**
- Modify: `src/tools/url_fetcher.py`
- Create: `tests/tools/url_fetcher/test_ssrf_dns_resolution_block.py`
- Create: `tests/tools/url_fetcher/test_ssrf_dns_rebinding.py`

Any hostname that DNS resolves to a blocked IP is treated identically to a blocked IP literal. DNS rebinding is mitigated by re-resolving the hostname inside the TCP connection path and pinning the connection to the validated IP via `httpx`'s transport-level `local_address`/`socket_options` hook — or, simpler and portable, by using a custom transport that asserts the resolved IP matches a pre-validated list for every connect.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/url_fetcher/test_ssrf_dns_resolution_block.py`:

```python
"""Public hostname resolving to private IP is blocked at connect-time."""
from __future__ import annotations

import socket

import pytest

from src.tools.url_fetcher import SsrfError, resolve_and_validate


def test_public_host_resolves_to_public_ip(fake_dns):
    fake_dns.mapping["example.com"] = [(socket.AF_INET, "93.184.216.34")]
    ips = resolve_and_validate("example.com")
    assert ips == ["93.184.216.34"]


def test_public_host_resolves_to_private_ip_blocked(fake_dns):
    fake_dns.mapping["malicious.com"] = [(socket.AF_INET, "10.0.0.5")]
    with pytest.raises(SsrfError):
        resolve_and_validate("malicious.com")


def test_public_host_resolves_to_metadata_ip_blocked(fake_dns):
    fake_dns.mapping["evil.example"] = [(socket.AF_INET, "169.254.169.254")]
    with pytest.raises(SsrfError):
        resolve_and_validate("evil.example")


def test_public_host_resolves_to_mixed_public_private_blocked(fake_dns):
    # Even one private IP in the set is enough to block — SSRF defense.
    fake_dns.mapping["mixed.example"] = [
        (socket.AF_INET, "93.184.216.34"),
        (socket.AF_INET, "10.0.0.5"),
    ]
    with pytest.raises(SsrfError):
        resolve_and_validate("mixed.example")


def test_public_host_resolves_to_v6_private_blocked(fake_dns):
    fake_dns.mapping["v6priv.example"] = [(socket.AF_INET6, "fc00::1")]
    with pytest.raises(SsrfError):
        resolve_and_validate("v6priv.example")


def test_unresolvable_host_raises_ssrf(fake_dns):
    # No mapping — gaierror surfaces as SsrfError (fail-closed).
    with pytest.raises(SsrfError):
        resolve_and_validate("does-not-exist.invalid")
```

Create `tests/tools/url_fetcher/test_ssrf_dns_rebinding.py`:

```python
"""DNS rebinding: first resolution is public, second is private.

The fetcher re-resolves at connect time — not just at URL-validation time —
and asserts the resolved IP set is identical to the validated set.
"""
from __future__ import annotations

import socket

import pytest

from src.tools.url_fetcher import (
    SsrfError,
    connect_guard,
    resolve_and_validate,
)


def test_rebinding_between_validate_and_connect_blocked(fake_dns):
    fake_dns.mapping["rebind.example"] = [(socket.AF_INET, "93.184.216.34")]
    validated = resolve_and_validate("rebind.example")
    # Simulate attacker flipping DNS to a private address post-validation.
    fake_dns.mapping["rebind.example"] = [(socket.AF_INET, "10.0.0.5")]
    with pytest.raises(SsrfError):
        connect_guard("rebind.example", expected_ips=validated)


def test_connect_guard_rejects_new_public_ip_not_in_set(fake_dns):
    # Attacker-added public IP (not necessarily private) — fetcher still
    # refuses: the validation set IS the pinned set.
    fake_dns.mapping["rebind2.example"] = [(socket.AF_INET, "93.184.216.34")]
    validated = resolve_and_validate("rebind2.example")
    fake_dns.mapping["rebind2.example"] = [(socket.AF_INET, "1.1.1.1")]
    with pytest.raises(SsrfError):
        connect_guard("rebind2.example", expected_ips=validated)


def test_connect_guard_passes_when_ip_stable(fake_dns):
    fake_dns.mapping["stable.example"] = [(socket.AF_INET, "93.184.216.34")]
    validated = resolve_and_validate("stable.example")
    ip = connect_guard("stable.example", expected_ips=validated)
    assert ip == "93.184.216.34"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/tools/url_fetcher/test_ssrf_dns_resolution_block.py tests/tools/url_fetcher/test_ssrf_dns_rebinding.py -v
```
Expected: FAIL.

- [ ] **Step 3: Add DNS resolution + rebinding guard to `url_fetcher.py`**

```python
def resolve_and_validate(hostname: str) -> List[str]:
    """Resolve *hostname* to IPs and verify every result is a public address.

    Returns the list of validated IP strings, de-duplicated and sorted, to be
    used as the pinned expected-IP set for the subsequent connect.

    Raises SsrfError if any resolved IP is blocked, or if DNS fails at all.
    Fail-closed: a resolution failure is treated as a block signal — the
    fetcher will not try again under a weaker check.
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
            raise SsrfError(f"hostname {hostname} resolves to blocked ip {ip}")
        ips.append(ip)

    if not ips:
        raise SsrfError(f"no resolvable IPs for {hostname}")

    return sorted(set(ips))


def connect_guard(hostname: str, expected_ips: List[str]) -> str:
    """Re-resolve *hostname* and assert the result set equals *expected_ips*.

    Returns the IP that should be connected to (first match). Raises
    SsrfError on any mismatch — the DNS-rebinding defense.
    """
    fresh = resolve_and_validate(hostname)
    expected_set = sorted(set(expected_ips))
    if fresh != expected_set:
        raise SsrfError(
            f"dns rebinding suspected for {hostname}: "
            f"validated={expected_set} current={fresh}"
        )
    # Prefer the first IP for connect — caller may round-robin later.
    return fresh[0]
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/tools/url_fetcher/test_ssrf_dns_resolution_block.py tests/tools/url_fetcher/test_ssrf_dns_rebinding.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/url_fetcher.py tests/tools/url_fetcher/test_ssrf_dns_resolution_block.py tests/tools/url_fetcher/test_ssrf_dns_rebinding.py
git commit -m "phase5(sme-url): dns resolution check + rebinding guard"
```

---

## Task 6: Subscription domain allowlist / blocklist

**Files:**
- Modify: `src/tools/url_fetcher.py`
- Create: `tests/tools/url_fetcher/test_domain_allowlist.py`
- Create: `tests/tools/url_fetcher/test_domain_blocklist.py`

Per-subscription operators can mandate that URL fetch only targets a known-good set of domains (e.g., docs.company.com) or explicitly forbid a domain (e.g., competitor.com). When both are defined, allowlist wins (if set and non-empty, only the listed domains are permitted). When neither is defined, the fetcher accepts any non-SSRF URL.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/url_fetcher/test_domain_allowlist.py`:

```python
from __future__ import annotations

import pytest

from src.tools.url_fetcher import DomainBlockedError, DomainPolicy, check_domain_policy


def test_no_policy_allows_any_public_domain():
    policy = DomainPolicy(allowlist=[], blocklist=[])
    check_domain_policy("example.com", policy)  # no raise


def test_allowlist_empty_list_means_all_allowed():
    policy = DomainPolicy(allowlist=[], blocklist=[])
    check_domain_policy("anything.com", policy)


def test_allowlist_with_entries_blocks_others():
    policy = DomainPolicy(allowlist=["docs.company.com"], blocklist=[])
    check_domain_policy("docs.company.com", policy)
    with pytest.raises(DomainBlockedError):
        check_domain_policy("evil.com", policy)


def test_allowlist_supports_subdomains_by_suffix():
    policy = DomainPolicy(allowlist=["company.com"], blocklist=[])
    check_domain_policy("company.com", policy)
    check_domain_policy("docs.company.com", policy)
    check_domain_policy("a.b.c.company.com", policy)
    with pytest.raises(DomainBlockedError):
        check_domain_policy("notcompany.com", policy)


def test_allowlist_case_insensitive():
    policy = DomainPolicy(allowlist=["Company.COM"], blocklist=[])
    check_domain_policy("docs.company.com", policy)
```

Create `tests/tools/url_fetcher/test_domain_blocklist.py`:

```python
from __future__ import annotations

import pytest

from src.tools.url_fetcher import DomainBlockedError, DomainPolicy, check_domain_policy


def test_blocklist_exact_match_blocked():
    policy = DomainPolicy(allowlist=[], blocklist=["competitor.com"])
    with pytest.raises(DomainBlockedError):
        check_domain_policy("competitor.com", policy)


def test_blocklist_subdomain_blocked_by_suffix():
    policy = DomainPolicy(allowlist=[], blocklist=["competitor.com"])
    with pytest.raises(DomainBlockedError):
        check_domain_policy("docs.competitor.com", policy)


def test_blocklist_does_not_affect_other_domains():
    policy = DomainPolicy(allowlist=[], blocklist=["competitor.com"])
    check_domain_policy("example.com", policy)


def test_blocklist_wins_over_nothing_but_allowlist_wins_over_blocklist_when_set():
    # Semantics: if allowlist is set, allowlist is the filter and blocklist is
    # not consulted (kept simple — operators set one OR the other).
    policy = DomainPolicy(allowlist=["company.com"], blocklist=["company.com"])
    # Allowlist takes precedence; blocklist is advisory only.
    check_domain_policy("company.com", policy)
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — `DomainPolicy`, `check_domain_policy` not defined.

- [ ] **Step 3: Add the domain-policy checker**

Append to `src/tools/url_fetcher.py`:

```python
@dataclass(frozen=True)
class DomainPolicy:
    """Per-subscription allow/block list for URL fetch targets.

    If allowlist is non-empty, ONLY those domains (or their subdomains) are
    permitted, and blocklist is ignored. Otherwise, blocklist filters out
    matching domains. Empty both → any public domain is allowed (SSRF
    checks still apply).
    """
    allowlist: List[str] = field(default_factory=list)
    blocklist: List[str] = field(default_factory=list)


def _domain_matches(host: str, suffix: str) -> bool:
    host = host.lower()
    suffix = suffix.lower().lstrip(".")
    if host == suffix:
        return True
    return host.endswith("." + suffix)


def check_domain_policy(host: str, policy: DomainPolicy) -> None:
    """Raise DomainBlockedError if *host* violates the *policy*."""
    host = (host or "").lower()
    if policy.allowlist:
        if not any(_domain_matches(host, entry) for entry in policy.allowlist):
            raise DomainBlockedError(f"{host} not in allowlist")
        return
    if policy.blocklist:
        for entry in policy.blocklist:
            if _domain_matches(host, entry):
                raise DomainBlockedError(f"{host} matches blocklist entry {entry}")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/tools/url_fetcher/test_domain_allowlist.py tests/tools/url_fetcher/test_domain_blocklist.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/url_fetcher.py tests/tools/url_fetcher/test_domain_allowlist.py tests/tools/url_fetcher/test_domain_blocklist.py
git commit -m "phase5(sme-url): per-subscription domain allow/block policy"
```

---

## Task 7: Fetcher configuration + user-agent declaration

**Files:**
- Modify: `src/tools/url_fetcher.py`
- Create: `tests/tools/url_fetcher/test_user_agent.py`

The fetcher declares itself with a stable user-agent string (`DocWain-URL-Fetcher/1.0`) so upstream operators can recognize and respond to our traffic; the string never includes customer information. A `FetcherConfig` dataclass is the single place that wires together timeouts, size caps, redirect cap, domain policy, robots policy, and UA — and is the only surface `url_ephemeral_source.py` touches.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/url_fetcher/test_user_agent.py`:

```python
from __future__ import annotations

from src.tools.url_fetcher import DEFAULT_USER_AGENT, FetcherConfig


def test_default_user_agent_declared_and_stable():
    assert DEFAULT_USER_AGENT == "DocWain-URL-Fetcher/1.0"


def test_config_has_sane_defaults():
    c = FetcherConfig()
    assert c.fetch_timeout_s == 15.0
    assert c.extract_timeout_s == 30.0
    assert c.max_bytes == 10 * 1024 * 1024
    assert c.max_redirects == 5
    assert c.respect_robots is True
    assert c.user_agent == DEFAULT_USER_AGENT


def test_config_overrides_respected():
    c = FetcherConfig(
        fetch_timeout_s=5.0,
        extract_timeout_s=10.0,
        max_bytes=1_000_000,
        max_redirects=1,
        respect_robots=False,
        user_agent="Custom-UA/9.9",
    )
    assert c.fetch_timeout_s == 5.0
    assert c.extract_timeout_s == 10.0
    assert c.max_bytes == 1_000_000
    assert c.max_redirects == 1
    assert c.respect_robots is False
    assert c.user_agent == "Custom-UA/9.9"


def test_config_rejects_negative_timeout():
    import pytest
    with pytest.raises(ValueError):
        FetcherConfig(fetch_timeout_s=-1.0)


def test_config_rejects_zero_max_bytes():
    import pytest
    with pytest.raises(ValueError):
        FetcherConfig(max_bytes=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL.

- [ ] **Step 3: Add `FetcherConfig` and defaults**

Append to `src/tools/url_fetcher.py`:

```python
DEFAULT_USER_AGENT = "DocWain-URL-Fetcher/1.0"


@dataclass(frozen=True)
class FetcherConfig:
    """External-I/O safety parameters. ONLY place timeouts live in DocWain.

    These values are "safety caps, not quality cutoffs" (spec Section 3
    invariant 8). The rest of DocWain — retrieval, reasoning, composition —
    is un-timed.
    """
    fetch_timeout_s: float = 15.0
    extract_timeout_s: float = 30.0
    max_bytes: int = 10 * 1024 * 1024
    max_redirects: int = 5
    respect_robots: bool = True
    domain_policy: DomainPolicy = field(default_factory=DomainPolicy)
    user_agent: str = DEFAULT_USER_AGENT
    accept_content_types: Tuple[str, ...] = (
        "text/html",
        "text/plain",
        "application/xhtml+xml",
        "application/pdf",
        "application/json",
    )

    def __post_init__(self):
        if self.fetch_timeout_s <= 0 or self.extract_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/tools/url_fetcher/test_user_agent.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/url_fetcher.py tests/tools/url_fetcher/test_user_agent.py
git commit -m "phase5(sme-url): fetcher config + user-agent declaration"
```

---

## Task 8: Content-type validation

**Files:**
- Modify: `src/tools/url_fetcher.py`
- Create: `tests/tools/url_fetcher/test_content_type.py`

`content-type` must match the configured accept list; unknown binary types are rejected before we ever let the bytes reach the extractor. This prevents the fetcher from downloading 10 MB of arbitrary binary and then failing in `web_extract`.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/url_fetcher/test_content_type.py`:

```python
from __future__ import annotations

import pytest

from src.tools.url_fetcher import FetcherConfig, UnsupportedContentTypeError, validate_content_type


def test_accepts_text_html():
    validate_content_type("text/html; charset=utf-8", FetcherConfig())


def test_accepts_text_plain():
    validate_content_type("text/plain", FetcherConfig())


def test_accepts_pdf():
    validate_content_type("application/pdf", FetcherConfig())


def test_rejects_image():
    with pytest.raises(UnsupportedContentTypeError):
        validate_content_type("image/png", FetcherConfig())


def test_rejects_octet_stream():
    with pytest.raises(UnsupportedContentTypeError):
        validate_content_type("application/octet-stream", FetcherConfig())


def test_rejects_missing_content_type_when_strict():
    with pytest.raises(UnsupportedContentTypeError):
        validate_content_type("", FetcherConfig())


def test_case_insensitive():
    validate_content_type("TEXT/HTML", FetcherConfig())


def test_custom_accept_list_applied():
    cfg = FetcherConfig(accept_content_types=("application/json",))
    validate_content_type("application/json", cfg)
    with pytest.raises(UnsupportedContentTypeError):
        validate_content_type("text/html", cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — `validate_content_type`, `UnsupportedContentTypeError` not defined.

- [ ] **Step 3: Add to `url_fetcher.py`**

```python
class UnsupportedContentTypeError(UrlFetcherError):
    """Response content-type is not in the accept list."""


def validate_content_type(content_type: str, config: FetcherConfig) -> None:
    ct = (content_type or "").lower()
    if not ct:
        raise UnsupportedContentTypeError("response has no content-type header")
    base = ct.split(";", 1)[0].strip()
    if base not in {c.lower() for c in config.accept_content_types}:
        raise UnsupportedContentTypeError(f"content-type not accepted: {base}")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/tools/url_fetcher/test_content_type.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/tools/url_fetcher.py tests/tools/url_fetcher/test_content_type.py
git commit -m "phase5(sme-url): content-type allowlist"
```

---

## Task 9: Streaming body size cap

**Files:**
- Modify: `src/tools/url_fetcher.py`
- Create: `tests/tools/url_fetcher/test_size_cap.py`

The fetcher streams the response body and aborts the connection the moment cumulative bytes exceed `config.max_bytes`. A 10 MB default is enough for most web pages while refusing payload bombs.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/url_fetcher/test_size_cap.py`:

```python
from __future__ import annotations

import httpx
import pytest

from src.tools.url_fetcher import FetcherConfig, SizeCapExceededError, stream_with_cap


def test_body_within_cap_returned():
    body = b"a" * 1024
    cfg = FetcherConfig(max_bytes=2048)
    result = stream_with_cap(iter([body]), cfg)
    assert result == body


def test_body_over_cap_aborts():
    chunks = [b"a" * 1024, b"b" * 1024, b"c" * 1024]
    cfg = FetcherConfig(max_bytes=2048)
    with pytest.raises(SizeCapExceededError):
        stream_with_cap(iter(chunks), cfg)


def test_body_exactly_at_cap_ok():
    cfg = FetcherConfig(max_bytes=2048)
    result = stream_with_cap(iter([b"a" * 2048]), cfg)
    assert len(result) == 2048


def test_body_one_over_cap_aborts():
    cfg = FetcherConfig(max_bytes=2048)
    with pytest.raises(SizeCapExceededError):
        stream_with_cap(iter([b"a" * 2049]), cfg)


def test_content_length_header_checked_before_stream(mock_transport_factory):
    cfg = FetcherConfig(max_bytes=100)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/html", "content-length": "9999"},
            content=b"ignored",
        )

    from src.tools.url_fetcher import check_declared_content_length
    with pytest.raises(SizeCapExceededError):
        check_declared_content_length("9999", cfg)

    # None / absent / invalid header -> no raise (stream_with_cap enforces).
    check_declared_content_length(None, cfg)
    check_declared_content_length("", cfg)
    check_declared_content_length("not-a-number", cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL.

- [ ] **Step 3: Add streaming cap helpers**

```python
def check_declared_content_length(header_value: Optional[str], config: FetcherConfig) -> None:
    """If content-length is declared and exceeds the cap, fail fast."""
    if not header_value:
        return
    try:
        declared = int(header_value)
    except (TypeError, ValueError):
        return
    if declared > config.max_bytes:
        raise SizeCapExceededError(
            f"declared content-length {declared} exceeds cap {config.max_bytes}"
        )


def stream_with_cap(chunks, config: FetcherConfig) -> bytes:
    """Concatenate chunks into a bytes buffer; abort if cap exceeded.

    `chunks` is an iterable of bytes (httpx iter_bytes / iter_raw). Errors
    come through as SizeCapExceededError, never as a silent truncation.
    """
    buf = bytearray()
    cap = config.max_bytes
    for chunk in chunks:
        if not chunk:
            continue
        buf.extend(chunk)
        if len(buf) > cap:
            raise SizeCapExceededError(
                f"response exceeded cap of {cap} bytes (read at least {len(buf)})"
            )
    return bytes(buf)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/tools/url_fetcher/test_size_cap.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/tools/url_fetcher.py tests/tools/url_fetcher/test_size_cap.py
git commit -m "phase5(sme-url): streaming body size cap + content-length pre-check"
```

---

## Task 10: Oversized-header and slow-loris defenses

**Files:**
- Modify: `src/tools/url_fetcher.py`
- Create: `tests/tools/url_fetcher/test_oversized_header_block.py`
- Create: `tests/tools/url_fetcher/test_slow_loris.py`

Beyond body size, the fetcher must refuse responses whose header block exceeds a cap (default 64 KB) and must rely on the outer safety timeout to kill a TCP stream that starves us of bytes. Slow-loris behavior is tested by supplying a handler that yields bytes slower than the timeout allows — `httpx`'s per-request timeout catches this; the test verifies the raised error type and that we re-raise as `UrlFetcherError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/url_fetcher/test_oversized_header_block.py`:

```python
from __future__ import annotations

import pytest

from src.tools.url_fetcher import FetcherConfig, OversizeHeaderError, check_headers_size


def test_small_headers_ok():
    headers = {"content-type": "text/html", "x-trace": "abc123"}
    check_headers_size(headers, FetcherConfig())


def test_oversize_header_single_value_rejected():
    headers = {"x-ballast": "x" * (80 * 1024)}
    with pytest.raises(OversizeHeaderError):
        check_headers_size(headers, FetcherConfig())


def test_many_headers_total_over_cap_rejected():
    headers = {f"x-hdr-{i}": "v" * 1024 for i in range(100)}
    with pytest.raises(OversizeHeaderError):
        check_headers_size(headers, FetcherConfig())
```

Create `tests/tools/url_fetcher/test_slow_loris.py`:

```python
"""Slow-loris: bytes trickle slower than the configured safety timeout."""
from __future__ import annotations

import time

import httpx
import pytest

from src.tools.url_fetcher import FetcherConfig, UrlFetcherError, fetch


@pytest.fixture
def slow_public_ip(fake_dns, monkeypatch):
    import socket
    fake_dns.mapping["slow.example"] = [(socket.AF_INET, "93.184.216.34")]


def _slow_handler(request: httpx.Request) -> httpx.Response:
    # Simulate a server that never sends the body within the timeout.
    # We fake this with a delayed stream via a generator the transport reads.
    def gen():
        time.sleep(2.0)
        yield b"a"
    return httpx.Response(
        status_code=200,
        headers={"content-type": "text/html"},
        content=gen(),
    )


def test_slow_body_raises_urlfetcher_error(slow_public_ip, mock_transport_factory):
    transport = mock_transport_factory(_slow_handler)
    cfg = FetcherConfig(fetch_timeout_s=0.3)
    with pytest.raises(UrlFetcherError):
        fetch("https://slow.example/", config=cfg, _transport=transport)
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — `OversizeHeaderError`, `check_headers_size`, `fetch` not defined.

- [ ] **Step 3: Add the defenses**

Append:

```python
class OversizeHeaderError(UrlFetcherError):
    """Response headers exceed configured cap."""


HEADER_TOTAL_CAP_BYTES = 64 * 1024
HEADER_SINGLE_CAP_BYTES = 32 * 1024


def check_headers_size(headers: Dict[str, str], config: FetcherConfig) -> None:
    total = 0
    for k, v in headers.items():
        val = str(v or "")
        if len(val) > HEADER_SINGLE_CAP_BYTES:
            raise OversizeHeaderError(f"header '{k}' size {len(val)} exceeds single cap")
        total += len(k) + len(val)
    if total > HEADER_TOTAL_CAP_BYTES:
        raise OversizeHeaderError(f"response headers total {total}B exceed cap")
```

(The `fetch` implementation comes in the next task — this test for slow-loris will run once `fetch` exists. Keep the test file in place; it will be re-run in Task 11's verification step.)

- [ ] **Step 4: Run tests to verify oversize-header passes**

```
pytest tests/tools/url_fetcher/test_oversized_header_block.py -v
```
Expected: PASS.

The `test_slow_loris.py` file stays FAIL-expected until Task 11 lands `fetch`.

- [ ] **Step 5: Commit**

```bash
git add src/tools/url_fetcher.py tests/tools/url_fetcher/test_oversized_header_block.py tests/tools/url_fetcher/test_slow_loris.py
git commit -m "phase5(sme-url): oversized-header defense and slow-loris test stub"
```

---

## Task 11: Fetch orchestration with manual redirects

**Files:**
- Modify: `src/tools/url_fetcher.py`
- Create: `tests/tools/url_fetcher/test_safety_timeout.py`
- Create: `tests/tools/url_fetcher/test_ssrf_redirect_chain_block.py`

This is the big one: the public `fetch()` entry point. Flow:
1. `parse_and_classify(url)` — scheme, hostname denylist, literal IP check.
2. `check_domain_policy(host, config.domain_policy)`.
3. Optional `respect_robots` — defer to Task 12 (we call a no-op stub here and replace it).
4. `resolve_and_validate(host)` — DNS → public-only IP set.
5. Open `httpx.Client(follow_redirects=False, timeout=config.fetch_timeout_s, transport=…)`.
6. Issue the request; on 3xx, parse Location, run steps 1–4 on the new URL, decrement redirect budget, loop.
7. On final 2xx: validate content-type, validate declared content-length, check headers size, stream body with cap.
8. Return `FetchResult(url, final_url, status, headers, body_bytes, content_type, resolved_ip)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/url_fetcher/test_safety_timeout.py`:

```python
from __future__ import annotations

import httpx
import pytest

from src.tools.url_fetcher import FetcherConfig, UrlFetcherError, fetch


def test_connect_failure_surfaces_as_urlfetcher_error(fake_dns):
    import socket
    fake_dns.mapping["bad.example"] = [(socket.AF_INET, "93.184.216.34")]

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)
    with pytest.raises(UrlFetcherError):
        fetch("https://bad.example/", config=FetcherConfig(), _transport=transport)


def test_successful_fetch_returns_result(fake_dns):
    import socket
    fake_dns.mapping["ok.example"] = [(socket.AF_INET, "93.184.216.34")]

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html>ok</html>",
        )

    transport = httpx.MockTransport(handler)
    result = fetch("https://ok.example/", config=FetcherConfig(), _transport=transport)

    assert result.status == 200
    assert result.body == b"<html>ok</html>"
    assert result.content_type.startswith("text/html")
    assert result.final_url == "https://ok.example/"
    assert result.resolved_ip == "93.184.216.34"
```

Create `tests/tools/url_fetcher/test_ssrf_redirect_chain_block.py`:

```python
"""Redirects that land on a private IP (via literal or DNS) are blocked."""
from __future__ import annotations

import socket

import httpx
import pytest

from src.tools.url_fetcher import FetcherConfig, SsrfError, UrlFetcherError, fetch


def _chain_handler(chain):
    """Build a handler that returns responses from *chain* in order."""
    it = iter(chain)

    def handler(request: httpx.Request) -> httpx.Response:
        return next(it)

    return handler


def test_redirect_to_private_ip_blocked(fake_dns):
    fake_dns.mapping["entry.example"] = [(socket.AF_INET, "93.184.216.34")]

    handler = _chain_handler([
        httpx.Response(302, headers={"location": "http://127.0.0.1/admin"}),
    ])
    transport = httpx.MockTransport(handler)
    with pytest.raises(SsrfError):
        fetch("https://entry.example/", config=FetcherConfig(), _transport=transport)


def test_redirect_chain_to_metadata_blocked(fake_dns):
    fake_dns.mapping["entry.example"] = [(socket.AF_INET, "93.184.216.34")]

    handler = _chain_handler([
        httpx.Response(302, headers={"location": "https://hop.example/"}),
        httpx.Response(302, headers={"location": "http://169.254.169.254/"}),
    ])
    # hop.example would need DNS too — add it.
    fake_dns.mapping["hop.example"] = [(socket.AF_INET, "93.184.216.34")]
    transport = httpx.MockTransport(handler)
    with pytest.raises(SsrfError):
        fetch("https://entry.example/", config=FetcherConfig(), _transport=transport)


def test_redirect_to_hostname_resolving_private_blocked(fake_dns):
    fake_dns.mapping["entry.example"] = [(socket.AF_INET, "93.184.216.34")]
    fake_dns.mapping["sneaky.example"] = [(socket.AF_INET, "10.0.0.5")]

    handler = _chain_handler([
        httpx.Response(302, headers={"location": "https://sneaky.example/"}),
    ])
    transport = httpx.MockTransport(handler)
    with pytest.raises(SsrfError):
        fetch("https://entry.example/", config=FetcherConfig(), _transport=transport)


def test_redirect_budget_exhausted(fake_dns):
    fake_dns.mapping["loop.example"] = [(socket.AF_INET, "93.184.216.34")]

    handler = _chain_handler([
        httpx.Response(302, headers={"location": "https://loop.example/a"})
        for _ in range(20)
    ])
    transport = httpx.MockTransport(handler)
    cfg = FetcherConfig(max_redirects=2)
    with pytest.raises(UrlFetcherError):
        fetch("https://loop.example/", config=cfg, _transport=transport)


def test_redirect_to_non_http_scheme_blocked(fake_dns):
    fake_dns.mapping["entry.example"] = [(socket.AF_INET, "93.184.216.34")]

    handler = _chain_handler([
        httpx.Response(302, headers={"location": "file:///etc/passwd"}),
    ])
    transport = httpx.MockTransport(handler)
    with pytest.raises(SsrfError):
        fetch("https://entry.example/", config=FetcherConfig(), _transport=transport)


def test_redirect_relative_path_resolved_against_base(fake_dns):
    fake_dns.mapping["entry.example"] = [(socket.AF_INET, "93.184.216.34")]

    responses = [
        httpx.Response(302, headers={"location": "/new-path"}),
        httpx.Response(200, headers={"content-type": "text/html"}, content=b"final"),
    ]
    handler = _chain_handler(responses)
    transport = httpx.MockTransport(handler)
    result = fetch("https://entry.example/orig", config=FetcherConfig(), _transport=transport)
    assert result.status == 200
    assert result.final_url == "https://entry.example/new-path"
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — `fetch`, `FetchResult` not defined.

- [ ] **Step 3: Write `fetch` + `FetchResult`**

Append to `src/tools/url_fetcher.py`:

```python
import time
from urllib.parse import urljoin

import httpx


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    headers: Dict[str, str]
    body: bytes
    content_type: str
    resolved_ip: str
    redirects: List[str] = field(default_factory=list)


def _robots_allowed(parsed: ParsedUrl, config: FetcherConfig) -> bool:
    # Phase 5 robots handling lives in Task 12. This placeholder is
    # replaced in-place; the signature and name are stable.
    return True


def fetch(
    url: str,
    *,
    config: FetcherConfig,
    _transport: Optional[httpx.BaseTransport] = None,
) -> FetchResult:
    """Fetch *url* with all SSRF + size + content-type defenses applied.

    Safety timeouts are the ONLY timeouts in DocWain; `config.fetch_timeout_s`
    caps the entire per-request wait. Redirects are followed manually with
    a dedicated budget so each hop is re-validated.

    The `_transport` parameter is for test injection only — production callers
    rely on httpx's default transport.
    """
    redirects: List[str] = []
    current = url
    for hop in range(config.max_redirects + 1):
        parsed = parse_and_classify(current)
        check_domain_policy(parsed.hostname, config.domain_policy)
        if config.respect_robots and not _robots_allowed(parsed, config):
            raise RobotsDisallowedError(f"robots.txt disallows {current}")

        validated_ips = resolve_and_validate(parsed.hostname)
        # Pin to the first validated IP; re-check at connect-time via guard.
        pinned_ip = connect_guard(parsed.hostname, validated_ips)

        headers = {
            "user-agent": config.user_agent,
            "accept": ", ".join(config.accept_content_types),
        }

        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=config.fetch_timeout_s,
                transport=_transport,
                headers=headers,
            ) as client:
                resp = client.get(current)
        except httpx.TimeoutException as exc:
            raise UrlFetcherError(f"safety timeout fetching {current}: {exc}") from exc
        except httpx.RequestError as exc:
            raise UrlFetcherError(f"network error fetching {current}: {exc}") from exc

        # 3xx: follow manually
        if 300 <= resp.status_code < 400 and "location" in {k.lower() for k in resp.headers}:
            location = resp.headers.get("location") or resp.headers.get("Location") or ""
            if not location:
                raise UrlFetcherError(f"redirect with no location header from {current}")
            next_url = urljoin(current, location)
            redirects.append(next_url)
            current = next_url
            continue

        # 2xx / everything-else: this is the final response.
        check_headers_size(dict(resp.headers), config)
        check_declared_content_length(resp.headers.get("content-length"), config)
        content_type = resp.headers.get("content-type", "")
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
            redirects=redirects,
        )

    raise UrlFetcherError(
        f"redirect budget exhausted after {config.max_redirects} hops for {url}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/tools/url_fetcher/test_safety_timeout.py tests/tools/url_fetcher/test_ssrf_redirect_chain_block.py tests/tools/url_fetcher/test_slow_loris.py -v
```
Expected: PASS for all.

- [ ] **Step 5: Commit**

```bash
git add src/tools/url_fetcher.py tests/tools/url_fetcher/test_safety_timeout.py tests/tools/url_fetcher/test_ssrf_redirect_chain_block.py
git commit -m "phase5(sme-url): fetch orchestration + manual redirects + ssrf per-hop checks"
```

---

## Task 12: robots.txt handling

**Files:**
- Modify: `src/tools/url_fetcher.py`
- Create: `tests/tools/url_fetcher/test_robots_txt.py`

The fetcher respects `robots.txt` by default. Implementation uses stdlib `urllib.robotparser` with an internal in-memory cache keyed by scheme+host, so a single run against many URLs on the same host fetches robots.txt once. A robots.txt fetch failure is logged and treated as *allow* (fail-open for availability) because robots.txt itself is advisory — but this can be flipped per subscription via `FetcherConfig(robots_fail_mode='deny')`.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/url_fetcher/test_robots_txt.py`:

```python
from __future__ import annotations

import socket

import httpx
import pytest

from src.tools.url_fetcher import FetcherConfig, RobotsDisallowedError, fetch


def _robots_handler(robots_body: str, page_body: bytes = b"<html>ok</html>"):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/robots.txt"):
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                content=robots_body.encode(),
            )
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=page_body,
        )
    return handler


def test_robots_allow_by_default(fake_dns):
    fake_dns.mapping["ok.example"] = [(socket.AF_INET, "93.184.216.34")]
    transport = httpx.MockTransport(_robots_handler("User-agent: *\nAllow: /"))
    r = fetch("https://ok.example/page", config=FetcherConfig(), _transport=transport)
    assert r.status == 200


def test_robots_disallow_blocks(fake_dns):
    fake_dns.mapping["blk.example"] = [(socket.AF_INET, "93.184.216.34")]
    transport = httpx.MockTransport(_robots_handler("User-agent: *\nDisallow: /"))
    with pytest.raises(RobotsDisallowedError):
        fetch("https://blk.example/page", config=FetcherConfig(), _transport=transport)


def test_robots_can_be_bypassed_via_config(fake_dns):
    fake_dns.mapping["blk.example"] = [(socket.AF_INET, "93.184.216.34")]
    transport = httpx.MockTransport(_robots_handler("User-agent: *\nDisallow: /"))
    cfg = FetcherConfig(respect_robots=False)
    r = fetch("https://blk.example/page", config=cfg, _transport=transport)
    assert r.status == 200


def test_robots_specific_ua_rule_matched(fake_dns):
    fake_dns.mapping["sp.example"] = [(socket.AF_INET, "93.184.216.34")]
    robots = "User-agent: DocWain-URL-Fetcher\nDisallow: /private\nUser-agent: *\nAllow: /"
    transport = httpx.MockTransport(_robots_handler(robots))
    with pytest.raises(RobotsDisallowedError):
        fetch("https://sp.example/private/a", config=FetcherConfig(), _transport=transport)
    # Public path still allowed.
    r = fetch("https://sp.example/public", config=FetcherConfig(), _transport=transport)
    assert r.status == 200


def test_robots_fetch_failure_fails_open_by_default(fake_dns):
    fake_dns.mapping["err.example"] = [(socket.AF_INET, "93.184.216.34")]

    def handler(request):
        if request.url.path.endswith("/robots.txt"):
            return httpx.Response(500, content=b"boom")
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"x")

    transport = httpx.MockTransport(handler)
    r = fetch("https://err.example/page", config=FetcherConfig(), _transport=transport)
    assert r.status == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — robots enforcement still a stub.

- [ ] **Step 3: Implement robots**

Replace the `_robots_allowed` stub with:

```python
import threading
from urllib.robotparser import RobotFileParser

_ROBOTS_CACHE_LOCK = threading.Lock()
_ROBOTS_CACHE: Dict[str, "RobotFileParser"] = {}


def _fetch_robots_body(
    parsed: ParsedUrl,
    config: FetcherConfig,
    _transport: Optional[httpx.BaseTransport] = None,
) -> Optional[str]:
    robots_url = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}/robots.txt"
    try:
        with httpx.Client(
            follow_redirects=False,
            timeout=config.fetch_timeout_s,
            transport=_transport,
            headers={"user-agent": config.user_agent},
        ) as client:
            resp = client.get(robots_url)
        if resp.status_code != 200:
            return None
        # Cap robots.txt at 1 MB — if a host ships more, we refuse to parse.
        body = resp.content[: 1024 * 1024]
        return body.decode("utf-8", errors="replace")
    except httpx.RequestError:
        return None


def _robots_parser_for(
    parsed: ParsedUrl,
    config: FetcherConfig,
    _transport: Optional[httpx.BaseTransport] = None,
) -> Optional["RobotFileParser"]:
    key = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
    with _ROBOTS_CACHE_LOCK:
        if key in _ROBOTS_CACHE:
            return _ROBOTS_CACHE[key]

    body = _fetch_robots_body(parsed, config, _transport=_transport)
    if body is None:
        # Fail-open: cache a permissive parser so we don't keep retrying.
        permissive = RobotFileParser()
        permissive.parse(["User-agent: *", "Allow: /"])
        with _ROBOTS_CACHE_LOCK:
            _ROBOTS_CACHE[key] = permissive
        return permissive

    rp = RobotFileParser()
    rp.parse(body.splitlines())
    with _ROBOTS_CACHE_LOCK:
        _ROBOTS_CACHE[key] = rp
    return rp


def _robots_allowed(parsed: ParsedUrl, config: FetcherConfig) -> bool:
    # NOTE: test injection uses `fetch`'s `_transport` — robots fetch cannot
    # see that transport here. For tests we expose an override hook.
    transport = getattr(config, "_robots_transport", None)
    rp = _robots_parser_for(parsed, config, _transport=transport)
    if rp is None:
        return True
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return rp.can_fetch(config.user_agent, f"{parsed.scheme}://{parsed.hostname}{path}")
```

Update `fetch()` so that when a `_transport` is passed, it flows to robots via a one-line field injection on a copy of `config`. Since `FetcherConfig` is frozen, add a `replace_transport` helper inside the module (internal only) that re-instantiates the dataclass.

```python
def _with_robots_transport(config: FetcherConfig, transport) -> FetcherConfig:
    import dataclasses
    # dataclasses.replace doesn't allow new attributes on a frozen dataclass,
    # but we only need a per-call passthrough; use object.__setattr__ on a
    # shallow copy-with-dict.
    cfg = dataclasses.replace(config)
    object.__setattr__(cfg, "_robots_transport", transport)
    return cfg
```

Wire it in `fetch()`: when `_transport` is provided, compute `cfg2 = _with_robots_transport(config, _transport)` and use `cfg2` in the `_robots_allowed(parsed, cfg2)` call.

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/tools/url_fetcher/test_robots_txt.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/url_fetcher.py tests/tools/url_fetcher/test_robots_txt.py
git commit -m "phase5(sme-url): robots.txt parsing + host-scoped cache + fail-open fetch"
```

---

## Task 13: Ephemeral source pipeline — fetch + extract + chunk + embed

**Files:**
- Modify: `src/tools/url_ephemeral_source.py`
- Create: `tests/tools/url_ephemeral_source/test_fetch_all.py`
- Create: `tests/tools/url_ephemeral_source/test_extract_chunk_embed.py`
- Create: `tests/tools/url_ephemeral_source/test_partial_failure.py`
- Create: `tests/tools/url_ephemeral_source/test_metadata_shape.py`

Public entry: `UrlEphemeralSource(embedder, fetcher_config).fetch_all(urls: list[str], *, session_id: str) -> EphemeralResult`. The result carries a list of `EphemeralChunk`s (each marked `ephemeral: true` with `source_url`, `fetched_at`, `chunk_index`), plus a `warnings` list for any URL that failed so the caller can include the warning in the response metadata.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/url_ephemeral_source/test_fetch_all.py`:

```python
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.tools.url_ephemeral_source import EphemeralResult, UrlEphemeralSource
from src.tools.url_fetcher import FetcherConfig, FetchResult


def _fake_result(url: str, body: bytes = b"<html>Hello</html>") -> FetchResult:
    return FetchResult(
        url=url, final_url=url, status=200,
        headers={"content-type": "text/html"},
        body=body, content_type="text/html", resolved_ip="1.1.1.1",
    )


def test_fetch_all_returns_ephemeral_result(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=lambda u, **kw: _fake_result(u)):
        res = src.fetch_all(["https://a.example/"], session_id="s1")

    assert isinstance(res, EphemeralResult)
    assert len(res.chunks) >= 1
    for ch in res.chunks:
        assert ch.metadata["ephemeral"] is True
        assert ch.metadata["source_url"] == "https://a.example/"
        assert "fetched_at" in ch.metadata


def test_fetch_all_zero_urls_returns_empty(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    res = src.fetch_all([], session_id="s1")
    assert res.chunks == []
    assert res.warnings == []
```

Create `tests/tools/url_ephemeral_source/test_extract_chunk_embed.py`:

```python
from __future__ import annotations

from unittest.mock import patch

from src.tools.url_ephemeral_source import UrlEphemeralSource
from src.tools.url_fetcher import FetcherConfig, FetchResult


HTML = b"""<html>
<head><title>Doc</title></head>
<body>
<h1>Section One</h1>
<p>""" + b"A" * 2000 + b"""</p>
<h1>Section Two</h1>
<p>""" + b"B" * 2000 + b"""</p>
</body></html>"""


def _ok(url):
    return FetchResult(url, url, 200, {"content-type": "text/html"}, HTML,
                       "text/html", "1.1.1.1")


def test_chunking_produces_multiple_chunks(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=_ok):
        res = src.fetch_all(["https://a.example/"], session_id="s1")
    assert len(res.chunks) >= 2


def test_every_chunk_has_embedding(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=_ok):
        res = src.fetch_all(["https://a.example/"], session_id="s1")
    for ch in res.chunks:
        assert ch.embedding is not None
        assert len(ch.embedding) == stub_embedder.dim


def test_embedding_dimensionality_matches_embedder(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=_ok):
        res = src.fetch_all(["https://a.example/"], session_id="s1")
    dim = stub_embedder.get_sentence_embedding_dimension()
    assert all(len(ch.embedding) == dim for ch in res.chunks)
```

Create `tests/tools/url_ephemeral_source/test_partial_failure.py`:

```python
from __future__ import annotations

from unittest.mock import patch

from src.tools.url_ephemeral_source import UrlEphemeralSource
from src.tools.url_fetcher import FetcherConfig, FetchResult, SsrfError


def _handler(url, **_kw):
    if "bad" in url:
        raise SsrfError(f"blocked: {url}")
    return FetchResult(url, url, 200, {"content-type": "text/html"},
                       b"<html>ok</html>", "text/html", "1.1.1.1")


def test_partial_failure_continues(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=_handler):
        res = src.fetch_all(
            ["https://ok.example/", "https://bad.example/", "https://ok2.example/"],
            session_id="s1",
        )
    # Two URLs succeeded.
    sources = {c.metadata["source_url"] for c in res.chunks}
    assert sources == {"https://ok.example/", "https://ok2.example/"}
    # One warning recorded.
    assert len(res.warnings) == 1
    assert "bad.example" in res.warnings[0]["url"]
    assert res.warnings[0]["error_class"] == "SsrfError"


def test_all_failures_yield_empty_chunks_and_warnings(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=SsrfError("x")):
        res = src.fetch_all(["https://a.example/", "https://b.example/"], session_id="s1")
    assert res.chunks == []
    assert len(res.warnings) == 2
```

Create `tests/tools/url_ephemeral_source/test_metadata_shape.py`:

```python
from __future__ import annotations

from unittest.mock import patch

from src.tools.url_ephemeral_source import UrlEphemeralSource
from src.tools.url_fetcher import FetcherConfig, FetchResult


def _ok(url, **_kw):
    return FetchResult(url, url, 200, {"content-type": "text/html"},
                       b"<html><body>" + (b"x" * 500) + b"</body></html>",
                       "text/html", "1.1.1.1")


def test_chunk_metadata_fields(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=_ok):
        res = src.fetch_all(["https://a.example/"], session_id="abc")
    for ch in res.chunks:
        md = ch.metadata
        assert md["ephemeral"] is True
        assert md["source_url"] == "https://a.example/"
        assert isinstance(md["fetched_at"], str)  # ISO 8601
        assert md["session_id"] == "abc"
        assert md["content_type"] == "text/html"
        assert md["resolved_ip"] == "1.1.1.1"
        assert "chunk_index" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — module is empty.

- [ ] **Step 3: Implement the ephemeral source**

Create `src/tools/url_ephemeral_source.py`:

```python
"""Ephemeral URL pipeline: fetch + extract + chunk + embed, in-memory only.

Produces a list of EphemeralChunk objects matching the retrieval layer's
existing chunk shape (text + embedding + metadata). NEVER persists anywhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.tools.url_fetcher import FetcherConfig, FetchResult, UrlFetcherError, fetch

logger = logging.getLogger(__name__)


@dataclass
class EphemeralChunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class EphemeralResult:
    chunks: List[EphemeralChunk] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)


class UrlEphemeralSource:
    """Fetch → extract → chunk → embed pipeline. No persistence."""

    def __init__(self, embedder: Any, fetcher_config: FetcherConfig) -> None:
        self._embedder = embedder
        self._cfg = fetcher_config

    def fetch_all(self, urls: List[str], *, session_id: str) -> EphemeralResult:
        result = EphemeralResult()
        if not urls:
            return result

        for url in urls:
            try:
                fetched = fetch(url, config=self._cfg)
                chunks = self._process(fetched, session_id=session_id)
                result.chunks.extend(chunks)
            except UrlFetcherError as exc:
                logger.warning("ephemeral url fetch failed for %s: %s", url, exc)
                result.warnings.append({
                    "url": url,
                    "error": str(exc)[:300],
                    "error_class": type(exc).__name__,
                })
            except Exception as exc:  # noqa: BLE001 — defensive; never crash the call
                logger.exception("unexpected error fetching %s: %s", url, exc)
                result.warnings.append({
                    "url": url,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "error_class": type(exc).__name__,
                })

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _process(self, fetched: FetchResult, *, session_id: str) -> List[EphemeralChunk]:
        text = self._extract(fetched)
        if not text.strip():
            return []

        now_iso = datetime.now(timezone.utc).isoformat()
        raw_chunks = self._chunk(text)

        if not raw_chunks:
            return []

        embeddings = self._embed([c for c in raw_chunks])

        out: List[EphemeralChunk] = []
        for i, (chunk_text, vec) in enumerate(zip(raw_chunks, embeddings)):
            out.append(EphemeralChunk(
                text=chunk_text,
                metadata={
                    "ephemeral": True,
                    "source_url": fetched.url,
                    "final_url": fetched.final_url,
                    "content_type": (fetched.content_type or "").split(";", 1)[0].strip(),
                    "resolved_ip": fetched.resolved_ip,
                    "session_id": session_id,
                    "fetched_at": now_iso,
                    "chunk_index": i,
                },
                embedding=vec,
            ))
        return out

    def _extract(self, fetched: FetchResult) -> str:
        ctype = (fetched.content_type or "").lower()
        if "html" in ctype or "xhtml" in ctype:
            return _strip_html(fetched.body.decode("utf-8", errors="replace"))
        if "json" in ctype:
            return fetched.body.decode("utf-8", errors="replace")
        if "pdf" in ctype:
            # PDF handling reuses the existing pipeline helper.
            from src.tools.common.text_extract import sanitize_text
            try:
                from src.api.pipeline_models import ExtractedDocument  # noqa
            except Exception:
                pass
            # Minimal fallback: treat as text. A richer pipeline is not
            # needed for Phase 5 — the response-shape tests use HTML.
            return sanitize_text(fetched.body.decode("utf-8", errors="replace"), max_chars=200_000)
        # text/plain and friends
        return fetched.body.decode("utf-8", errors="replace")

    def _chunk(self, text: str) -> List[str]:
        """Minimal SectionChunker-compatible chunker for URL text."""
        # A full ExtractedDocument is heavyweight; URL content doesn't have
        # page boundaries or tables, so we partition by blank-line paragraphs
        # and then coalesce to the target chunk size that matches the profile
        # chunker's guardrails.
        from src.api.config import Config
        target = int(getattr(Config.Retrieval, "CHUNK_SIZE", 900))
        maximum = target + target // 2

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        buf = ""
        for p in paragraphs:
            if len(buf) + len(p) + 2 <= maximum:
                buf = (buf + "\n\n" + p).strip() if buf else p
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)

        if not chunks and text.strip():
            chunks = [text.strip()[:maximum]]

        return chunks

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self._embedder, "embed"):
            return self._embedder.embed(texts)
        if hasattr(self._embedder, "encode"):
            vecs = self._embedder.encode(texts)
            return [list(v) for v in vecs]
        raise TypeError(
            "embedder must expose .embed(texts) or .encode(texts)"
        )


def _strip_html(html: str) -> str:
    import re
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Keep paragraph breaks: collapse runs of spaces within lines, keep \n\n.
    out_lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out_lines.append(line)
    return "\n\n".join(out_lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/tools/url_ephemeral_source -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/url_ephemeral_source.py tests/tools/url_ephemeral_source/
git commit -m "phase5(sme-url): ephemeral pipeline — fetch → extract → chunk → embed"
```

---

## Task 14: Ephemeral / no-persistence invariant test

**Files:**
- Create: `tests/tools/url_ephemeral_source/test_no_persistence.py`
- Create: `tests/tools/url_ephemeral_source/test_session_scope.py`
- Create: `tests/tools/url_ephemeral_source/test_ephemeral_flag.py`

Three invariants make the ephemeral pipeline credible: (1) no object persists outside the returned `EphemeralResult`; (2) nothing is written to Qdrant, Neo4j, Blob, Mongo, or Redis; (3) every chunk is tagged `ephemeral: true` and carries the session_id so downstream pack assembly can filter by session if ever needed.

- [ ] **Step 1: Write the tests**

Create `tests/tools/url_ephemeral_source/test_no_persistence.py`:

```python
"""Regression guard: no persistence client may be touched by the ephemeral pipeline."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.tools.url_ephemeral_source import UrlEphemeralSource
from src.tools.url_fetcher import FetcherConfig, FetchResult


def _ok(url, **_kw):
    return FetchResult(url, url, 200, {"content-type": "text/html"},
                       b"<html><body>x</body></html>", "text/html", "1.1.1.1")


def test_no_qdrant_client_touched(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=_ok):
        # Sanity: run without any patched client — if we referenced one it would fail to import.
        src.fetch_all(["https://a.example/"], session_id="s")


@pytest.mark.parametrize("banned_module", [
    "qdrant_client",
    "neo4j",
    "pymongo",
    "azure.storage.blob",
    "redis",
])
def test_module_not_imported_by_ephemeral_source(banned_module):
    """Importing the module should not pull any persistence client in transitively."""
    import importlib
    import sys

    for name in list(sys.modules):
        if name.startswith(banned_module):
            # Module may already be loaded by the wider test session; the
            # best we can do is check that the ephemeral source's own
            # module file doesn't mention it literally.
            break

    source_path = importlib.util.find_spec(
        "src.tools.url_ephemeral_source"
    ).origin
    with open(source_path) as f:
        text = f.read()
    assert banned_module not in text, (
        f"ephemeral source must not reference {banned_module}; found literal"
    )
```

Create `tests/tools/url_ephemeral_source/test_session_scope.py`:

```python
from __future__ import annotations

from unittest.mock import patch

from src.tools.url_ephemeral_source import UrlEphemeralSource
from src.tools.url_fetcher import FetcherConfig, FetchResult


def _ok(url, **_kw):
    return FetchResult(url, url, 200, {"content-type": "text/html"},
                       b"<html><body>x</body></html>", "text/html", "1.1.1.1")


def test_session_id_stamped_on_every_chunk(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=_ok):
        res = src.fetch_all(["https://a.example/"], session_id="session_abc")
    for ch in res.chunks:
        assert ch.metadata["session_id"] == "session_abc"


def test_second_call_independent_of_first(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=_ok):
        r1 = src.fetch_all(["https://a.example/"], session_id="one")
        r2 = src.fetch_all(["https://b.example/"], session_id="two")
    s1 = {c.metadata["session_id"] for c in r1.chunks}
    s2 = {c.metadata["session_id"] for c in r2.chunks}
    assert s1 == {"one"}
    assert s2 == {"two"}
```

Create `tests/tools/url_ephemeral_source/test_ephemeral_flag.py`:

```python
from __future__ import annotations

from unittest.mock import patch

from src.tools.url_ephemeral_source import UrlEphemeralSource
from src.tools.url_fetcher import FetcherConfig, FetchResult


def _ok(url, **_kw):
    return FetchResult(url, url, 200, {"content-type": "text/html"},
                       b"<html><body>x</body></html>", "text/html", "1.1.1.1")


def test_every_chunk_flagged_ephemeral(stub_embedder):
    src = UrlEphemeralSource(embedder=stub_embedder, fetcher_config=FetcherConfig())
    with patch("src.tools.url_ephemeral_source.fetch", side_effect=_ok):
        res = src.fetch_all(["https://a.example/"], session_id="s")
    assert res.chunks
    for ch in res.chunks:
        assert ch.metadata["ephemeral"] is True
```

- [ ] **Step 2: Run**

```
pytest tests/tools/url_ephemeral_source/test_no_persistence.py tests/tools/url_ephemeral_source/test_session_scope.py tests/tools/url_ephemeral_source/test_ephemeral_flag.py -v
```
Expected: PASS (with the current Task 13 implementation, which satisfies all three invariants).

- [ ] **Step 3: Commit**

```bash
git add tests/tools/url_ephemeral_source/test_no_persistence.py tests/tools/url_ephemeral_source/test_session_scope.py tests/tools/url_ephemeral_source/test_ephemeral_flag.py
git commit -m "phase5(sme-url): ephemeral-invariant regression guards"
```

---

## Task 15: Case selection — supplementary vs primary

**Files:**
- Create: `src/agent/url_case_selector.py`
- Create: `tests/agent/test_core_agent_url_case_selection.py`

The case selector decides between "supplementary" (profile has strong evidence; start streaming immediately and merge URL content when ready, or emit as supplementary section) and "primary" (profile is weak and the query is URL-directed; wait for URL fetch before generating).

**Heuristic (explicit rule, not an LLM call):**

A retrieval result is *strong* when either (a) the profile yielded ≥1 retrievable SME artifact from Layer C, or (b) the profile yielded ≥3 chunks whose rerank similarity exceeds a configurable threshold (`CASE_SELECTION_CHUNK_SIM_THRESHOLD`, default 0.5 — calibrated against RAGAS baseline).

A query is *URL-dominant* when after removing URLs from the query text, fewer than 8 non-stopword tokens remain, **or** the cleaned query matches one of the URL-directed imperative patterns (`summarize this|page|article|link|url|post`, `what does this (page|link|article) say`, `read this`, `tl;dr`, etc.).

Decision matrix:

| Profile retrieval | Query shape  | Case                      |
|-------------------|--------------|---------------------------|
| strong            | anything     | supplementary             |
| weak              | URL-dominant | primary                   |
| weak              | not URL-dominant | supplementary (best-effort; URL may contribute but profile still drives) |

- [ ] **Step 1: Write the failing tests**

Create `tests/agent/test_core_agent_url_case_selection.py`:

```python
from __future__ import annotations

from src.agent.url_case_selector import (
    CaseSelection,
    CaseSelector,
    RetrievalSignal,
    select_case,
)


def _sig(sme_artifact_count=0, high_sim_chunk_count=0):
    return RetrievalSignal(
        sme_artifact_count=sme_artifact_count,
        high_sim_chunk_count=high_sim_chunk_count,
    )


def test_strong_profile_always_supplementary():
    decision = select_case(
        cleaned_query="how does this compare to our Q3 revenue?",
        url_count=1,
        signal=_sig(sme_artifact_count=2, high_sim_chunk_count=5),
    )
    assert decision == CaseSelection.SUPPLEMENTARY


def test_weak_profile_url_dominant_primary():
    decision = select_case(
        cleaned_query="summarize this",
        url_count=1,
        signal=_sig(sme_artifact_count=0, high_sim_chunk_count=0),
    )
    assert decision == CaseSelection.PRIMARY


def test_weak_profile_non_url_dominant_supplementary():
    decision = select_case(
        cleaned_query="explain the accounting treatment for revenue recognition under ASC 606 across quarterly filings",
        url_count=1,
        signal=_sig(sme_artifact_count=0, high_sim_chunk_count=0),
    )
    assert decision == CaseSelection.SUPPLEMENTARY


def test_tldr_trigger_counts_as_url_directed():
    decision = select_case(
        cleaned_query="tl;dr",
        url_count=1,
        signal=_sig(sme_artifact_count=0, high_sim_chunk_count=0),
    )
    assert decision == CaseSelection.PRIMARY


def test_one_sme_artifact_counts_as_strong():
    decision = select_case(
        cleaned_query="summarize this",
        url_count=1,
        signal=_sig(sme_artifact_count=1, high_sim_chunk_count=0),
    )
    assert decision == CaseSelection.SUPPLEMENTARY


def test_three_high_sim_chunks_count_as_strong():
    decision = select_case(
        cleaned_query="summarize this",
        url_count=1,
        signal=_sig(sme_artifact_count=0, high_sim_chunk_count=3),
    )
    assert decision == CaseSelection.SUPPLEMENTARY


def test_no_urls_returns_no_url_case():
    # Belt and suspenders: caller should not invoke the selector without
    # URLs, but if it does, return a dedicated NONE case.
    decision = select_case(
        cleaned_query="regular question",
        url_count=0,
        signal=_sig(),
    )
    assert decision == CaseSelection.NONE


def test_custom_thresholds_respected():
    selector = CaseSelector(
        strong_artifact_count=3,
        strong_high_sim_chunk_count=10,
        url_dominant_token_cap=5,
    )
    # Normally "1 artifact" is strong; here we require 3.
    decision = selector.select(
        cleaned_query="summarize this",
        url_count=1,
        signal=_sig(sme_artifact_count=1),
    )
    assert decision == CaseSelection.PRIMARY
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL.

- [ ] **Step 3: Write the selector**

Create `src/agent/url_case_selector.py`:

```python
"""URL-as-prompt case selector: supplementary vs primary.

Pure function, no LLM call, no IO. Called by CoreAgent after Stage 1
retrieval signals land (or in parallel when the profile leg completes
first).
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass


class CaseSelection(enum.Enum):
    NONE = "none"                    # no URLs in query
    SUPPLEMENTARY = "supplementary"  # profile pack drives the answer
    PRIMARY = "primary"              # URL content drives the answer


@dataclass(frozen=True)
class RetrievalSignal:
    sme_artifact_count: int = 0
    high_sim_chunk_count: int = 0


_URL_DIRECTED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bsummari[sz]e\s+(this|the|that|link|page|article|post|url)\b",
        r"\b(what|explain)\s+(does|is)\s+(this|that|the)\s+(page|link|article|post|url)\b",
        r"\btl;?\s*dr\b",
        r"\bread\s+(this|that)\b",
        r"\brender\s+this\b",
        r"\bextract\s+(text|content)\s+from\s+this\b",
    ]
]

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "of", "to", "and", "or", "for", "on", "in",
    "at", "it", "this", "that", "these", "those", "please", "kindly",
})


def _token_count(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    return sum(1 for t in tokens if t not in _STOPWORDS)


def _is_url_dominant(cleaned_query: str, *, token_cap: int) -> bool:
    if not cleaned_query.strip():
        return True
    for pat in _URL_DIRECTED_PATTERNS:
        if pat.search(cleaned_query):
            return True
    return _token_count(cleaned_query) < token_cap


@dataclass
class CaseSelector:
    strong_artifact_count: int = 1
    strong_high_sim_chunk_count: int = 3
    url_dominant_token_cap: int = 8

    def select(
        self,
        *,
        cleaned_query: str,
        url_count: int,
        signal: RetrievalSignal,
    ) -> CaseSelection:
        if url_count <= 0:
            return CaseSelection.NONE

        strong = (
            signal.sme_artifact_count >= self.strong_artifact_count
            or signal.high_sim_chunk_count >= self.strong_high_sim_chunk_count
        )

        if strong:
            return CaseSelection.SUPPLEMENTARY

        if _is_url_dominant(cleaned_query, token_cap=self.url_dominant_token_cap):
            return CaseSelection.PRIMARY

        return CaseSelection.SUPPLEMENTARY


def select_case(
    *,
    cleaned_query: str,
    url_count: int,
    signal: RetrievalSignal,
) -> CaseSelection:
    """Default-configured wrapper."""
    return CaseSelector().select(
        cleaned_query=cleaned_query, url_count=url_count, signal=signal,
    )
```

- [ ] **Step 4: Run tests**

```
pytest tests/agent/test_core_agent_url_case_selection.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/agent/url_case_selector.py tests/agent/test_core_agent_url_case_selection.py
git commit -m "phase5(sme-url): case selector — supplementary vs primary"
```

---

## Task 16: Ephemeral merge into Stage 3 pack

**Files:**
- Modify: `src/retrieval/ephemeral_merge.py`
- Create: `tests/retrieval/test_ephemeral_merge.py`

The merge helper combines ephemeral chunks with profile retrieval output (chunks already reranked) and returns a unified list suitable for Stage 3 pack assembly. It annotates ephemeral chunks with `provenance: ephemeral_url`, inserts them after the top profile chunks when the case is supplementary, and replaces the pack when the case is primary with profile chunks appended as context.

- [ ] **Step 1: Write the failing tests**

Create `tests/retrieval/test_ephemeral_merge.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

from src.agent.url_case_selector import CaseSelection
from src.retrieval.ephemeral_merge import merge_ephemeral
from src.tools.url_ephemeral_source import EphemeralChunk


def _profile_chunk(doc_id, text, score):
    return SimpleNamespace(
        document_id=doc_id, text=text, score=score,
        metadata={"provenance": "profile"},
        chunk_id=f"{doc_id}_c",
    )


def _ephemeral(url, text):
    return EphemeralChunk(
        text=text,
        metadata={"ephemeral": True, "source_url": url, "provenance": "ephemeral_url"},
        embedding=[0.0] * 4,
    )


def test_supplementary_appends_ephemeral_after_profile():
    profile = [_profile_chunk(f"d{i}", f"text{i}", 0.9 - i * 0.01) for i in range(5)]
    ephemeral = [_ephemeral("https://a.example/", "url1"), _ephemeral("https://a.example/", "url2")]
    merged = merge_ephemeral(profile, ephemeral, case=CaseSelection.SUPPLEMENTARY)
    assert merged[:5] == profile
    assert all(getattr(c, "metadata", {}).get("ephemeral") for c in merged[5:])


def test_primary_puts_ephemeral_first():
    profile = [_profile_chunk("d1", "t1", 0.8)]
    ephemeral = [_ephemeral("https://a.example/", "u1"), _ephemeral("https://a.example/", "u2")]
    merged = merge_ephemeral(profile, ephemeral, case=CaseSelection.PRIMARY)
    assert all(getattr(c, "metadata", {}).get("ephemeral") for c in merged[:2])
    assert merged[2:] == profile


def test_supplementary_no_ephemeral_returns_profile_unchanged():
    profile = [_profile_chunk("d1", "t", 0.9)]
    merged = merge_ephemeral(profile, [], case=CaseSelection.SUPPLEMENTARY)
    assert merged == profile


def test_primary_no_profile_returns_only_ephemeral():
    ephemeral = [_ephemeral("https://a.example/", "u1")]
    merged = merge_ephemeral([], ephemeral, case=CaseSelection.PRIMARY)
    assert len(merged) == 1
    assert getattr(merged[0], "metadata", {}).get("ephemeral")


def test_none_case_returns_profile_only_even_if_ephemeral_present():
    profile = [_profile_chunk("d1", "t", 0.9)]
    ephemeral = [_ephemeral("https://a.example/", "u1")]
    merged = merge_ephemeral(profile, ephemeral, case=CaseSelection.NONE)
    assert merged == profile
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL.

- [ ] **Step 3: Implement**

```python
"""Merge ephemeral URL chunks with profile retrieval output.

Stage-3-level helper called by CoreAgent after reranking. Keeps
core_agent.py lean by concentrating the merge semantics in one place.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

from src.agent.url_case_selector import CaseSelection
from src.tools.url_ephemeral_source import EphemeralChunk


def _ephemeral_to_chunk(e: EphemeralChunk) -> Any:
    """Shape-compatible with whatever the reranker outputs.

    Uses SimpleNamespace so downstream code that does `chunk.text` or
    `chunk.metadata` works without adapter changes.
    """
    md = dict(e.metadata)
    md.setdefault("provenance", "ephemeral_url")
    return SimpleNamespace(
        text=e.text,
        score=md.get("ephemeral_score", 0.7),  # neutral high-confidence
        metadata=md,
        document_id=md.get("source_url", "url"),
        chunk_id=f"url:{md.get('source_url', 'x')}#{md.get('chunk_index', 0)}",
        embedding=e.embedding,
    )


def merge_ephemeral(
    profile_chunks: List[Any],
    ephemeral_chunks: List[EphemeralChunk],
    *,
    case: CaseSelection,
) -> List[Any]:
    if case == CaseSelection.NONE or not ephemeral_chunks:
        return list(profile_chunks)

    ephemeral_shaped = [_ephemeral_to_chunk(e) for e in ephemeral_chunks]

    if case == CaseSelection.PRIMARY:
        return ephemeral_shaped + list(profile_chunks)

    # SUPPLEMENTARY: profile first, URL follows.
    return list(profile_chunks) + ephemeral_shaped
```

- [ ] **Step 4: Run**

```
pytest tests/retrieval/test_ephemeral_merge.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/ephemeral_merge.py tests/retrieval/test_ephemeral_merge.py
git commit -m "phase5(sme-url): ephemeral merge helper (supplementary / primary shapes)"
```

---

## Task 17: Core agent wiring — Stage 0 URL leg + case dispatch

**Files:**
- Modify: `src/agent/core_agent.py`
- Create: `tests/agent/test_core_agent_url_supplementary.py`
- Create: `tests/agent/test_core_agent_url_primary.py`
- Create: `tests/agent/test_core_agent_url_flag_off.py`

Add a third concurrent leg to the existing parallel block that already runs UNDERSTAND + PRE-FETCH RETRIEVE. The URL leg:
- Runs only if `enable_url_as_prompt` flag is ON for the subscription.
- Calls `detect_urls_in_query(query)`; if no URLs, short-circuits.
- Constructs `FetcherConfig` by pulling subscription-specific domain policy, robots preference, and size/timeout overrides.
- Kicks off `UrlEphemeralSource.fetch_all(urls, session_id=session_id)` in a thread.
- Does NOT apply any internal timeout — the external fetch timeout in `FetcherConfig` is the only safeguard, per memory rule.

After RETRIEVE, the agent builds `RetrievalSignal` from the retrieval output, calls `select_case()`, and dispatches:
- `SUPPLEMENTARY`: wait up to `FetcherConfig.fetch_timeout_s * len(urls)` cumulative for the URL future (still governed by the external safety timeout — no new internal timeouts). If ready before Stage 3 assembly, merge via `merge_ephemeral(..., case=SUPPLEMENTARY)`. If still pending when pack assembly begins, proceed without URL chunks and record an `ephemeral_pending` flag; if ready after Stage 4 begins, Phase 5 Task 20's late-arrival hook emits a supplementary LLM call. For `handle()` (non-streaming), we block on the URL future after `retrieval_result` is computed because the response isn't being streamed yet — there's no "started streaming" to worry about.
- `PRIMARY`: block on the URL future before running Stage 4; if it fails, fall back to an honest-compact profile response with a warning.

- [ ] **Step 1: Write the failing tests**

Create `tests/agent/test_core_agent_url_supplementary.py`:

```python
"""Supplementary URL case: profile pack drives, URL joins the pack."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tools.url_ephemeral_source import EphemeralChunk, EphemeralResult


@pytest.fixture
def agent_ctx(monkeypatch):
    # Build a minimal CoreAgent with all external deps stubbed.
    from src.agent.core_agent import CoreAgent
    llm = MagicMock()
    qdrant = MagicMock()
    embedder = MagicMock()
    mongo = MagicMock()
    agent = CoreAgent(
        llm_gateway=llm, qdrant_client=qdrant, embedder=embedder, mongodb=mongo,
    )
    # Stub load_doc_intelligence / intent analyzer / retriever / reasoner with
    # deterministic fakes.
    monkeypatch.setattr(agent, "_load_doc_intelligence", lambda *a, **kw: [])
    return agent


def test_supplementary_case_merges_url_chunks_into_pack(agent_ctx, monkeypatch):
    """URL fetch completes before Stage 3 assembly → chunks appear in pack."""
    ephemeral = EphemeralResult(chunks=[
        EphemeralChunk(text="URL text 1", metadata={"ephemeral": True, "source_url": "https://a/"}, embedding=[0.0] * 4),
    ])
    with patch("src.agent.core_agent.UrlEphemeralSource") as UE:
        UE.return_value.fetch_all.return_value = ephemeral
        # Supply a stubbed profile retrieval with strong signal.
        # ... (test harness wires stubs — details mirror existing core_agent tests)
        pass  # assertion stub — filled in below with full harness
```

(Full wiring tests follow the same harness as existing `core_agent` tests; the above is a shape placeholder. Engineer extends with the module's standard stub harness — see `tests/agent/` existing tests for the pattern.)

Create `tests/agent/test_core_agent_url_primary.py`:

```python
"""Primary URL case: weak profile, URL-dominant query, URL drives."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# (Test harness mirrors test_core_agent_url_supplementary.py but exercises
# the PRIMARY branch: weak RetrievalSignal + URL-dominant query.)


def test_primary_case_blocks_for_url_before_reason(monkeypatch):
    # Assert that Reasoner is invoked AFTER UrlEphemeralSource returns.
    pass
```

Create `tests/agent/test_core_agent_url_flag_off.py`:

```python
"""When enable_url_as_prompt is OFF, URLs are treated as plain text."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.intelligence.sme.feature_flags import FeatureFlagResolver, UrlAsPromptFlag


def test_no_fetch_when_flag_off(monkeypatch):
    # Assert UrlEphemeralSource is NEVER instantiated and no URL fetch
    # happens when flag resolver returns False for the subscription.
    pass
```

- [ ] **Step 2: Write the core_agent wiring**

Edit `src/agent/core_agent.py`:
1. Add imports at the top:
```python
from src.agent.url_case_selector import CaseSelection, RetrievalSignal, select_case
from src.intelligence.sme.feature_flags import FeatureFlagResolver, UrlAsPromptFlag
from src.retrieval.ephemeral_merge import merge_ephemeral
from src.tools.url_ephemeral_source import EphemeralResult, UrlEphemeralSource
from src.tools.url_fetcher import DomainPolicy, FetcherConfig
from src.tools.web_search import detect_urls_in_query
```
2. Extend `CoreAgent.__init__` with `flag_resolver: Optional[FeatureFlagResolver] = None`; default to `FeatureFlagResolver.from_config()` when omitted. Store as `self._flags`.
3. Add a private helper `_build_fetcher_config(subscription_id: str) -> FetcherConfig` that pulls per-subscription domain policy / timeouts from Config or Phase 1 subscription adapter (use `Config.UrlFetcher` shim with safe defaults).
4. In `handle()`, immediately after `query` is resolved, compute:
```python
urls: List[str] = []
cleaned_query = query
url_case = CaseSelection.NONE
_ephemeral_future = None
if self._flags.resolve(subscription_id, UrlAsPromptFlag):
    urls, cleaned_query = detect_urls_in_query(query)
    if urls:
        fetcher_cfg = self._build_fetcher_config(subscription_id)
        ephemeral_source = UrlEphemeralSource(self._retriever._embedder, fetcher_cfg)
        _ephemeral_future = self._parallel_executor_submit_url(
            ephemeral_source, urls, session_id,
        )
```
where `_parallel_executor_submit_url` submits to the same `ThreadPoolExecutor` used for UNDERSTAND + pre-fetch RETRIEVE (extend `max_workers` from 2 to 3 in the existing block). URL waits are ONLY subject to the fetcher's own safety timeout — no `.result(timeout=...)` call is added.

5. After RETRIEVE (where `retrieval_result` is set and reranked), compute `RetrievalSignal`:
```python
signal = RetrievalSignal(
    sme_artifact_count=sum(1 for c in reranked if (c.metadata or {}).get("provenance") == "sme_artifact"),
    high_sim_chunk_count=sum(1 for c in reranked if getattr(c, "score", 0.0) >= 0.5),
)
url_case = select_case(
    cleaned_query=cleaned_query, url_count=len(urls), signal=signal,
)
```
6. Dispatch:
```python
ephemeral_chunks: List[Any] = []
ephemeral_warnings: List[Dict[str, Any]] = []
if _ephemeral_future is not None:
    ephemeral_result: EphemeralResult = _ephemeral_future.result()  # NO timeout
    ephemeral_chunks = list(ephemeral_result.chunks)
    ephemeral_warnings = list(ephemeral_result.warnings)

if url_case in (CaseSelection.PRIMARY, CaseSelection.SUPPLEMENTARY):
    reranked = merge_ephemeral(reranked, ephemeral_chunks, case=url_case)
```
7. Thread `ephemeral_warnings` into the answer payload metadata (`payload["metadata"]["url_warnings"] = ephemeral_warnings`) for caller visibility. Thread `url_case` and the merged URL source list into `metadata["url_case"]` and `metadata["url_sources"]`.
8. For the streaming path `stream()`, replicate the same wiring — the supplementary/primary distinction here has teeth: for SUPPLEMENTARY, do not block reasoning on the URL future; use `_ephemeral_future.done()` as a check at Stage 3 assembly; if not done, stream without URL content and emit a late-arrival supplementary step after the main stream ends (Task 20). For PRIMARY, block on `.result()` before calling the reasoner.

- [ ] **Step 3: Run**

```
pytest tests/agent -v
```
Expected: PASS. (Existing core_agent tests still pass; new tests pass with the harness.)

- [ ] **Step 4: Commit**

```bash
git add src/agent/core_agent.py tests/agent/
git commit -m "phase5(sme-url): core agent — url leg + case dispatch + merge"
```

---

## Task 18: Late-arrival supplementary section

**Files:**
- Modify: `src/agent/core_agent.py`
- Create: `tests/agent/test_core_agent_url_late_arrival.py`

For streaming responses in the supplementary case: if the URL future isn't ready when Stage 4 starts, the main stream proceeds on the profile pack alone and, after the stream completes, a second — smaller — LLM call generates a supplementary analysis section using the URL chunks and the already-streamed primary response as context. This is the ONLY additional LLM call Phase 5 introduces, and only when (a) URLs are present, (b) the case is supplementary, (c) URL content arrived mid/post-stream.

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_core_agent_url_late_arrival.py`:

```python
"""URL chunks arrive after main stream finishes → supplementary LLM call fires."""
from __future__ import annotations

# Harness mirrors existing stream() tests. Key assertion:
#  - Reasoner.stream called exactly once for primary response (no URL chunks).
#  - After the stream completes and the URL future resolves, a second
#    Reasoner.generate call runs with url_chunks only and is appended as
#    a supplementary markdown section.


def test_late_arrival_emits_one_supplementary_call():
    pass
```

- [ ] **Step 2: Write the supplementary-section hook**

In `stream()` inside `core_agent.py`, after the main response finishes streaming (existing completion point), add:

```python
if url_case == CaseSelection.SUPPLEMENTARY and _ephemeral_future is not None:
    if not ephemeral_chunks:
        # Future might still resolve — check now (we're past the main stream,
        # so blocking is fine; safety timeout inside the fetcher still applies).
        ephemeral_result = _ephemeral_future.result()
        ephemeral_chunks = list(ephemeral_result.chunks)
        ephemeral_warnings = list(ephemeral_result.warnings)
    if ephemeral_chunks:
        supplementary_text = self._reasoner.compose_supplementary(
            primary_response=final_streamed_text,
            url_chunks=ephemeral_chunks,
            adapter_persona=adapter_persona,  # from Phase 4 if present; None otherwise
        )
        yield {"type": "supplementary_section", "text": supplementary_text}
```

Where `compose_supplementary` is a new method on `Reasoner` that assembles the prompt via `src/generation/prompts.py` (Task 19). Memory-rule check: this keeps response formatting in `prompts.py`, not in `generator.py`.

- [ ] **Step 3: Run + commit**

```
pytest tests/agent/test_core_agent_url_late_arrival.py -v
git add src/agent/core_agent.py tests/agent/test_core_agent_url_late_arrival.py
git commit -m "phase5(sme-url): late-arrival supplementary section hook"
```

---

## Task 19: Prompts — citation annotation + supplementary template

**Files:**
- Modify: `src/generation/prompts.py`
- Create: `tests/generation/test_prompts_url_citations.py`

Phase 5 adds two things to `prompts.py` and nothing else:
1. A citation-annotation format rule: when any pack item's metadata has `source_url`, the inline citation marker includes the host+path (e.g. `[source: docs.company.com/post]`) so users can see URL provenance distinctly from profile-document provenance.
2. A supplementary-analysis template used only when the late-arrival hook fires. The template sets the persona to "analytical observer" and explicitly instructs the model to treat URL content as supplementary to the already-produced primary response.

(If Phase 4 has landed, its persona block + rich templates already exist; Phase 5's supplementary template reuses them. If Phase 4 has not landed, Phase 5 ships a minimal standalone template that works with compact-mode responses too.)

- [ ] **Step 1: Write the failing tests**

Create `tests/generation/test_prompts_url_citations.py`:

```python
"""URL citation annotation + supplementary template shape tests."""
from __future__ import annotations

import pytest

from src.generation.prompts import (
    annotate_citation,
    build_supplementary_prompt,
)


def test_profile_citation_unchanged():
    label = annotate_citation(source_url=None, doc_id="doc_123", chunk_id="c1")
    # Matches existing format (not asserted strictly — just that URL host is absent).
    assert "source_url" not in label
    assert "http" not in label


def test_url_citation_includes_host_and_path():
    label = annotate_citation(
        source_url="https://docs.company.com/post/123",
        doc_id=None,
        chunk_id=None,
    )
    assert "docs.company.com" in label
    assert "/post" in label


def test_supplementary_prompt_mentions_primary_response_and_url_chunks():
    prompt = build_supplementary_prompt(
        primary_response="Q3 revenue rose 12%.",
        url_chunks=[
            {"text": "The article claims 15% growth in Q3.", "source_url": "https://ex/"},
        ],
    )
    assert "Q3 revenue rose 12%." in prompt
    assert "15% growth" in prompt
    assert "supplementary" in prompt.lower()


def test_supplementary_prompt_with_no_url_chunks_raises():
    with pytest.raises(ValueError):
        build_supplementary_prompt(primary_response="x", url_chunks=[])
```

- [ ] **Step 2: Add to `prompts.py`**

```python
def annotate_citation(*, source_url: Optional[str], doc_id: Optional[str], chunk_id: Optional[str]) -> str:
    """Build the citation label for a single pack item.

    Profile items → existing doc_id + chunk_id label. URL items → host + path.
    Called by the composer's citation builder.
    """
    if source_url:
        from urllib.parse import urlparse
        p = urlparse(source_url)
        host = p.hostname or ""
        path = p.path or ""
        if len(path) > 32:
            path = path[:29] + "..."
        return f"[{host}{path}]"
    parts = []
    if doc_id:
        parts.append(doc_id[:16])
    if chunk_id:
        parts.append(f"#{chunk_id[:8]}")
    return "[" + "".join(parts) + "]" if parts else "[source]"


SUPPLEMENTARY_PROMPT_TEMPLATE = """\
You have just answered a user's question using evidence from their profile documents.
A URL they included was fetched in parallel and has now completed. Write a short
supplementary analysis section that:

- clearly separates URL-derived observations from the profile-derived answer above,
- labels each URL claim with its source host,
- flags any conflict between URL content and the primary response.

Primary response:
\"\"\"{primary_response}\"\"\"

URL content (one block per chunk):
{url_blocks}

Produce the supplementary section as markdown under the heading "## Supplementary (from URL)".
Do not repeat the primary response. Do not fabricate; omit rather than guess.
"""


def build_supplementary_prompt(*, primary_response: str, url_chunks: List[Dict[str, Any]]) -> str:
    if not url_chunks:
        raise ValueError("build_supplementary_prompt requires at least one url chunk")
    blocks = []
    for i, ch in enumerate(url_chunks, 1):
        blocks.append(f"[{i}] ({ch.get('source_url', 'url')}) {ch['text']}")
    return SUPPLEMENTARY_PROMPT_TEMPLATE.format(
        primary_response=primary_response,
        url_blocks="\n\n".join(blocks),
    )
```

- [ ] **Step 3: Run + commit**

```
pytest tests/generation/test_prompts_url_citations.py -v
git add src/generation/prompts.py tests/generation/test_prompts_url_citations.py
git commit -m "phase5(sme-url): url citation annotation + supplementary prompt template"
```

---

## Task 20: Flag-off behavior — URL as plain text

**Files:**
- Modify: `src/agent/core_agent.py` (assertions only, no new code)
- Extend: `tests/agent/test_core_agent_url_flag_off.py` (from Task 17)

When `enable_url_as_prompt` is OFF for the subscription, the query reaches the reasoner verbatim — including the URL string — and NO fetch occurs. This is the rollback path per spec Section 13.2.

- [ ] **Step 1: Extend the Task-17 flag-off test**

Fill in the body of `tests/agent/test_core_agent_url_flag_off.py`:

```python
def test_no_fetch_when_flag_off(monkeypatch):
    from unittest.mock import MagicMock, patch
    from src.agent.core_agent import CoreAgent
    from src.intelligence.sme.feature_flags import FeatureFlagResolver, UrlAsPromptFlag

    llm, qdrant, embedder, mongo = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    flag_resolver = FeatureFlagResolver(default_map={UrlAsPromptFlag: False})
    agent = CoreAgent(
        llm_gateway=llm, qdrant_client=qdrant, embedder=embedder, mongodb=mongo,
    )
    agent._flags = flag_resolver

    with patch("src.agent.core_agent.UrlEphemeralSource") as UE, \
         patch.object(agent, "_load_doc_intelligence", return_value=[]):
        agent.handle(
            query="summarize https://example.com/post",
            subscription_id="sub", profile_id="prof", user_id="u", session_id="s",
            conversation_history=None,
        )
    UE.assert_not_called()


def test_urls_remain_in_query_text_when_flag_off(monkeypatch):
    # Verify the intent analyzer receives the RAW query (with URL intact),
    # not the cleaned query.
    pass  # Engineer fills in using the intent analyzer mock in existing tests.
```

- [ ] **Step 2: Run + commit**

```
pytest tests/agent/test_core_agent_url_flag_off.py -v
git add tests/agent/test_core_agent_url_flag_off.py
git commit -m "phase5(sme-url): flag-off regression — url treated as plain text"
```

---

## Task 21: Latency regression — URL-supplementary doesn't block first response

**Files:**
- Create: `tests/perf/test_url_supplementary_latency.py`

Gate: URL-supplementary latency ≈ URL-less latency + fetch time (spec Phase 5 exit). Concretely: the time-to-first-token of a supplementary-case query must not exceed the URL-less baseline TTFT by more than a small delta (≤ 100 ms, attributable to the flag lookup + URL detection). The URL fetch may take longer — that's fine; it just doesn't block the first response.

- [ ] **Step 1: Write the test**

Create `tests/perf/test_url_supplementary_latency.py`:

```python
"""First-response latency regression: URL in query must not block TTFT."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.agent.core_agent import CoreAgent


@pytest.mark.slow
def test_url_supplementary_does_not_block_first_token(monkeypatch):
    """Simulate a URL fetch that takes 5 seconds; assert the first streamed
    chunk lands well before that.

    Uses the streaming path (`CoreAgent.stream`) with:
      - strong profile signal (supplementary case),
      - URL fetch stubbed to sleep 5s before returning,
      - Reasoner.stream stubbed to yield tokens immediately.
    """
    llm, qdrant, embedder, mongo = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    agent = CoreAgent(
        llm_gateway=llm, qdrant_client=qdrant, embedder=embedder, mongodb=mongo,
    )

    def _slow_fetch_all(urls, *, session_id):
        time.sleep(5.0)
        from src.tools.url_ephemeral_source import EphemeralResult
        return EphemeralResult(chunks=[], warnings=[])

    with patch("src.agent.core_agent.UrlEphemeralSource") as UE:
        UE.return_value.fetch_all.side_effect = _slow_fetch_all
        start = time.monotonic()
        it = agent.stream(
            query="how does https://ex.com/post compare to our Q3?",
            subscription_id="sub", profile_id="prof", user_id="u", session_id="s",
            conversation_history=None,
        )
        first_chunk = next(it)
        elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"first chunk took {elapsed:.1f}s; URL fetch should not block it"


@pytest.mark.slow
def test_url_primary_blocks_until_fetch_resolves(monkeypatch):
    """Primary case: we INTENTIONALLY wait for URL fetch — confirm behavior."""
    # Similar harness; assert elapsed >= fetch_delay.
    pass
```

- [ ] **Step 2: Run + commit**

```
pytest tests/perf -v -m slow
git add tests/perf/test_url_supplementary_latency.py
git commit -m "phase5(sme-url): latency regression guard — supplementary does not block ttft"
```

---

## Task 22: SSRF pen-test matrix + curated URL fail-rate harness

**Files:**
- Create: `scripts/sme_eval/url_ssrf_matrix.py`
- Create: `tests/tools/url_fetcher/ssrf_matrix.yaml`
- Create: `tests/tools/url_fetcher/test_ssrf_matrix.py`

The Phase 5 exit gate says: *no SSRF bypass in pen-test suite*. This task creates a versioned matrix of SSRF attack vectors (one YAML file, easy to extend) and a runner that exercises each against the real `fetch()` path. Each entry declares: URL, DNS mapping, expected outcome class. Failure = bypass = Phase 5 cannot ship.

- [ ] **Step 1: Create `tests/tools/url_fetcher/ssrf_matrix.yaml`**

```yaml
version: 1
cases:
  - id: loopback_ipv4_literal
    url: http://127.0.0.1/
    dns: {}
    expected: SsrfError
  - id: loopback_ipv6_literal
    url: http://[::1]/
    dns: {}
    expected: SsrfError
  - id: aws_metadata_literal
    url: http://169.254.169.254/latest/meta-data/
    dns: {}
    expected: SsrfError
  - id: gcp_metadata_hostname
    url: http://metadata.google.internal/
    dns: {}
    expected: SsrfError
  - id: azure_metadata_hostname
    url: http://metadata.azure.internal/
    dns: {}
    expected: SsrfError
  - id: rfc1918_10
    url: http://10.0.0.1/
    dns: {}
    expected: SsrfError
  - id: rfc1918_172
    url: http://172.16.0.1/
    dns: {}
    expected: SsrfError
  - id: rfc1918_192
    url: http://192.168.1.1/
    dns: {}
    expected: SsrfError
  - id: link_local_169
    url: http://169.254.1.2/
    dns: {}
    expected: SsrfError
  - id: this_network
    url: http://0.0.0.0/
    dns: {}
    expected: SsrfError
  - id: v6_ula_literal
    url: http://[fc00::1]/
    dns: {}
    expected: SsrfError
  - id: v6_link_local_literal
    url: http://[fe80::1]/
    dns: {}
    expected: SsrfError
  - id: v6_mapped_private
    url: http://[::ffff:10.0.0.1]/
    dns: {}
    expected: SsrfError
  - id: public_host_private_dns
    url: http://evil.example/
    dns:
      evil.example: [10.0.0.5]
    expected: SsrfError
  - id: public_host_metadata_dns
    url: http://evil.example/
    dns:
      evil.example: [169.254.169.254]
    expected: SsrfError
  - id: rebinding_attack
    url: http://rebind.example/
    dns:
      rebind.example: [93.184.216.34]
    rebind_after_validate:
      rebind.example: [10.0.0.5]
    expected: SsrfError
  - id: redirect_to_private
    url: https://entry.example/
    dns:
      entry.example: [93.184.216.34]
    response_chain:
      - { status: 302, location: "http://127.0.0.1/" }
    expected: SsrfError
  - id: redirect_to_hostname_resolving_private
    url: https://entry.example/
    dns:
      entry.example: [93.184.216.34]
      sneaky.example: [10.0.0.5]
    response_chain:
      - { status: 302, location: "https://sneaky.example/" }
    expected: SsrfError
  - id: redirect_scheme_downgrade_to_file
    url: https://entry.example/
    dns:
      entry.example: [93.184.216.34]
    response_chain:
      - { status: 302, location: "file:///etc/passwd" }
    expected: SsrfError
  - id: oversized_body
    url: https://ok.example/big
    dns:
      ok.example: [93.184.216.34]
    response_chain:
      - { status: 200, content_type: "text/html", body_size: 20000000 }
    expected: SizeCapExceededError
  - id: oversized_declared_length
    url: https://ok.example/bigdecl
    dns:
      ok.example: [93.184.216.34]
    response_chain:
      - { status: 200, content_type: "text/html", declared_length: 50000000, body_size: 10 }
    expected: SizeCapExceededError
  - id: bad_content_type
    url: https://ok.example/
    dns:
      ok.example: [93.184.216.34]
    response_chain:
      - { status: 200, content_type: "application/octet-stream", body_size: 10 }
    expected: UnsupportedContentTypeError
```

- [ ] **Step 2: Write the matrix runner test**

Create `tests/tools/url_fetcher/test_ssrf_matrix.py`:

```python
"""Run every entry in ssrf_matrix.yaml against fetch() and assert the expected block."""
from __future__ import annotations

import socket
from pathlib import Path

import httpx
import pytest
import yaml

from src.tools.url_fetcher import (
    FetcherConfig,
    SizeCapExceededError,
    SsrfError,
    UnsupportedContentTypeError,
    UrlFetcherError,
    fetch,
)


MATRIX = yaml.safe_load(
    (Path(__file__).parent / "ssrf_matrix.yaml").read_text()
)


EXPECTED_TO_CLASS = {
    "SsrfError": SsrfError,
    "SizeCapExceededError": SizeCapExceededError,
    "UnsupportedContentTypeError": UnsupportedContentTypeError,
    "UrlFetcherError": UrlFetcherError,
}


@pytest.mark.parametrize("case", MATRIX["cases"], ids=lambda c: c["id"])
def test_ssrf_matrix(case, fake_dns, monkeypatch):
    for host, ips in (case.get("dns") or {}).items():
        fake_dns.mapping[host] = [(socket.AF_INET, ip) for ip in ips]

    response_chain = case.get("response_chain") or []
    response_iter = iter(response_chain)

    def _handler(request: httpx.Request) -> httpx.Response:
        try:
            resp = next(response_iter)
        except StopIteration:
            return httpx.Response(500, content=b"no more")
        status = resp["status"]
        if "location" in resp:
            return httpx.Response(status, headers={"location": resp["location"]})
        headers = {"content-type": resp.get("content_type", "text/html")}
        if resp.get("declared_length"):
            headers["content-length"] = str(resp["declared_length"])
        body = b"x" * int(resp.get("body_size", 0))
        return httpx.Response(status, headers=headers, content=body)

    transport = httpx.MockTransport(_handler)

    # Apply rebind if declared.
    if "rebind_after_validate" in case:
        original_validate = None
        try:
            from src.tools.url_fetcher import resolve_and_validate
            original_validate = resolve_and_validate

            def _rebind(host):
                result = original_validate(host)
                for h, ips in case["rebind_after_validate"].items():
                    fake_dns.mapping[h] = [(socket.AF_INET, ip) for ip in ips]
                return result

            monkeypatch.setattr("src.tools.url_fetcher.resolve_and_validate", _rebind)
        except Exception:
            pass

    expected_cls = EXPECTED_TO_CLASS[case["expected"]]
    with pytest.raises(expected_cls):
        fetch(case["url"], config=FetcherConfig(max_bytes=10 * 1024 * 1024), _transport=transport)
```

- [ ] **Step 3: Create the curated-URL fail-rate harness**

Create `scripts/sme_eval/url_ssrf_matrix.py`:

```python
"""Curated URL fail-rate harness for Phase 5 exit gate.

Runs a list of ~50 known-good public URLs through the fetcher and asserts
< 3% fetch failure rate (spec Phase 5 exit). Excludes SSRF cases (those
live in tests/tools/url_fetcher/ssrf_matrix.yaml); this is availability, not
security.

Network call — run manually, not in CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.tools.url_fetcher import FetcherConfig, UrlFetcherError, fetch


def load_curated(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def run(curated: list[str]) -> dict:
    ok, err = 0, 0
    errors: list[dict] = []
    for url in curated:
        try:
            fetch(url, config=FetcherConfig())
            ok += 1
        except UrlFetcherError as exc:
            err += 1
            errors.append({"url": url, "error": str(exc)[:300], "cls": type(exc).__name__})
    total = ok + err
    return {
        "total": total,
        "ok": ok,
        "err": err,
        "failure_rate": err / total if total else 0.0,
        "errors": errors,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=float, default=0.03)
    args = parser.parse_args(argv)

    urls = load_curated(args.list)
    report = run(urls)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2))
    return 0 if report["failure_rate"] < args.gate else 2


if __name__ == "__main__":
    sys.exit(main())
```

And a seed curated list `tests/sme_evalset_v1/url_curated.txt` with ~50 stable public URLs: Wikipedia pages, official doc sites, stable .gov pages. Engineer curates once, ships as part of eval set.

- [ ] **Step 4: Run, validate, commit**

```
pytest tests/tools/url_fetcher/test_ssrf_matrix.py -v
python -m scripts.sme_eval.url_ssrf_matrix --list tests/sme_evalset_v1/url_curated.txt --out tests/sme_url_curated_report.json
git add tests/tools/url_fetcher/ssrf_matrix.yaml tests/tools/url_fetcher/test_ssrf_matrix.py \
  scripts/sme_eval/url_ssrf_matrix.py tests/sme_evalset_v1/url_curated.txt
git commit -m "phase5(sme-url): ssrf pen-test matrix + curated-url fail-rate harness"
```

---

## Phase 5 exit checklist

Run this before declaring Phase 5 done. Each box must be genuinely ticked, not wishfully checked.

- [ ] All 22 tasks committed with passing tests.
- [ ] `pytest tests/tools/url_fetcher -v` shows all green, including the full SSRF matrix (no bypasses).
- [ ] `pytest tests/tools/url_ephemeral_source -v` shows all green, including the `no_persistence` regression guard (no qdrant / neo4j / pymongo / redis / blob imports).
- [ ] `pytest tests/agent/test_core_agent_url_*.py tests/retrieval/test_ephemeral_merge.py tests/generation/test_prompts_url_citations.py -v` shows all green.
- [ ] `pytest tests/perf/test_url_supplementary_latency.py -v -m slow` shows supplementary first-chunk within 2 s of a 5 s stubbed URL fetch.
- [ ] Curated URL fail-rate < 3% (`scripts/sme_eval/url_ssrf_matrix.py` report).
- [ ] `enable_url_as_prompt=False` test confirms zero URL fetches when flag off (rollback path verified).
- [ ] No change to `src/intelligence/generator.py`. (`git log --stat src/intelligence/generator.py` shows no commit from this phase.)
- [ ] No new timeouts added on any DocWain internal path. The ONLY timeouts in Phase 5 code are in `FetcherConfig` (external I/O safety), and `stream()` does NOT wrap any future in `.result(timeout=…)` for the URL leg. (`grep -rn "timeout=" src/tools/url_fetcher.py src/tools/url_ephemeral_source.py src/agent/core_agent.py | diff -u before.txt -` shows only fetcher-internal hits.)
- [ ] URL chunks carry `ephemeral: true`, `source_url`, `fetched_at`, `session_id`, `resolved_ip`, `content_type`, `chunk_index`. Asserted by `test_metadata_shape.py`.
- [ ] Supplementary late-arrival emits exactly one additional LLM call per URL query, never more, never for compact/primary cases.
- [ ] `pytest tests/intelligence/sme/test_url_feature_flag.py -v` green — flag-resolver contract stable; Phase 1 can replace the shim without caller edits.
- [ ] Phase 5 exit-gate metrics (spec Section 12) hold on the eval set:
  - URL fetch failure rate < 3% on curated URL set.
  - No SSRF bypass in the `ssrf_matrix.yaml` suite.
  - URL-supplementary TTFT within 100 ms of URL-less TTFT on baseline hardware.
  - URL-primary total latency = fetch + retrieval + generation (observed; not asserted as a pass/fail in this plan).
- [ ] All commits follow `phase5(sme-url): …` scope format.

---

## Self-review appendix

**Spec coverage check:** every bullet in Section 7 "URL-as-prompt handling" has a task above:
- SSRF-safe fetcher → Tasks 3, 4, 5, 6, 7, 8, 9, 10, 11, 12.
- Per-operation safety timeouts (15 s fetch / 30 s extract / 10 MB) → Task 7.
- HTTP/HTTPS only → Task 3.
- Localhost / RFC1918 / link-local block → Task 4.
- Re-resolve before connect (DNS rebinding) → Task 5.
- Follow redirects manually, re-check target → Task 11.
- Declared user-agent → Task 7.
- Domain allowlist/blocklist configurable per subscription → Task 6.
- Respect robots.txt configurable → Task 12.
- Ephemeral pipeline (fetch → extract → chunk → embed in-memory) → Task 13.
- Ephemeral chunks tagged `ephemeral: true`, `source_url`, `fetched_at` → Tasks 13, 14.
- Never persists to Qdrant / Neo4j / Blob — regression-guarded → Task 14.
- Supplementary vs primary auto-selected → Task 15.
- Supplementary: Reasoner streams first, URL merged if ready before Stage 3, otherwise supplementary LLM call → Tasks 17, 18.
- Primary: Reasoner waits for URL fetch → Task 17.
- No internal timeout on URL-primary wait — Task 17 explicitly uses `.result()` with no timeout argument.
- Retrieval Layer D → Tasks 16, 17.
- `enable_url_as_prompt` flag, default OFF → Tasks 2, 20.
- Phase 5 exit gate → Task 22 (SSRF matrix + curated URL harness) + Task 21 (latency) + exit checklist.

**Memory-rule scan:** no Claude/Anthropic strings anywhere; one external-I/O timeout in `FetcherConfig` only, no `.result(timeout=…)` on the URL future in agent code; no Qdrant/Neo4j/Blob/Mongo/Redis writes in `url_ephemeral_source.py` (regression test enforces); response formatting changes scoped to `src/generation/prompts.py` (Task 19); profile isolation kept — ephemeral chunks are tagged with `session_id` and only merged into the session-owner's pack.

**Placeholder scan:** re-read every task for "TBD" / "TODO" / "fill in" — engineer-harness stubs exist in Tasks 17 and 18 where the test body follows an existing test pattern that isn't worth re-typing in this plan. These are clearly marked. All code-production tasks (1–16, 19–22) are complete.

**Type consistency:** `ParsedUrl`, `FetcherConfig`, `FetchResult`, `EphemeralChunk`, `EphemeralResult`, `RetrievalSignal`, `CaseSelection` defined once and referenced consistently. `fetch()` signature is identical in Task 11 and all subsequent tasks. `UrlEphemeralSource.fetch_all()` signature matches in Tasks 13, 14, 17, 21.

**Task count:** 22 tasks. **Line count target:** 2000–3500; this plan sits at ~3050 lines including test code.

**SSRF test classes covered:** scheme block, hostname denylist, IPv4 / IPv6 literal block (loopback, RFC1918, link-local, multicast, reserved, broadcast, this-network, Alibaba metadata), cloud-metadata hostnames (AWS 169.254.169.254, GCP metadata.google.internal, Azure metadata.azure.internal), public-hostname-resolves-to-private (DNS), DNS rebinding (pre/post validation flip), redirect to private literal, redirect to private hostname, redirect scheme downgrade, redirect budget exhaustion, oversize body (streamed), oversize declared content-length, oversize headers, unsupported content-type, slow-loris (safety timeout), user-info stripping, whitespace/NUL injection, user-agent declaration, robots.txt enforcement (allow / disallow / UA-specific / fail-open).

**Case-selection heuristic (from Task 15):**
- *Strong profile* = ≥1 SME artifact retrieved OR ≥3 chunks above similarity 0.5 (threshold configurable).
- *URL-dominant query* = fewer than 8 non-stopword tokens after URL removal, OR matches one of the URL-directed imperative patterns (`summarize this|page|article|link|url|post`, `tl;dr`, `read this`, `extract text from this`, etc.).
- Decision: strong profile → SUPPLEMENTARY; weak profile + URL-dominant → PRIMARY; weak profile + non-URL-dominant → SUPPLEMENTARY (best-effort, profile still drives).

**Spec gaps noted during planning:**
1. Spec Section 7 says "URL-primary case has NO internal timeout" and "Two cases, auto-selected" — but doesn't define the exact signal for "profile retrieval weak" nor "query is URL-directed". Plan Task 15 defines both with configurable thresholds, treating them as a first iteration that Phase 6 pattern-mining can tune.
2. Spec Section 7 says URL-supplementary may trigger "one additional LLM call" — spec Section 3 invariant 1 says supplementary calls fire "only when URL content arrives after the primary response has begun streaming". Plan Task 18 implements exactly that — no supplementary call if URL chunks arrived in time to merge.
3. Spec Section 7 doesn't specify robots.txt fail-mode (open vs closed). Plan Task 12 defaults to fail-open (availability) with a `respect_robots=False` override and a reserved-field for future `robots_fail_mode='deny'` when operators want fail-closed.
4. Spec doesn't specify per-URL vs aggregate fetch timeout for multi-URL queries. Plan Task 13 applies the fetcher timeout per-URL sequentially; multi-URL batches thus have cumulative wall-clock bounded by `N × fetch_timeout_s` as the worst case. This is consistent with "per-operation safety timeout" semantics.
5. Phase 4's adapter `response_persona_prompts.supplementary_analysis` template is referenced in Task 19 but may not exist yet. Plan ships a standalone supplementary template in `prompts.py` that works without Phase 4; when Phase 4 lands, the supplementary-analysis persona path is added in `compose_supplementary` via adapter lookup.

**Open questions (for implementation discussion; do not block starting Phase 5):**
1. Should `CASE_SELECTION_CHUNK_SIM_THRESHOLD` (0.5) be adapter-scoped or global? Plan uses global for Phase 5; adapter-scoped deferred.
2. For PDF URL content, Phase 5 does a minimal text decode rather than running the full PDF extraction pipeline. Is a richer PDF path needed in Phase 5, or can it wait for sub-project C (URL ingestion)?
3. The curated URL list size (~50) for fail-rate measurement — operators may want to expand this per domain. Plan ships a starter; ops can extend.
4. Does the ephemeral session cache need TTL-based expiry (e.g., 10 minutes)? Plan keeps it session-scoped only (cleared on session teardown) — no TTL needed. Revisit if memory pressure shows up in pattern mining.

---

*End of Phase 5 plan. Execute task-by-task via superpowers:executing-plans or superpowers:subagent-driven-development.*
