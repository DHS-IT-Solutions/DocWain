"""Ephemeral URL pipeline: fetch + extract + chunk + embed, in-memory only.

Produces a list of :class:`EphemeralChunk` objects matching the retrieval
layer's chunk shape (text + embedding + metadata). NEVER persists anywhere:

* no Qdrant / Neo4j / Blob / Mongo / Redis writes
* results live only inside the returned :class:`EphemeralResult`
* every chunk is tagged ``ephemeral: true`` and carries ``source_url`` +
  ``fetched_at`` + ``session_id`` so pack assembly and auditing can see
  provenance without the caller threading extra metadata around

The chunker here is intentionally lighter than
:class:`src.embedding.chunking.section_chunker.SectionChunker`, which
requires a full ``ExtractedDocument``. URL content has no page boundaries
or table objects; paragraph-level splitting is sufficient and keeps the
per-fetch latency low.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.tools.url_fetcher import (
    FetcherConfig,
    FetchResult,
    UrlFetcherError,
    fetch as _default_fetch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------
@dataclass
class EphemeralChunk:
    """A single chunk of URL-derived text, with embedding + provenance."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    # Convenience accessors matching the plan's contract.
    @property
    def source_url(self) -> str:
        return self.metadata.get("source_url", "")

    @property
    def fetched_at(self) -> str:
        return self.metadata.get("fetched_at", "")


@dataclass
class EphemeralResult:
    chunks: List[EphemeralChunk] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level fetch indirection (tests patch this)
# ---------------------------------------------------------------------------
def fetch(*args, **kwargs):
    """Indirection so tests can patch ``src.tools.url_ephemeral_source.fetch``."""
    return _default_fetch(*args, **kwargs)


# ---------------------------------------------------------------------------
# Chunker / extractor utilities
# ---------------------------------------------------------------------------
_HTML_SCRIPT = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
_HTML_STYLE = re.compile(r"<style[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

_HTML_ENTITIES = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
    "&nbsp;": " ",
}


def _strip_html(html: str) -> str:
    text = _HTML_SCRIPT.sub(" ", html)
    text = _HTML_STYLE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    for entity, replacement in _HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    out_lines: List[str] = []
    for line in text.splitlines():
        collapsed = _WHITESPACE.sub(" ", line).strip()
        if collapsed:
            out_lines.append(collapsed)
    return "\n\n".join(out_lines)


def _extract_text(fetched: FetchResult) -> str:
    ctype = (fetched.content_type or "").lower()
    body = fetched.body
    decoded = body.decode("utf-8", errors="replace") if body else ""
    if "html" in ctype or "xhtml" in ctype:
        return _strip_html(decoded)
    # text/plain, application/json, or unknown: return body as-is.
    return decoded


def _chunk_text(text: str, target: int, maximum: int) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) <= maximum:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # If a single paragraph exceeds the max, hard-split at the cap.
            if len(p) > maximum:
                for i in range(0, len(p), maximum):
                    chunks.append(p[i:i + maximum])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)

    if not chunks and text.strip():
        chunks = [text.strip()[:maximum]]
    return chunks


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class EphemeralSource:
    """Fetch -> extract -> chunk -> embed pipeline. No persistence."""

    # Plan uses both names (EphemeralSource / UrlEphemeralSource) — alias below.
    def __init__(
        self,
        *,
        fetcher: Any = None,
        chunker: Any = None,
        embedder: Any,
        fetcher_config: Optional[FetcherConfig] = None,
    ) -> None:
        """Arguments:

        * ``fetcher`` — optional callable compatible with ``url_fetcher.fetch``.
          When ``None`` we dispatch via the module-level ``fetch`` seam so tests
          can patch ``src.tools.url_ephemeral_source.fetch`` directly.
        * ``chunker`` — optional callable ``(text) -> list[str]``; falls back
          to the built-in paragraph splitter.
        * ``embedder`` — required. Must expose ``embed(texts)`` or
          ``encode(texts)``.
        * ``fetcher_config`` — :class:`FetcherConfig`. When omitted the
          defaults apply.
        """
        self._fetcher = fetcher
        self._chunker = chunker
        self._embedder = embedder
        self._cfg = fetcher_config or FetcherConfig()

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def fetch_all(
        self,
        urls: List[str],
        *,
        profile_id: str = "",
        subscription_id: str = "",
        session_id: str = "",
    ) -> EphemeralResult:
        """Fetch + extract + chunk + embed each URL. No persistence.

        On fetch failure: skip that URL, emit a warning entry, continue.
        Identifiers (``profile_id`` / ``subscription_id`` / ``session_id``)
        are tagged into the chunk metadata purely for downstream audit;
        nothing is filtered or routed on them inside this module.
        """
        result = EphemeralResult()
        if not urls:
            return result

        for url in urls:
            try:
                fetched = self._fetch(url)
                chunks = self._process(
                    fetched,
                    profile_id=profile_id,
                    subscription_id=subscription_id,
                    session_id=session_id,
                )
                result.chunks.extend(chunks)
            except UrlFetcherError as exc:
                logger.warning(
                    "ephemeral url fetch failed for %s: %s", url, exc
                )
                result.warnings.append({
                    "url": url,
                    "error": str(exc)[:300],
                    "error_class": type(exc).__name__,
                })
            except Exception as exc:  # noqa: BLE001
                logger.exception("unexpected error fetching %s", url)
                result.warnings.append({
                    "url": url,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "error_class": type(exc).__name__,
                })

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fetch(self, url: str) -> FetchResult:
        if self._fetcher is not None:
            return self._fetcher(url, config=self._cfg)
        return fetch(url, config=self._cfg)

    def _process(
        self,
        fetched: FetchResult,
        *,
        profile_id: str,
        subscription_id: str,
        session_id: str,
    ) -> List[EphemeralChunk]:
        text = _extract_text(fetched)
        if not text.strip():
            return []

        raw_chunks = self._chunk(text)
        if not raw_chunks:
            return []

        embeddings = self._embed(raw_chunks)
        if len(embeddings) != len(raw_chunks):
            raise ValueError(
                "embedder returned mismatched embedding count"
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        out: List[EphemeralChunk] = []
        for i, (chunk_text, vec) in enumerate(zip(raw_chunks, embeddings)):
            out.append(EphemeralChunk(
                text=chunk_text,
                metadata={
                    "ephemeral": True,
                    "source_url": fetched.source_url or fetched.url,
                    "final_url": fetched.final_url,
                    "content_type": (fetched.content_type or "").split(";", 1)[0].strip(),
                    "resolved_ip": fetched.resolved_ip,
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                    "session_id": session_id,
                    "fetched_at": now_iso,
                    "chunk_index": i,
                },
                embedding=list(vec) if vec is not None else None,
            ))
        return out

    def _chunk(self, text: str) -> List[str]:
        if self._chunker is not None:
            return list(self._chunker(text))
        try:
            from src.api.config import Config
            target = int(getattr(Config.Retrieval, "CHUNK_SIZE", 900))
        except Exception:  # noqa: BLE001
            target = 900
        maximum = target + target // 2
        return _chunk_text(text, target=target, maximum=maximum)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self._embedder, "embed"):
            return list(self._embedder.embed(texts))
        if hasattr(self._embedder, "encode"):
            vecs = self._embedder.encode(texts)
            return [list(v) for v in vecs]
        raise TypeError(
            "embedder must expose .embed(texts) or .encode(texts)"
        )


# Alias used by the plan / core agent wiring.
UrlEphemeralSource = EphemeralSource
