"""NICE guidelines scraper — fetches published guidance from nice.org.uk.

Handles pagination and rate limiting when scraping the NICE public API
and guideline pages. Returns structured dicts with guideline metadata
and raw HTML/text content for parsing.
"""

from __future__ import annotations

import time
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from src.knowledge_packs.base import KnowledgePackScraper
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://www.nice.org.uk"
_API_URL = "https://www.nice.org.uk/guidance/published"
_GUIDELINE_TYPES = {"NG", "TA", "QS", "CG"}

# Rate limiting
_REQUEST_DELAY_SECONDS = 1.0
_MAX_PAGES = 50


class NICEScraper(KnowledgePackScraper):
    """Scraper for NICE clinical guidelines.

    Fetches the published guidance list from nice.org.uk, then downloads
    individual guideline pages for detailed content extraction.
    """

    def __init__(
        self,
        *,
        base_url: str = _BASE_URL,
        content_types: Optional[List[str]] = None,
        max_pages: int = _MAX_PAGES,
        request_delay: float = _REQUEST_DELAY_SECONDS,
    ) -> None:
        self._base_url = base_url
        self._content_types = set(content_types or _GUIDELINE_TYPES)
        self._max_pages = max_pages
        self._delay = request_delay

    def scrape(self) -> List[Dict[str, Any]]:
        """Full scrape of published NICE guidelines.

        Returns a list of dicts, each with:
        - id: guideline identifier (e.g., "NG123")
        - title: guideline title
        - url: full URL to the guideline
        - type: guideline type (NG/TA/QS/CG)
        - sections: list of dicts with heading and content
        - last_updated: ISO date string
        """
        logger.info("Starting full NICE scrape", extra={"types": list(self._content_types)})

        guideline_list = self._fetch_guideline_list()
        results: List[Dict[str, Any]] = []

        for item in guideline_list:
            gtype = item.get("type", "")
            if gtype not in self._content_types:
                continue

            try:
                detail = self._fetch_guideline_detail(item)
                results.append(detail)
            except Exception:
                logger.warning(
                    "Failed to fetch guideline detail",
                    extra={"id": item.get("id", "?")},
                    exc_info=True,
                )

            time.sleep(self._delay)

        logger.info("NICE scrape complete", extra={"total": len(results)})
        return results

    def check_updates(self, since: datetime) -> List[Dict[str, Any]]:
        """Check for new or updated guidance since the given date.

        Fetches the guideline list and filters by last_updated date.
        """
        logger.info("Checking NICE updates", extra={"since": since.isoformat()})

        guideline_list = self._fetch_guideline_list()
        updated: List[Dict[str, Any]] = []

        for item in guideline_list:
            gtype = item.get("type", "")
            if gtype not in self._content_types:
                continue

            last_updated = item.get("last_updated", "")
            if last_updated:
                try:
                    item_date = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
                    if item_date.replace(tzinfo=None) < since:
                        continue
                except (ValueError, TypeError):
                    pass

            try:
                detail = self._fetch_guideline_detail(item)
                updated.append(detail)
            except Exception:
                logger.warning(
                    "Failed to fetch updated guideline",
                    extra={"id": item.get("id", "?")},
                    exc_info=True,
                )

            time.sleep(self._delay)

        logger.info("NICE update check complete", extra={"updated": len(updated)})
        return updated

    # -- Internal methods ----------------------------------------------------

    def _fetch_guideline_list(self) -> List[Dict[str, Any]]:
        """Fetch the paginated list of published guidelines.

        Tries the NICE API first; falls back to HTML scraping if needed.
        Returns a list of dicts with id, title, url, type, last_updated.
        """
        import urllib.request
        import urllib.error
        import json

        guidelines: List[Dict[str, Any]] = []
        page = 1

        while page <= self._max_pages:
            url = f"{self._base_url}/guidance/published?type=&page={page}"
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "DocWain-KnowledgePack/1.0",
                        "Accept": "application/json, text/html",
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    content_type = resp.headers.get("Content-Type", "")
                    body = resp.read().decode("utf-8", errors="replace")

                    if "application/json" in content_type:
                        data = json.loads(body)
                        items = self._parse_api_response(data)
                    else:
                        items = self._parse_html_list(body)

                    if not items:
                        break

                    guidelines.extend(items)
                    page += 1
                    time.sleep(self._delay)

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
                logger.warning("HTTP error fetching guideline list", extra={"page": page, "code": e.code})
                break
            except Exception:
                logger.warning("Error fetching guideline list", extra={"page": page}, exc_info=True)
                break

        logger.info("Fetched guideline list", extra={"total": len(guidelines)})
        return guidelines

    def _parse_api_response(self, data: Any) -> List[Dict[str, Any]]:
        """Parse JSON API response into guideline items."""
        items: List[Dict[str, Any]] = []
        results = data if isinstance(data, list) else data.get("results", data.get("items", []))

        for r in results:
            gid = r.get("id", r.get("reference", ""))
            title = r.get("title", "")
            url = r.get("url", r.get("link", ""))
            if url and not url.startswith("http"):
                url = urljoin(self._base_url, url)

            gtype = self._extract_type(gid)

            items.append({
                "id": gid,
                "title": title,
                "url": url,
                "type": gtype,
                "last_updated": r.get("lastUpdated", r.get("last_updated", "")),
            })

        return items

    def _parse_html_list(self, html: str) -> List[Dict[str, Any]]:
        """Extract guideline references from HTML listing page."""
        items: List[Dict[str, Any]] = []

        # Pattern: links like /guidance/NG123
        link_pattern = re.compile(
            r'href="(/guidance/([A-Z]{2,3}\d+))"[^>]*>([^<]+)<',
            re.IGNORECASE,
        )

        for match in link_pattern.finditer(html):
            path, gid, title = match.groups()
            gtype = self._extract_type(gid)
            items.append({
                "id": gid.upper(),
                "title": title.strip(),
                "url": urljoin(self._base_url, path),
                "type": gtype,
                "last_updated": "",
            })

        # Deduplicate by id
        seen = set()
        unique: List[Dict[str, Any]] = []
        for item in items:
            if item["id"] not in seen:
                seen.add(item["id"])
                unique.append(item)

        return unique

    def _fetch_guideline_detail(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch full guideline page and extract sections."""
        import urllib.request

        url = item.get("url", "")
        if not url:
            return item

        # Try to fetch the recommendations/chapter page
        detail_url = url.rstrip("/")
        sections: List[Dict[str, str]] = []

        try:
            req = urllib.request.Request(
                detail_url,
                headers={"User-Agent": "DocWain-KnowledgePack/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="replace")
                sections = self._extract_sections(html)
        except Exception:
            logger.debug("Could not fetch detail page", extra={"url": detail_url})

        result = dict(item)
        result["sections"] = sections
        return result

    def _extract_sections(self, html: str) -> List[Dict[str, str]]:
        """Extract heading/content sections from guideline HTML."""
        sections: List[Dict[str, str]] = []

        # Extract content between heading tags
        heading_pattern = re.compile(
            r"<h([2-4])[^>]*>(.*?)</h\1>\s*(.*?)(?=<h[2-4]|$)",
            re.DOTALL | re.IGNORECASE,
        )

        for match in heading_pattern.finditer(html):
            heading = self._strip_html(match.group(2)).strip()
            content = self._strip_html(match.group(3)).strip()
            if heading and content:
                sections.append({"heading": heading, "content": content})

        return sections

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags and decode entities."""
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&#\d+;", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_type(guideline_id: str) -> str:
        """Extract guideline type prefix (NG, TA, QS, CG) from ID."""
        match = re.match(r"^([A-Z]{2,3})", guideline_id.upper())
        if match:
            prefix = match.group(1)
            if prefix in _GUIDELINE_TYPES:
                return prefix
        return "OTHER"
