"""OneDrive/SharePoint file download via Microsoft Graph API."""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_ONEDRIVE_PATTERN = re.compile(
    r"https?://[^/]*(?:sharepoint\.com|onedrive\.live\.com|1drv\.ms)[^\s]*",
    re.IGNORECASE,
)


def is_onedrive_url(text: str) -> Optional[str]:
    """Extract a OneDrive/SharePoint URL from message text. Returns URL or None."""
    match = _ONEDRIVE_PATTERN.search(text)
    return match.group(0) if match else None


async def download_shared_file(
    url: str,
    access_token: str,
    max_bytes: int = 50 * 1024 * 1024,
    timeout: int = 60,
) -> Tuple[bytes, str]:
    """Download a file from a OneDrive/SharePoint sharing URL.

    Uses the Graph API shares endpoint to resolve sharing links.

    Args:
        url: OneDrive/SharePoint sharing URL.
        access_token: Graph API access token with Files.Read.All scope.
        max_bytes: Maximum file size to download.
        timeout: HTTP timeout in seconds.

    Returns:
        Tuple of (file_bytes, filename).

    Raises:
        ValueError: If the URL can't be resolved or file is too large.
        httpx.HTTPStatusError: On API errors.
    """
    import base64

    encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    share_id = f"u!{encoded}"

    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        meta_url = f"https://graph.microsoft.com/v1.0/shares/{share_id}/driveItem"
        meta_resp = await client.get(meta_url, headers=headers)
        meta_resp.raise_for_status()
        item = meta_resp.json()

        filename = item.get("name", "unknown_file")
        size = item.get("size", 0)

        if size > max_bytes:
            raise ValueError(
                f"File {filename} is {size / 1024 / 1024:.1f}MB, "
                f"exceeds limit of {max_bytes / 1024 / 1024:.0f}MB"
            )

        content_url = f"{meta_url}/content"
        content_resp = await client.get(content_url, headers=headers, follow_redirects=True)
        content_resp.raise_for_status()

        logger.info("Downloaded %s (%d bytes) from OneDrive/SharePoint", filename, len(content_resp.content))
        return content_resp.content, filename
