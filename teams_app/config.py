"""Teams App service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Set


EXPRESS_FILE_TYPES: Set[str] = {
    ".txt", ".md", ".csv", ".xlsx", ".json", ".xml", ".html",
}

FULL_FILE_TYPES: Set[str] = {
    ".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}


@dataclass
class TeamsAppConfig:
    """Configuration for the standalone Teams service."""

    # Service
    port: int = field(default=0)
    host: str = field(default="")

    # Query proxy
    main_app_url: str = field(default="")
    proxy_timeout: int = field(default=0)

    # Pipeline
    max_concurrent_documents: int = field(default=0)
    express_min_chars: int = field(default=0)
    express_chunk_size: int = field(default=0)
    full_chunk_size: int = field(default=0)

    # Learning signals
    signals_dir: str = field(default="")

    def __post_init__(self) -> None:
        # Service
        if not self.port:
            self.port = int(os.getenv("TEAMS_APP_PORT", "8300"))
        if not self.host:
            self.host = os.getenv("TEAMS_APP_HOST", "0.0.0.0")

        # Query proxy
        if not self.main_app_url:
            self.main_app_url = os.getenv("TEAMS_MAIN_APP_URL", "http://localhost:8000")
        if not self.proxy_timeout:
            self.proxy_timeout = int(os.getenv("TEAMS_PROXY_TIMEOUT", "120"))

        # Pipeline
        if not self.max_concurrent_documents:
            self.max_concurrent_documents = int(os.getenv("TEAMS_MAX_CONCURRENT_DOCS", "3"))
        if not self.express_min_chars:
            self.express_min_chars = int(os.getenv("TEAMS_EXPRESS_MIN_CHARS", "50"))
        if not self.express_chunk_size:
            self.express_chunk_size = int(os.getenv("TEAMS_EXPRESS_CHUNK_SIZE", "1024"))
        if not self.full_chunk_size:
            self.full_chunk_size = int(os.getenv("TEAMS_FULL_CHUNK_SIZE", "2048"))

        # Learning signals
        if not self.signals_dir:
            self.signals_dir = os.getenv(
                "TEAMS_SIGNALS_DIR",
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "outputs", "learning_signals"),
            )
