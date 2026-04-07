"""HTTP proxy to main app /api/ask with SSE streaming support."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class QueryRequest:
    """A query to proxy to the main app."""
    query: str
    user_id: str
    subscription_id: str
    tenant_id: str = ""
    profile_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "user_id": self.user_id,
            "subscription_id": self.subscription_id,
            "profile_id": self.profile_id,
            "session_id": self.session_id,
            "stream": self.stream,
        }

    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-source": "teams",
            "x-teams-tenant": self.tenant_id,
        }


@dataclass
class QueryResult:
    """Result from the main app."""
    response: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    grounded: bool = False
    context_found: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def from_response(cls, data: Dict[str, Any]) -> QueryResult:
        return cls(
            response=data.get("response", ""),
            sources=data.get("sources", []),
            grounded=data.get("grounded", False),
            context_found=data.get("context_found", False),
            metadata=data.get("metadata", {}),
        )


class QueryProxy:
    """Proxies queries to the main DocWain app."""

    def __init__(self, main_app_url: str, timeout: int = 120):
        self.main_app_url = main_app_url.rstrip("/")
        self.timeout = timeout

    async def ask(self, request: QueryRequest) -> QueryResult:
        """Send a query to the main app and return the result."""
        url = f"{self.main_app_url}/api/ask"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=request.to_dict(), headers=request.headers())
                resp.raise_for_status()
                import inspect
                json_result = resp.json()
                if inspect.isawaitable(json_result):
                    json_result = await json_result
                return QueryResult.from_response(json_result)
        except Exception as exc:
            logger.error("Query proxy failed: %s", exc)
            return QueryResult(
                response="I'm having trouble right now. Please try again in a moment.",
                error=str(exc),
            )

    async def ask_stream(self, request: QueryRequest) -> AsyncIterator[str]:
        """Stream SSE tokens from the main app. Yields partial text chunks."""
        url = f"{self.main_app_url}/api/ask"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=request.to_dict(), headers=request.headers()) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            payload = line[6:]
                            if payload == "[DONE]":
                                break
                            try:
                                chunk = json.loads(payload)
                                token = chunk.get("token", chunk.get("response", ""))
                                if token:
                                    yield token
                            except json.JSONDecodeError:
                                yield payload
        except Exception as exc:
            logger.error("Stream proxy failed: %s", exc)
            yield "I'm having trouble right now. Please try again in a moment."
