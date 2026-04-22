"""Query runner: hits DocWain /api/ask and captures results + latency.

Extracts the pattern from scripts/intensive_test.py into a reusable, testable
component. The runner uses httpx with a per-request safety timeout — this is
the ONLY timeout in the eval path, and it exists to prevent a dead server
from blocking the whole baseline. It is NOT a response-quality cutoff.
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


@dataclass(frozen=True)
class RunnerConfig:
    base_url: str
    path: str = "/api/ask"
    user_id: str = "sme_eval@docwain.internal"
    fetch_timeout_s: float = 120.0  # per-operation safety; not a quality cutoff


class QueryRunner:
    """Synchronous query runner. Hits DocWain /api/ask once per query.

    Sequential by design — parallelism would confound latency measurement.
    """

    def __init__(self, config: RunnerConfig):
        self._config = config
        self._client = httpx.Client(timeout=config.fetch_timeout_s)

    def run_one(self, query: EvalQuery, *, run_id: str) -> EvalResult:
        """Run one query. Captures response or error; never raises."""
        payload = {
            "query": query.query_text,
            "subscription_id": query.subscription_id,
            "profile_id": query.profile_id,
            "user_id": self._config.user_id,
        }

        url = f"{self._config.base_url}{self._config.path}"
        start = time.perf_counter()
        captured_at = datetime.now(timezone.utc).replace(tzinfo=None)

        try:
            response = self._client.post(url, json=payload)
            total_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()
            body = response.json()
            resp_payload = body.get("payload", body)
            return EvalResult(
                query=query,
                response_text=resp_payload.get("response", "") or "",
                sources=resp_payload.get("sources", []) or [],
                metadata=resp_payload.get("metadata", {}) or {},
                latency=LatencyBreakdown(total_ms=total_ms),
                run_id=run_id,
                captured_at=captured_at,
                api_status=response.status_code,
                api_error=None,
            )
        except httpx.HTTPStatusError as e:
            total_ms = (time.perf_counter() - start) * 1000.0
            return EvalResult(
                query=query,
                response_text="",
                sources=[],
                metadata={},
                latency=LatencyBreakdown(total_ms=total_ms),
                run_id=run_id,
                captured_at=captured_at,
                api_status=e.response.status_code,
                api_error=f"HTTPStatusError: {e}",
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError) as e:
            total_ms = (time.perf_counter() - start) * 1000.0
            return EvalResult(
                query=query,
                response_text="",
                sources=[],
                metadata={},
                latency=LatencyBreakdown(total_ms=total_ms),
                run_id=run_id,
                captured_at=captured_at,
                api_status=0,
                api_error=f"{type(e).__name__}: {e}",
            )

    def run_batch(self, queries: Iterable[EvalQuery], *, run_id: str) -> Iterator[EvalResult]:
        """Run queries sequentially. Yields results as they complete."""
        for q in queries:
            yield self.run_one(q, run_id=run_id)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "QueryRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
