"""Semaphore-bounded concurrent document processing workers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Tuple

logger = logging.getLogger(__name__)


class WorkerPool:
    """Runs async tasks with bounded concurrency."""

    def __init__(self, max_concurrent: int = 3):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(
        self, task_id: str, coro_fn: Callable[..., Coroutine], args: Tuple
    ) -> Tuple[str, Any]:
        async with self.semaphore:
            try:
                result = await coro_fn(*args)
                return task_id, result
            except Exception as exc:
                logger.error("Worker task %s failed: %s", task_id, exc)
                return task_id, exc

    async def run_all(
        self, tasks: List[Tuple[str, Callable[..., Coroutine], Tuple]]
    ) -> Dict[str, Any]:
        """Run all tasks with bounded concurrency.

        Args:
            tasks: List of (task_id, async_callable, args) tuples.

        Returns:
            Dict mapping task_id to result (or Exception on failure).
        """
        if not tasks:
            return {}

        coros = [self._run_one(tid, fn, args) for tid, fn, args in tasks]
        completed = await asyncio.gather(*coros)
        return dict(completed)
