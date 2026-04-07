import asyncio
import pytest
from teams_app.pipeline.workers import WorkerPool


@pytest.mark.asyncio
async def test_worker_pool_limits_concurrency():
    """Verify that only max_concurrent tasks run at once."""
    pool = WorkerPool(max_concurrent=2)
    running = []
    max_running = 0

    async def track_task(task_id: str):
        nonlocal max_running
        running.append(task_id)
        current = len(running)
        if current > max_running:
            max_running = current
        await asyncio.sleep(0.05)
        running.remove(task_id)
        return f"done-{task_id}"

    results = await pool.run_all([
        ("a", track_task, ("a",)),
        ("b", track_task, ("b",)),
        ("c", track_task, ("c",)),
        ("d", track_task, ("d",)),
    ])

    assert max_running <= 2
    assert len(results) == 4
    assert all(r.startswith("done-") for r in results.values())


@pytest.mark.asyncio
async def test_worker_pool_captures_errors():
    pool = WorkerPool(max_concurrent=2)

    async def fail_task(task_id: str):
        raise ValueError(f"fail-{task_id}")

    async def ok_task(task_id: str):
        return f"ok-{task_id}"

    results = await pool.run_all([
        ("good", ok_task, ("good",)),
        ("bad", fail_task, ("bad",)),
    ])

    assert results["good"] == "ok-good"
    assert isinstance(results["bad"], Exception)


@pytest.mark.asyncio
async def test_worker_pool_empty_input():
    pool = WorkerPool(max_concurrent=3)
    results = await pool.run_all([])
    assert results == {}
