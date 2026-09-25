import asyncio
import pytest
from main import _supervise


@pytest.mark.asyncio
async def test_supervise_restarts_on_crash():
    run_count = 0

    async def flaky_task():
        nonlocal run_count
        run_count += 1
        if run_count == 1:
            raise RuntimeError("Simulated crash in stage")
        # Keep running on second attempt
        while True:
            await asyncio.sleep(0.01)

    task = asyncio.create_task(_supervise("test_stage", flaky_task, initial_backoff=0.01))
    await asyncio.sleep(0.08)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert run_count >= 2


@pytest.mark.asyncio
async def test_supervise_cancels_cleanly():
    async def normal_task():
        while True:
            await asyncio.sleep(0.01)

    task = asyncio.create_task(_supervise("normal_stage", normal_task))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
