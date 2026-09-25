import asyncio
import pytest
from core.turn import TurnController


@pytest.mark.asyncio
async def test_turn_controller_bump():
    tc = TurnController()
    assert tc.current == 0

    g1 = await tc.bump()
    assert g1 == 1
    assert tc.current == 1

    g2 = await tc.bump()
    assert g2 == 2
    assert tc.current == 2


@pytest.mark.asyncio
async def test_turn_controller_concurrency():
    tc = TurnController()
    
    # Bump 50 times concurrently
    async def bumper():
        return await tc.bump()

    results = await asyncio.gather(*(bumper() for _ in range(50)))
    assert len(results) == 50
    assert len(set(results)) == 50  # all unique
    assert tc.current == 50
