import asyncio
import numpy as np
import pytest
from core.retry import retry_with_backoff, generate_error_tone


@pytest.mark.asyncio
async def test_retry_with_backoff_success():
    call_count = 0

    async def flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Temporary network hiccup")
        return "success"

    result = await retry_with_backoff(
        flaky_func,
        max_retries=3,
        initial_delay=0.01,
        backoff_factor=1.5,
    )
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_exhaustion():
    async def always_failing():
        raise TimeoutError("Endpoint timeout")

    with pytest.raises(TimeoutError):
        await retry_with_backoff(
            always_failing,
            max_retries=2,
            initial_delay=0.01,
            backoff_factor=1.5,
        )


@pytest.mark.asyncio
async def test_retry_unrecoverable_auth_error():
    call_count = 0

    async def auth_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid API key provided")

    with pytest.raises(ValueError):
        await retry_with_backoff(
            auth_error,
            max_retries=3,
            initial_delay=0.01,
        )
    # Must fail immediately without retrying
    assert call_count == 1


def test_generate_error_tone():
    tone = generate_error_tone(sample_rate=16000, duration=0.2)
    assert isinstance(tone, np.ndarray)
    assert tone.dtype == np.float32
    assert len(tone) == int(16000 * 0.2)
    assert np.max(np.abs(tone)) <= 1.0
