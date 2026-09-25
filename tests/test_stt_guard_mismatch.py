import asyncio
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from config.settings import BUFFER_MUTE_GUARD, SAMPLE_RATE
from core.queues import audio_queue, text_queue
from core.events import interrupt_event, assistant_speaking
import core.events as ev
from core.sentinel import SILENCE_MARKER, INTERRUPT


@pytest.mark.asyncio
async def test_stt_does_not_discard_barge_in_within_mute_guard():
    """
    Test that confirmed barge-in audio arriving within BUFFER_MUTE_GUARD
    (e.g., 1.0s after assistant finished speaking) is NOT discarded as echo.
    """
    interrupt_event.clear()
    assistant_speaking.clear()
    # Simulate assistant speaking ended 1.0s ago (which is < BUFFER_MUTE_GUARD = 4.0s)
    ev.speaking_ended_at = time.monotonic() - 1.0

    # Create dummy audio chunk with audible speech energy (sine wave)
    t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5), endpoint=False)
    speech_data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    chunk = speech_data[:512]

    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Hello world"
    mock_model.transcribe.return_value = ([mock_segment], None)

    # Queue an INTERRUPT sentinel (barge-in signal), speech chunks, and SILENCE_MARKER
    await audio_queue.put(INTERRUPT)
    for _ in range(5):
        await audio_queue.put(chunk)
    await audio_queue.put(SILENCE_MARKER)

    with patch("pipeline.stt.WhisperModel", return_value=mock_model):
        from pipeline.stt import speech_to_text_stream

        task = asyncio.create_task(speech_to_text_stream())

        try:
            transcript = await asyncio.wait_for(text_queue.get(), timeout=2.0)
            assert transcript == "Hello world"
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_stt_discards_ordinary_echo_within_mute_guard():
    """
    Test that ordinary microphone audio (NOT barge-in) collected within BUFFER_MUTE_GUARD is discarded.
    """
    interrupt_event.clear()
    assistant_speaking.clear()
    ev.speaking_ended_at = time.monotonic() - 1.0

    t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5), endpoint=False)
    chunk = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[:512]

    mock_model = MagicMock()

    # NO INTERRUPT sentinel — just ordinary chunks (echo candidate)
    for _ in range(5):
        await audio_queue.put(chunk)
    await audio_queue.put(SILENCE_MARKER)

    with patch("pipeline.stt.WhisperModel", return_value=mock_model):
        from pipeline.stt import speech_to_text_stream

        task = asyncio.create_task(speech_to_text_stream())

        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(text_queue.get(), timeout=0.5)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
