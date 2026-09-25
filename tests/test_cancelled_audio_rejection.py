import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.queues import token_queue, tts_queue
from core.events import interrupt_event, assistant_speaking
import core.events as ev
from core.sentinel import END_OF_RESPONSE, END_OF_SPEECH


@pytest.mark.asyncio
async def test_tts_synthesizer_discards_audio_if_interrupted_during_synthesis():
    """
    Test that _synthesizer discards synthesized audio if interrupt_event is set
    before or during phrase synthesis, preventing stale audio from reaching tts_queue.
    """
    phrase_queue = asyncio.Queue()
    mock_voice = MagicMock()
    mock_voice.config.sample_rate = 16000

    from pipeline.tts import _synthesizer

    task = asyncio.create_task(_synthesizer(mock_voice, phrase_queue))

    # Trigger interrupt BEFORE phrase is processed
    interrupt_event.set()
    await phrase_queue.put("This phrase should be discarded because of interrupt.")

    await asyncio.sleep(0.1)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Verify nothing was added to tts_queue
    assert tts_queue.empty(), f"tts_queue should be empty, but contains {tts_queue.qsize()} items"


@pytest.mark.asyncio
async def test_speaker_discards_pending_phrases_on_interrupt():
    """
    Test that when an interruption occurs in speaker_stream, all remaining
    pending phrases in tts_queue from the cancelled turn are discarded.
    """
    # Pre-populate tts_queue with 3 phrases
    dummy_samples = np.zeros(2048, dtype=np.float32)
    sample_rate = 16000
    await tts_queue.put((dummy_samples, sample_rate))
    await tts_queue.put((dummy_samples, sample_rate))
    await tts_queue.put((dummy_samples, sample_rate))

    # Set interrupt event (barge-in already fired)
    interrupt_event.set()

    mock_pyaudio = MagicMock()

    with patch("pipeline.speaker.pyaudio.PyAudio", return_value=mock_pyaudio):
        from pipeline.speaker import speaker_stream

        task = asyncio.create_task(speaker_stream())
        await asyncio.sleep(0.1)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # All phrases should have been flushed from tts_queue upon interruption
    assert tts_queue.empty(), f"tts_queue still has {tts_queue.qsize()} items after interrupt"
    assert not assistant_speaking.is_set(), "assistant_speaking should be cleared"
    assert not interrupt_event.is_set(), "interrupt_event should be acknowledged and cleared"
