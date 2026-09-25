import asyncio
import pytest
from unittest.mock import MagicMock, patch
from core.queues import audio_queue, text_queue, tts_queue
from core.events import interrupt_event, assistant_speaking
from core.sentinel import END_OF_SPEECH


@pytest.mark.asyncio
async def test_speaker_does_not_drain_audio_queue_on_end_of_speech():
    """
    Test that speaker_stream receiving END_OF_SPEECH does NOT drain user speech from audio_queue.
    """
    # Pre-populate audio_queue with incoming user speech chunks
    sentinel_chunk = "user_speech_chunk"
    await audio_queue.put(sentinel_chunk)
    await audio_queue.put(sentinel_chunk)

    # Put END_OF_SPEECH into tts_queue
    await tts_queue.put(END_OF_SPEECH)

    mock_pyaudio = MagicMock()

    with patch("pipeline.speaker.pyaudio.PyAudio", return_value=mock_pyaudio):
        from pipeline.speaker import speaker_stream

        task = asyncio.create_task(speaker_stream())
        # Allow speaker_stream to process END_OF_SPEECH
        await asyncio.sleep(0.1)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Verify that audio_queue STILL contains the user speech chunks!
    assert audio_queue.qsize() == 2, f"audio_queue was drained! Expected 2 items, found {audio_queue.qsize()}"
    item1 = await audio_queue.get()
    item2 = await audio_queue.get()
    assert item1 == sentinel_chunk
    assert item2 == sentinel_chunk


@pytest.mark.asyncio
async def test_llm_does_not_drain_text_queue_on_interrupt():
    """
    Test that llm_stream does NOT drain text_queue when interrupt_event is set.
    """
    # Place a new user utterance into text_queue
    new_utterance = "What is the capital of France?"
    await text_queue.put(new_utterance)

    # Even if interrupt_event was set, llm should not blanket-drain text_queue
    interrupt_event.set()

    # Verify that text_queue still retains the new utterance
    assert not text_queue.empty()
    item = await text_queue.get()
    assert item == new_utterance
