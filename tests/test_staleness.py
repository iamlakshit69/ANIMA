import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, patch

from core.turn import turn_controller
from core.state import playback_state, TurnLatency
from core.sentinel import SILENCE_MARKER, END_OF_RESPONSE, END_OF_SPEECH
import core.queues as queues
from pipeline.stt import speech_to_text_stream
from pipeline.llm import llm_stream
from pipeline.tts import _accumulator, _synthesizer
from pipeline.speaker import speaker_stream


@pytest.fixture(autouse=True)
def reset_state():
    """Reset turn and queues between tests."""
    from core.queues import reset_queues
    reset_queues()
    turn_controller.reset_for_test(0)
    playback_state.mark_ended()


@pytest.mark.asyncio
async def test_stt_drops_stale_generation():
    """Verify STT discards audio buffers whose generation ID != current."""
    from unittest.mock import MagicMock
    await turn_controller.bump()  # gen = 1
    stale_gen = 1

    # Start STT task with mocked Whisper model
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], None)
    with patch("pipeline.stt.WhisperModel", return_value=mock_model):
        stt_task = asyncio.create_task(speech_to_text_stream())
        await asyncio.sleep(0.01)  # allow model warmup to complete
        mock_model.transcribe.reset_mock()

        # Put audio chunk for gen 1
        fake_chunk = np.ones(480, dtype=np.float32) * 0.1
        await queues.audio_queue.put((stale_gen, fake_chunk))

        # Now barge-in occurs: bump generation to 2!
        await turn_controller.bump()  # gen = 2

        # Send SILENCE_MARKER for the old gen 1
        await queues.audio_queue.put((stale_gen, SILENCE_MARKER))

        await asyncio.sleep(0.05)
        stt_task.cancel()
        try:
            await stt_task
        except asyncio.CancelledError:
            pass

        # Verify text_queue received nothing from the superseded turn
        assert queues.text_queue.empty()
        mock_model.transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_llm_drops_stale_text():
    """Verify LLM drops user utterances belonging to superseded turns without calling API."""
    await turn_controller.bump()  # gen = 1
    stale_gen = 1

    mock_client = AsyncMock()
    with patch("pipeline.llm.AsyncGroq", return_value=mock_client):
        llm_task = asyncio.create_task(llm_stream())

        # Now barge-in / new turn: bump to 2!
        await turn_controller.bump()  # gen = 2

        # Put stale text for gen 1 into text_queue
        await queues.text_queue.put((stale_gen, "Stale question", TurnLatency()))

        await asyncio.sleep(0.05)
        llm_task.cancel()
        try:
            await llm_task
        except asyncio.CancelledError:
            pass

        # API should not be called for the stale turn
        mock_client.chat.completions.create.assert_not_called()
        assert queues.token_queue.empty()


@pytest.mark.asyncio
async def test_tts_accumulator_drops_stale_tokens():
    """Verify TTS accumulator drops tokens belonging to superseded turns."""
    phrase_queue = asyncio.Queue()
    accum_task = asyncio.create_task(_accumulator(phrase_queue))

    await turn_controller.bump()  # gen = 1
    stale_gen = 1

    # Send a token for gen 1
    await queues.token_queue.put((stale_gen, "Hello world.", TurnLatency()))

    # Bump to gen 2 (barge-in)
    await turn_controller.bump()

    # Send END_OF_RESPONSE for gen 1
    await queues.token_queue.put((stale_gen, END_OF_RESPONSE, TurnLatency()))

    await asyncio.sleep(0.05)
    accum_task.cancel()
    try:
        await accum_task
    except asyncio.CancelledError:
        pass

    # phrase_queue should have discarded the stale phrase
    assert phrase_queue.empty()


@pytest.mark.asyncio
async def test_speaker_drops_stale_audio_and_eos():
    """Verify speaker drops stale audio and trailing END_OF_SPEECH without affecting new turn."""
    await turn_controller.bump()  # gen = 1
    stale_gen = 1

    with patch("pipeline.speaker.sd") as mock_sd:
        mock_sd.get_stream.return_value.active = False
        speaker_task = asyncio.create_task(speaker_stream())

        # Barge-in happens: bump to gen 2
        await turn_controller.bump()  # gen = 2

        # Queue stale audio chunk and stale END_OF_SPEECH from gen 1
        fake_audio = np.zeros(480, dtype=np.float32)
        await queues.tts_queue.put((stale_gen, (fake_audio, 16000), TurnLatency()))
        await queues.tts_queue.put((stale_gen, END_OF_SPEECH, TurnLatency()))

        await asyncio.sleep(0.05)
        speaker_task.cancel()
        try:
            await speaker_task
        except asyncio.CancelledError:
            pass

        # sd.play must NOT have been called for stale gen 1
        mock_sd.play.assert_not_called()
