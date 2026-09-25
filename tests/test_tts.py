import io
import wave
import numpy as np
import pytest

from pipeline.tts import _wav_bytes_to_numpy


def test_wav_bytes_to_numpy():
    sample_rate = 24000
    duration = 0.1
    num_samples = int(sample_rate * duration)
    expected_samples = np.linspace(-0.8, 0.8, num_samples, dtype=np.float32)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((expected_samples * 32768.0).astype(np.int16).tobytes())

    wav_bytes = buf.getvalue()
    samples, sr = _wav_bytes_to_numpy(wav_bytes)

    assert sr == sample_rate
    assert len(samples) == num_samples
    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float32
    # Allow 1-bit quantization difference from 16-bit PCM conversion
    np.testing.assert_allclose(samples, expected_samples, atol=1e-4)


@pytest.mark.asyncio
async def test_synthesizer_kokoro():
    from unittest.mock import MagicMock
    import asyncio
    import core.queues as queues
    from core.turn import turn_controller
    from core.sentinel import END_OF_SPEECH, END_OF_RESPONSE
    from core.state import TurnLatency
    from pipeline.tts import _synthesizer

    queues.reset_queues()
    turn_controller.reset_for_test(1)

    mock_kokoro = MagicMock()
    mock_kokoro.create.return_value = (np.zeros(2400, dtype=np.float32), 24000)

    phrase_q = asyncio.Queue()
    synth_task = asyncio.create_task(_synthesizer(mock_kokoro, phrase_q))

    # Send phrase for gen 1
    latency = TurnLatency()
    await phrase_q.put((1, "Hello world.", latency))
    # Send END_OF_RESPONSE
    await phrase_q.put((1, END_OF_RESPONSE, latency))

    # Get synthesized audio chunk from tts_queue
    item1 = await asyncio.wait_for(queues.tts_queue.get(), timeout=2.0)
    gen_id, (audio, sr), lat = item1
    assert gen_id == 1
    assert sr == 24000
    assert len(audio) == 2400

    # Get END_OF_SPEECH sentinel
    item2 = await asyncio.wait_for(queues.tts_queue.get(), timeout=2.0)
    gen_id2, sentinel, _ = item2
    assert gen_id2 == 1
    assert sentinel is END_OF_SPEECH

    synth_task.cancel()
    try:
        await synth_task
    except asyncio.CancelledError:
        pass
