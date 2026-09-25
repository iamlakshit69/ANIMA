import io
import wave
import numpy as np
import pytest

from pipeline.stt import _to_wav_bytes, _trim_trailing_silence
from config.settings import SAMPLE_RATE, SILENCE_RMS


def test_to_wav_bytes():
    # 0.5s of 440Hz tone
    t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5), False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    buf = _to_wav_bytes(audio)
    assert isinstance(buf, io.BytesIO)

    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnframes() == len(audio)


def test_trim_trailing_silence():
    # Speech chunk followed by multiple silent chunks
    speech = np.ones(480, dtype=np.float32) * 0.1
    silent = np.zeros(480, dtype=np.float32)

    buffer = [speech, speech] + [silent] * 10
    trimmed = _trim_trailing_silence(buffer)

    # Must preserve speech + safe trailing decay while trimming dead silence
    assert len(trimmed) < len(buffer)
    assert len(trimmed) >= 2
    # Verify original speech chunks are intact
    np.testing.assert_array_equal(trimmed[0], speech)
    np.testing.assert_array_equal(trimmed[1], speech)
