import pytest
from config.settings import (
    BUFFER_MUTE_GUARD,
    POST_SPEECH_MUTE,
    SAMPLE_RATE,
    CHUNK_SIZE,
    HALF_DUPLEX,
    QUEUE_MAX_SIZE,
)


def test_settings_invariants():
    """Verify cross-file invariant BUFFER_MUTE_GUARD >= POST_SPEECH_MUTE."""
    assert BUFFER_MUTE_GUARD >= POST_SPEECH_MUTE, (
        f"BUFFER_MUTE_GUARD ({BUFFER_MUTE_GUARD}) must be >= POST_SPEECH_MUTE ({POST_SPEECH_MUTE})"
    )


def test_settings_audio_config():
    assert SAMPLE_RATE == 16000
    assert CHUNK_SIZE > 0
    assert HALF_DUPLEX is True
    assert QUEUE_MAX_SIZE >= 10
