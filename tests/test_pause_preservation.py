import pytest
from config import settings

def test_settings_for_full_sentences():
    assert settings.SILENCE_DURATION >= 1.4, "Silence duration should allow natural inter-clause thinking pauses"
    assert settings.VAD_CONTINUATION_THRESHOLD < settings.VAD_THRESHOLD, "VAD should use lower continuation threshold to avoid cutting off soft syllables"
    assert settings.MIN_AUDIO_DURATION >= 0.4
    assert settings.GROQ_MAX_TOKENS <= 100, "Max tokens should be concise so voice assistant doesn't block the microphone"
