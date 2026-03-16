import asyncio
import io
import wave
import time
import numpy as np
from groq import AsyncGroq

from config.settings import (
    GROQ_API_KEY,
    GROQ_WHISPER_MODEL,
    WHISPER_LANGUAGE,
    SAMPLE_RATE,
)
from core.queues import audio_queue, text_queue
from core.events import interrupt_event
import core.events as ev
from core.sentinel import SILENCE_MARKER

# RMS threshold below which a chunk is considered silence
# Used to trim trailing silence before sending to Groq
SILENCE_RMS = 0.01


def _to_wav_bytes(audio_data):
    """Convert float32 numpy array to WAV bytes using stdlib — no soundfile needed."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32768).astype(np.int16).tobytes())
    buf.seek(0)
    return buf


def _trim_trailing_silence(audio_buffer):
    """
    Remove trailing silent chunks from the buffer before transcription.
    Silero VAD fires SILENCE_MARKER after SILENCE_DURATION seconds of silence,
    so every buffer ends with ~0.4s of dead audio. Trimming it reduces
    the WAV file size and Whisper processing time.
    """
    if not audio_buffer:
        return audio_buffer

    rms_values = [np.sqrt(np.mean(chunk ** 2)) for chunk in audio_buffer]

    # Find the last chunk with meaningful audio
    last_active = 0
    for i, rms in enumerate(rms_values):
        if rms > SILENCE_RMS:
            last_active = i

    # Keep one silent chunk after last speech for natural trailing edge
    end = min(last_active + 2, len(audio_buffer))
    return audio_buffer[:end]


async def _transcribe_groq(client, audio_data):
    """Convert numpy audio to WAV bytes in memory and send to Groq API."""
    buf = _to_wav_bytes(audio_data)

    result = await client.audio.transcriptions.create(
        model=GROQ_WHISPER_MODEL,
        file=("audio.wav", buf, "audio/wav"),
        language=WHISPER_LANGUAGE,
    )
    return result.text.strip()


async def speech_to_text_stream():
    client = AsyncGroq(api_key=GROQ_API_KEY)
    audio_buffer = []

    print("[stt] ready...")

    while True:
        chunk = await audio_queue.get()

        if interrupt_event.is_set():
            audio_buffer = []
            continue

        if chunk is SILENCE_MARKER:
            if len(audio_buffer) == 0:
                continue

            # Stamp the moment the user stopped speaking.
            # speaker.py reads this to compute total pipeline latency.
            ev.user_stopped_speaking_at = time.monotonic()

            # Trim trailing silence — reduces upload size and Whisper time
            trimmed = _trim_trailing_silence(audio_buffer)
            audio_buffer = []

            audio_data = np.concatenate(trimmed)

            try:
                transcript = await _transcribe_groq(client, audio_data)
            except Exception as e:
                print(f"[stt] groq error: {e}")
                continue

            if transcript:
                print(f"[stt] transcript: {transcript}")
                await text_queue.put(transcript)

        else:
            audio_buffer.append(chunk)