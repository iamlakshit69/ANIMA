import asyncio
import io
import time
import numpy as np
import soundfile as sf
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


async def _transcribe_groq(client, audio_data):
    """Convert numpy audio to WAV bytes in memory and send to Groq API."""
    buf = io.BytesIO()
    sf.write(buf, audio_data, SAMPLE_RATE, format='WAV')
    buf.seek(0)

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

            audio_data = np.concatenate(audio_buffer)
            audio_buffer = []

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