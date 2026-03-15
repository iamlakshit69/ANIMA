# pipeline/stt.py

import asyncio
import numpy as np
from faster_whisper import WhisperModel
from config.settings import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    SAMPLE_RATE,
)
from core.queues import audio_queue, text_queue
from core.events import interrupt_event, assistant_speaking
from core.sentinel import SILENCE_MARKER

async def speech_to_text_stream():
    model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE)
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

            audio_data = np.concatenate(audio_buffer)
            audio_buffer = []

            # ✅ Run blocking Whisper inference in a thread
            transcript = await asyncio.to_thread(_transcribe, model, audio_data)

            if transcript:
                print(f"[stt] transcript: {transcript}")
                await text_queue.put(transcript)

        else:
            audio_buffer.append(chunk)


def _transcribe(model, audio_data):
    """Synchronous helper — runs in thread pool via asyncio.to_thread."""
    segments, _ = model.transcribe(
        audio_data,
        language=WHISPER_LANGUAGE,
        beam_size=1,
        vad_filter=True,
    )
    # Iterate the generator HERE in the thread, not on the event loop
    return " ".join(segment.text.strip() for segment in segments).strip()