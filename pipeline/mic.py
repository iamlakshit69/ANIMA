import asyncio
import numpy as np
import pyaudio
from silero_vad import load_silero_vad, get_speech_timestamps

from config.settings import (
    SAMPLE_RATE,
    CHUNK_SIZE,
    CHANNELS,
    VAD_THRESHOLD,
    SILENCE_DURATION,
)
from core.queues import audio_queue
from core.events import interrupt_event, assistant_speaking
from core.sentinel import SILENCE_MARKER


async def microphone_stream():
    model = load_silero_vad()

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    silence_chunks = 0
    silence_limit = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)

    print("[mic] listening...")

    try:
        while True:
            raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            speech_prob = model(
                __import__("torch").tensor(chunk), SAMPLE_RATE
            ).item()

            if speech_prob > VAD_THRESHOLD:
                silence_chunks = 0

                if assistant_speaking.is_set():
                    interrupt_event.set()

                await audio_queue.put(chunk)

            else:
                silence_chunks += 1

                if silence_chunks >= silence_limit:
                    await audio_queue.put(SILENCE_MARKER)
                    silence_chunks = 0

            await asyncio.sleep(0)

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()