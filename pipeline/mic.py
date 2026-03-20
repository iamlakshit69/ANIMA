import asyncio
import time
import numpy as np
import pyaudio

from config.settings import (
    SAMPLE_RATE, CHUNK_SIZE, CHANNELS,
    SILENCE_DURATION,
)
from core.queues import audio_queue
from core.events import interrupt_event, assistant_speaking
import core.events as ev
from core.sentinel import SILENCE_MARKER

ECHO_DECAY_PAD   = 0.5
BARGE_IN_FRAMES  = 8
POST_SPEECH_MUTE = 1.5
ENERGY_THRESHOLD = 0.01   # RMS threshold — tune up if too sensitive, down if missing speech


def _read_mic_blocking(stream):
    raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


async def microphone_stream():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    silence_chunks = 0
    silence_limit  = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    speech_frames  = 0

    print("[mic] listening...")

    try:
        while True:
            chunk = await asyncio.to_thread(_read_mic_blocking, stream)
            now = time.monotonic()

            rms = np.sqrt(np.mean(chunk ** 2))
            is_speech = rms > ENERGY_THRESHOLD

            if is_speech:
                silence_chunks = 0
                speech_frames += 1

                if assistant_speaking.is_set():
                    cooldown = ev.current_phrase_duration + ECHO_DECAY_PAD
                    elapsed  = now - ev.speaking_started_at
                    if elapsed > cooldown and speech_frames >= BARGE_IN_FRAMES:
                        interrupt_event.set()
                else:
                    since_ended = now - ev.speaking_ended_at
                    if since_ended >= POST_SPEECH_MUTE:
                        await audio_queue.put(chunk)
            else:
                speech_frames = 0
                silence_chunks += 1
                if silence_chunks >= silence_limit:
                    since_ended = now - ev.speaking_ended_at
                    if since_ended >= POST_SPEECH_MUTE:
                        await audio_queue.put(SILENCE_MARKER)
                    silence_chunks = 0
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
