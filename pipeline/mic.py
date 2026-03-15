import asyncio
import time
import numpy as np
import pyaudio
import torch
from silero_vad import load_silero_vad

from config.settings import (
    SAMPLE_RATE, CHUNK_SIZE, CHANNELS,
    VAD_THRESHOLD, SILENCE_DURATION,
)
from core.queues import audio_queue
from core.events import interrupt_event, assistant_speaking
import core.events as ev
from core.sentinel import SILENCE_MARKER

# After the phrase ends, room echo rings for roughly this long.
# Cooldown = phrase_duration + ECHO_DECAY_PAD, computed per phrase.
ECHO_DECAY_PAD = 0.5   # seconds of room echo decay after phrase ends

# Consecutive VAD-positive frames required after cooldown to confirm real voice.
# ~8 frames x 32ms = ~256ms of sustained speech.
BARGE_IN_FRAMES = 8


def _read_mic_blocking(stream):
    raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


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
    silence_limit  = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    speech_frames  = 0

    print("[mic] listening...")

    try:
        while True:
            chunk = await asyncio.to_thread(_read_mic_blocking, stream)
            speech_prob = model(torch.tensor(chunk), SAMPLE_RATE).item()

            if speech_prob > VAD_THRESHOLD:
                silence_chunks = 0
                speech_frames += 1

                if assistant_speaking.is_set():
                    # Dynamic cooldown: covers the full duration of the phrase
                    # currently playing plus a decay buffer for room echo.
                    # Fixed cooldowns fail on long phrases - their own echo
                    # easily outlasts a static 1.2s window.
                    cooldown = ev.current_phrase_duration + ECHO_DECAY_PAD
                    elapsed  = time.monotonic() - ev.speaking_started_at

                    if elapsed > cooldown and speech_frames >= BARGE_IN_FRAMES:
                        interrupt_event.set()
                    # Never queue echo into the STT audio buffer
                else:
                    await audio_queue.put(chunk)

            else:
                speech_frames = 0
                silence_chunks += 1
                if silence_chunks >= silence_limit:
                    await audio_queue.put(SILENCE_MARKER)
                    silence_chunks = 0

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()