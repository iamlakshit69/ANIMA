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

ECHO_DECAY_PAD   = 0.5  # extra buffer after phrase ends for room echo decay
BARGE_IN_FRAMES  = 8    # consecutive VAD frames needed to trigger barge-in
POST_SPEECH_MUTE = 1.5  # seconds to ignore mic after assistant finishes speaking


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

            # from_numpy shares memory — no copy vs torch.tensor()
            speech_prob = model(torch.from_numpy(chunk), SAMPLE_RATE).item()

            # single syscall per iteration instead of two
            now = time.monotonic()

            if speech_prob > VAD_THRESHOLD:
                silence_chunks = 0
                speech_frames += 1

                if assistant_speaking.is_set():
                    # Barge-in: only after the current phrase's echo has decayed
                    cooldown = ev.current_phrase_duration + ECHO_DECAY_PAD
                    elapsed  = now - ev.speaking_started_at
                    if elapsed > cooldown and speech_frames >= BARGE_IN_FRAMES:
                        interrupt_event.set()
                    # Never queue echo into STT while assistant is speaking

                else:
                    # Post-speech mute: ignore mic for a brief window after
                    # assistant finishes — room echo still rings after END_OF_SPEECH
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
                    # reset regardless — avoids firing stale SILENCE_MARKER
                    # the moment mute window ends
                    silence_chunks = 0

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()