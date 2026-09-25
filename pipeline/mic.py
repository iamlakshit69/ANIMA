import asyncio
import collections
import time
import numpy as np
import pyaudio
import torch
from silero_vad import load_silero_vad

from config.settings import (
    SAMPLE_RATE, CHUNK_SIZE, CHANNELS,
    VAD_THRESHOLD, SILENCE_DURATION,
    BARGE_IN_PREROLL_SECONDS, BARGE_IN_FRAMES,
    ECHO_DECAY_PAD, POST_SPEECH_MUTE,
)
from core.queues import audio_queue
from core.events import interrupt_event, assistant_speaking
import core.events as ev
from core.sentinel import SILENCE_MARKER, INTERRUPT


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
    silence_limit = max(1, int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE))
    speech_frames = 0
    is_user_speaking = False

    # Rolling pre-roll buffer to preserve audio prior to VAD confirmation
    preroll_limit = max(1, int(BARGE_IN_PREROLL_SECONDS * SAMPLE_RATE / CHUNK_SIZE))
    preroll_buffer = collections.deque(maxlen=preroll_limit)

    was_speaking = False
    mute_was_active = False

    print("[mic] listening...")

    try:
        while True:
            chunk = await asyncio.to_thread(_read_mic_blocking, stream)
            now = time.monotonic()

            currently_speaking = assistant_speaking.is_set()

            # ── Transition: assistant just finished speaking ───────────────────
            if was_speaking and not currently_speaking:
                if not is_user_speaking:
                    mute_was_active = True
                silence_chunks = 0
                speech_frames = 0

            was_speaking = currently_speaking

            # ── Assistant is speaking ─────────────────────────────────────────
            if currently_speaking:
                preroll_buffer.append(chunk)
                speech_prob = model(torch.from_numpy(chunk), SAMPLE_RATE).item()

                if speech_prob > VAD_THRESHOLD:
                    speech_frames += 1
                    cooldown = ev.current_phrase_duration + ECHO_DECAY_PAD
                    elapsed = now - ev.speaking_started_at

                    # Barge-in: triggered once speech frames exceed threshold beyond cooldown
                    if elapsed > cooldown and speech_frames >= BARGE_IN_FRAMES:
                        if not interrupt_event.is_set():
                            ev.interrupt_counter += 1
                            interrupt_event.set()
                            print("[mic] barge-in triggered! Setting interrupt_event and sending pre-roll")
                            # Notify STT to clear stale audio and expect barge-in
                            await audio_queue.put(INTERRUPT)
                            # Transfer all pre-roll audio so beginning of user's speech is intact
                            while preroll_buffer:
                                await audio_queue.put(preroll_buffer.popleft())
                            is_user_speaking = True
                            silence_chunks = 0

                    if interrupt_event.is_set() and is_user_speaking:
                        await audio_queue.put(chunk)
                else:
                    speech_frames = 0
                    if interrupt_event.is_set() and is_user_speaking:
                        silence_chunks += 1
                        if silence_chunks >= silence_limit:
                            await audio_queue.put(SILENCE_MARKER)
                            is_user_speaking = False
                            silence_chunks = 0
                continue

            # ── Post-speech mute window (room echo decay) ──────────────────────
            # Only apply echo suppression if user is not already actively speaking (e.g. from barge-in)
            if not is_user_speaking:
                since_ended = now - ev.speaking_ended_at
                if since_ended < POST_SPEECH_MUTE:
                    mute_was_active = True
                    continue

            if mute_was_active:
                silence_chunks = 0
                speech_frames = 0
                mute_was_active = False

            # ── Normal listening ──────────────────────────────────────────────
            speech_prob = model(torch.from_numpy(chunk), SAMPLE_RATE).item()

            if speech_prob > VAD_THRESHOLD:
                if not is_user_speaking:
                    is_user_speaking = True
                    # Flush pre-roll chunks captured during silence to preserve initial phonemes
                    while preroll_buffer:
                        await audio_queue.put(preroll_buffer.popleft())
                silence_chunks = 0
                speech_frames += 1
                await audio_queue.put(chunk)
            else:
                speech_frames = 0
                if is_user_speaking:
                    silence_chunks += 1
                    if silence_chunks >= silence_limit:
                        await audio_queue.put(SILENCE_MARKER)
                        is_user_speaking = False
                        silence_chunks = 0
                else:
                    # User is silent. Keep pre-roll buffer updated
                    preroll_buffer.append(chunk)

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()