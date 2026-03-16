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
POST_SPEECH_MUTE = 2.0  # seconds to ignore mic after assistant finishes speaking


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

    silence_chunks   = 0
    silence_limit    = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    speech_frames    = 0
    barge_in_buffer  = []    # collects speech chunks while interrupt_event is set
                             # but assistant_speaking is still set
    pending_barge_in = []    # held until POST_SPEECH_MUTE expires so stt.py's
                             # BUFFER_MUTE_GUARD doesn't discard them
    was_speaking     = False # tracks assistant_speaking across iterations
    mute_was_active  = False # True while inside the post-speech mute window;
                             # cleared (and counters reset) exactly once on exit

    print("[mic] listening...")

    try:
        while True:
            chunk = await asyncio.to_thread(_read_mic_blocking, stream)
            now = time.monotonic()

            currently_speaking = assistant_speaking.is_set()

            # ── Transition: assistant just finished speaking ───────────────────
            # Move any barge-in speech to pending_barge_in; it will be flushed
            # into audio_queue once POST_SPEECH_MUTE expires. We cannot flush
            # immediately because stt.py's BUFFER_MUTE_GUARD (2.0 s) would
            # discard the buffer — speaking_ended_at was just stamped.
            if was_speaking and not currently_speaking:
                if barge_in_buffer:
                    pending_barge_in = barge_in_buffer[:]
                    print(f"[mic] {len(pending_barge_in)} barge-in chunks pending — "
                          f"will flush after mute window")
                barge_in_buffer  = []
                silence_chunks   = 0
                speech_frames    = 0
                mute_was_active  = True   # arm the mute window

            was_speaking = currently_speaking

            # ── Assistant is speaking ─────────────────────────────────────────
            if currently_speaking:
                speech_prob = model(torch.from_numpy(chunk), SAMPLE_RATE).item()

                if speech_prob > VAD_THRESHOLD:
                    speech_frames += 1
                    # Barge-in: only after the current phrase's echo has decayed
                    cooldown = ev.current_phrase_duration + ECHO_DECAY_PAD
                    elapsed  = now - ev.speaking_started_at
                    if elapsed > cooldown and speech_frames >= BARGE_IN_FRAMES:
                        interrupt_event.set()

                    # Once barge-in is confirmed start capturing speech so the
                    # user does not have to repeat themselves after the interrupt.
                    if interrupt_event.is_set():
                        barge_in_buffer.append(chunk)
                else:
                    speech_frames = 0
                # Never touch silence_chunks or audio_queue while assistant speaks
                continue

            # ── Post-speech mute window ───────────────────────────────────────
            since_ended = now - ev.speaking_ended_at
            if since_ended < POST_SPEECH_MUTE:
                # Skip VAD entirely — all audio here is room echo.
                mute_was_active = True
                continue

            # Mute just expired — boolean flag guarantees this block runs
            # exactly once regardless of event-loop timing or system load.
            # The old approach (since_ended < POST_SPEECH_MUTE + 0.1) was a
            # ~3-frame window that could be skipped entirely under load.
            if mute_was_active:
                silence_chunks  = 0
                speech_frames   = 0
                mute_was_active = False

                # Flush barge-in speech now that the echo decay window has
                # passed. stt.py's BUFFER_MUTE_GUARD (2.0 s) will also have
                # elapsed by this point so the buffer won't be discarded there.
                if pending_barge_in:
                    print(f"[mic] flushing {len(pending_barge_in)} barge-in chunks to audio_queue")
                    for buffered_chunk in pending_barge_in:
                        await audio_queue.put(buffered_chunk)
                    await audio_queue.put(SILENCE_MARKER)
                    pending_barge_in = []

            # ── Normal listening ──────────────────────────────────────────────
            speech_prob = model(torch.from_numpy(chunk), SAMPLE_RATE).item()

            if speech_prob > VAD_THRESHOLD:
                silence_chunks = 0
                speech_frames += 1
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