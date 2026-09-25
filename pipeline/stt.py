import asyncio
import time
import numpy as np
from faster_whisper import WhisperModel

from config.settings import (
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
    SAMPLE_RATE,
    BUFFER_MUTE_GUARD,
)
from core.queues import audio_queue, text_queue
from core.events import interrupt_event
import core.events as ev
from core.sentinel import SILENCE_MARKER, INTERRUPT

WHISPER_DEVICE     = "cpu"
WHISPER_COMPUTE    = "int8"

SILENCE_RMS        = 0.01   # RMS below which a chunk is considered silent
MIN_AUDIO_ENERGY   = 0.02   # minimum RMS of entire buffer — below this Whisper hallucinates


def _trim_trailing_silence(audio_buffer):
    """Single-pass trim — removes silent tail before sending to Whisper."""
    if not audio_buffer:
        return audio_buffer

    last_active = 0
    for i, chunk in enumerate(audio_buffer):
        if np.sqrt(np.mean(chunk ** 2)) > SILENCE_RMS:
            last_active = i

    end = min(last_active + 2, len(audio_buffer))
    return audio_buffer[:end]


def _transcribe(model, audio_data):
    segments, _ = model.transcribe(
        audio_data,
        language=WHISPER_LANGUAGE,
        beam_size=1,
        vad_filter=False,                 # already filtered by mic.py Silero VAD
        condition_on_previous_text=False, # independent per utterance — faster
        temperature=0,                    # greedy decoding — no sampling overhead
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


async def speech_to_text_stream():
    print(f"[stt] loading whisper {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE.upper()} ({WHISPER_COMPUTE})...")
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )
    audio_buffer = []
    is_barge_in = False

    print("[stt] ready...")

    while True:
        chunk = await audio_queue.get()

        if chunk is INTERRUPT:
            audio_buffer = []
            is_barge_in = True
            print("[stt] interrupt received — buffer cleared, barge-in armed")
            continue

        # If an interrupt is active and this is NOT barge-in speech, discard chunk
        if interrupt_event.is_set() and not is_barge_in:
            audio_buffer = []
            continue

        if chunk is SILENCE_MARKER:
            silence_received_at = time.monotonic()

            if len(audio_buffer) == 0:
                is_barge_in = False
                continue

            # Guard 1 — time based: discard buffer collected too soon after
            # assistant finished speaking, UNLESS this buffer is confirmed barge-in speech.
            since_ended = silence_received_at - ev.speaking_ended_at
            if not is_barge_in and since_ended < BUFFER_MUTE_GUARD:
                print(f"[stt] discarding stale buffer ({since_ended:.2f}s since speaking ended)")
                audio_buffer = []
                continue

            trimmed = _trim_trailing_silence(audio_buffer)
            audio_buffer = []

            # Reset barge-in state for next utterance
            was_barge_in = is_barge_in
            is_barge_in = False

            if not trimmed:
                continue

            audio_data = np.concatenate(trimmed)

            # Guard 2 — energy based: reject near-silence audio.
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < MIN_AUDIO_ENERGY:
                print(f"[stt] rejecting low energy audio (rms={rms:.4f} < {MIN_AUDIO_ENERGY})")
                continue

            # Commit the latency timestamp
            ev.user_stopped_speaking_at = silence_received_at

            int_generation = ev.interrupt_counter
            stt_start = time.monotonic()
            transcript = await asyncio.to_thread(_transcribe, model, audio_data)
            ev.stt_done_at = time.monotonic()

            stt_took = ev.stt_done_at - stt_start
            print(f"[stt] took {stt_took:.2f}s — transcript: {transcript}")

            # Check if a new interrupt fired while Whisper was computing in the worker thread
            if ev.interrupt_counter != int_generation:
                print("[stt] discarding transcript — new interrupt fired during transcription")
                continue

            if transcript:
                await text_queue.put(transcript)

        else:
            audio_buffer.append(chunk)