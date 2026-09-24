import asyncio
import time
import numpy as np
from faster_whisper import WhisperModel

from config.settings import (
    WHISPER_LANGUAGE,
    SAMPLE_RATE,
)
from core.queues import audio_queue, text_queue
from core.events import interrupt_event
import core.events as ev
from core.sentinel import SILENCE_MARKER

WHISPER_MODEL_SIZE = "distil-medium.en"
WHISPER_DEVICE     = "cpu"
WHISPER_COMPUTE    = "int8"

SILENCE_RMS        = 0.01   # RMS below which a chunk is considered silent
MIN_AUDIO_ENERGY   = 0.02   # minimum RMS of entire buffer — below this Whisper hallucinates

# Bug #5 fix: value raised from 2.0 → 4.0 to match the comment's own warning.
# A 2-second guard against a 4-second echo decay window means the second half
# of the echo passes both guards and reaches Whisper unblocked, causing
# hallucinated transcripts ("Thanks for watching!", "you", etc.) after any
# response longer than ~1 sentence.
BUFFER_MUTE_GUARD  = 4.0    # discard buffer if collected within this many seconds
                             # of assistant finishing — long responses need 4s+ to decay


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
    print(f"[stt] loading whisper {WHISPER_MODEL_SIZE} on GPU...")
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )
    audio_buffer = []

    print("[stt] ready...")

    while True:
        chunk = await audio_queue.get()

        if interrupt_event.is_set():
            audio_buffer = []
            continue

        if chunk is SILENCE_MARKER:
            # Bug #9 fix: stamp user_stopped_speaking_at HERE — the moment the
            # SILENCE_MARKER is dequeued — not after guards and preprocessing.
            # Previously it was stamped just before stt_start, meaning all
            # guard checks, _trim_trailing_silence, and np.concatenate were
            # silently excluded from the latency measurement, making total
            # latency appear artificially lower than the real end-to-end time.
            silence_received_at = time.monotonic()

            if len(audio_buffer) == 0:
                continue

            # Guard 1 — time based: discard buffer collected too soon after
            # assistant finished speaking. Catches echo chunks that sneak
            # through mic.py's mute window during the sd.play() gap.
            since_ended = silence_received_at - ev.speaking_ended_at
            if since_ended < BUFFER_MUTE_GUARD:
                print(f"[stt] discarding stale buffer ({since_ended:.2f}s since speaking ended)")
                audio_buffer = []
                continue

            trimmed = _trim_trailing_silence(audio_buffer)
            audio_buffer = []

            if not trimmed:
                continue

            audio_data = np.concatenate(trimmed)

            # Guard 2 — energy based: reject near-silence audio.
            # Whisper hallucinates plausible phrases ("Thanks for watching!",
            # "you", etc.) when given very quiet input. A real human voice
            # has measurably higher RMS than room echo or residual noise.
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < MIN_AUDIO_ENERGY:
                print(f"[stt] rejecting low energy audio (rms={rms:.4f} < {MIN_AUDIO_ENERGY})")
                continue

            # Commit the latency timestamp only after both guards pass —
            # buffers that get discarded above should not start the clock.
            ev.user_stopped_speaking_at = silence_received_at

            stt_start = time.monotonic()
            transcript = await asyncio.to_thread(_transcribe, model, audio_data)
            ev.stt_done_at = time.monotonic()

            stt_took = ev.stt_done_at - stt_start
            print(f"[stt] took {stt_took:.2f}s — transcript: {transcript}")

            # Bug #8 fix: check interrupt_event after Whisper returns.
            # _transcribe runs in a thread and can take 1-3 seconds on a
            # mid-range GPU. A barge-in that fires during that window sets
            # interrupt_event, but without this check the stale transcript
            # from the interrupted turn still flows to text_queue and then
            # into llm.py — adding latency and potentially being processed.
            if interrupt_event.is_set():
                print(f"[stt] discarding transcript — interrupt fired during transcription")
                continue

            if transcript:
                await text_queue.put(transcript)

        else:
            audio_buffer.append(chunk)