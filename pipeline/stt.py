import asyncio
import io
import wave
import time
import numpy as np
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

SILENCE_RMS      = 0.01  # RMS below which a chunk is considered silent
MIN_AUDIO_ENERGY = 0.02  # minimum RMS of entire buffer — below this Whisper hallucinates

# Discard buffer if collected within this many seconds of assistant finishing.
# Catches echo chunks that sneak through mic.py's mute window during the
# sd.play() gap. Must be >= POST_SPEECH_MUTE in mic.py to be effective.
# Previously 2.0s — raised to 4.0s because long responses need 4s+ to decay;
# a 2s guard meant the second half of the echo passed through unblocked,
# causing hallucinated transcripts ("Thanks for watching!", "you", etc.)
BUFFER_MUTE_GUARD = 0.5


def _to_wav_bytes(audio_data):
    """Convert float32 numpy array to WAV bytes using stdlib — no soundfile needed."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32768).astype(np.int16).tobytes())
    buf.seek(0)
    return buf


def _trim_trailing_silence(audio_buffer):
    """
    Single-pass trim — removes silent tail before sending to Groq Whisper.
    Every buffer ends with ~SILENCE_DURATION of dead audio (Silero VAD fires
    SILENCE_MARKER after the silence window). Trimming it reduces upload size
    and Whisper processing time.
    """
    if not audio_buffer:
        return audio_buffer

    last_active = 0
    for i, chunk in enumerate(audio_buffer):
        if np.sqrt(np.mean(chunk ** 2)) > SILENCE_RMS:
            last_active = i

    # Keep one silent chunk after last speech for natural trailing edge
    end = min(last_active + 2, len(audio_buffer))
    return audio_buffer[:end]


async def _transcribe_groq(client, audio_data):
    """Convert numpy audio to WAV bytes in memory and send to Groq API."""
    buf = _to_wav_bytes(audio_data)
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
            # Stamp the moment SILENCE_MARKER is dequeued — before any guards
            # or preprocessing — so the latency clock starts at the true
            # pipeline boundary, not after trimming and concatenation.
            silence_received_at = time.monotonic()

            if len(audio_buffer) == 0:
                continue

            # Guard 1 — time based: discard buffer collected too soon after
            # assistant finished speaking.
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
            # discarded buffers should not start the clock.
            ev.user_stopped_speaking_at = silence_received_at

            stt_start = time.monotonic()
            try:
                transcript = await _transcribe_groq(client, audio_data)
            except Exception as e:
                print(f"[stt] groq error: {e}")
                continue
            ev.stt_done_at = time.monotonic()

            stt_took = ev.stt_done_at - stt_start
            print(f"[stt] took {stt_took:.2f}s — transcript: {transcript}")

            # Check interrupt after the Groq call returns — the network round
            # trip can take 1-2s. A barge-in that fires during that window
            # sets interrupt_event, but without this check the stale transcript
            # still flows to text_queue and gets processed by llm.py.
            if interrupt_event.is_set():
                print("[stt] discarding transcript — interrupt fired during transcription")
                continue

            if transcript:
                await text_queue.put(transcript)

        else:
            audio_buffer.append(chunk)