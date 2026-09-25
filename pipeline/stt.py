import asyncio
import time
import numpy as np
from faster_whisper import WhisperModel

from config.settings import (
    LOCAL_WHISPER_MODEL,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    SAMPLE_RATE,
    SILENCE_RMS,
    MIN_AUDIO_ENERGY,
    MIN_AUDIO_DURATION,
    BUFFER_MUTE_GUARD,
    HALF_DUPLEX,
)
import core.queues as queues
from core.state import playback_state, TurnLatency
from core.turn import turn_controller
from core.sentinel import SILENCE_MARKER
from core.logger import get_logger

import io
import wave
logger = get_logger("anima.stt")


def _to_wav_bytes(audio_data: np.ndarray) -> io.BytesIO:
    """Convert float32 numpy array to WAV bytes in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32768.0).astype(np.int16).tobytes())
    buf.seek(0)
    return buf


def _trim_trailing_silence(audio_buffer: list[np.ndarray]) -> list[np.ndarray]:
    """Trim dead silence while preserving natural speech decay and unvoiced trailing consonants."""
    if not audio_buffer:
        return audio_buffer

    last_active = 0
    for i, chunk in enumerate(audio_buffer):
        if np.sqrt(np.mean(chunk ** 2)) > SILENCE_RMS:
            last_active = i

    # Preserve 10 chunks (~320ms) after last active speech so word-final consonants aren't cut
    end = min(last_active + 10, len(audio_buffer))
    return audio_buffer[:end]


def _normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalize speech volume so quiet microphone recordings have full dynamic range for Whisper."""
    peak = float(np.max(np.abs(audio_data)))
    if peak > 0.01:
        return audio_data * (0.95 / peak)
    return audio_data


def _transcribe_local(model: WhisperModel, audio_data: np.ndarray) -> str:
    """Run fast on-device inference using Faster-Whisper."""
    norm_audio = _normalize_audio(audio_data)
    segments, _ = model.transcribe(
        norm_audio,
        language=WHISPER_LANGUAGE,
        beam_size=1,
        temperature=0.0,
        vad_filter=False,  # Silero VAD already applied upstream
    )
    return " ".join(seg.text for seg in segments).strip()


async def speech_to_text_stream():
    """Continuously receive audio chunks tagged with generation IDs, transcribe complete
    utterances via on-device Faster-Whisper, and forward transcripts with per-turn TurnLatency.
    """
    logger.info(
        "[stt] loading local Whisper model '%s' (device=%s, compute=%s)...",
        LOCAL_WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
    )
    model = WhisperModel(
        LOCAL_WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )

    # Warmup inference to eliminate initial JIT latency
    try:
        dummy = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        await asyncio.to_thread(lambda: list(model.transcribe(dummy, beam_size=1)[0]))
    except Exception as e:
        logger.warning("[stt] warmup failed: %s", e)

    audio_buffer: list[np.ndarray] = []
    current_buffer_gen: int | None = None

    logger.info("[stt] ready (local model: %s)...", LOCAL_WHISPER_MODEL)

    while True:
        item = await queues.audio_queue.get()

        # Support both (gen_id, chunk) and legacy bare chunk
        if isinstance(item, tuple) and len(item) == 2:
            gen_id, chunk = item
        else:
            gen_id = turn_controller.current
            chunk = item

        # Staleness check: if generation has been superseded by a newer turn / barge-in,
        # discard the audio buffer immediately.
        if gen_id != turn_controller.current:
            if audio_buffer:
                logger.debug("[stt] dropping stale audio buffer (chunk gen %d != current %d)", gen_id, turn_controller.current)
                audio_buffer = []
                current_buffer_gen = None
            continue

        if chunk is SILENCE_MARKER:
            silence_received_at = time.monotonic()

            if not audio_buffer:
                continue

            # In non-half-duplex mode: guard against echo collected right after speaking ended
            if not HALF_DUPLEX:
                since_ended = silence_received_at - playback_state.ended_at
                if since_ended < BUFFER_MUTE_GUARD:
                    logger.debug(
                        "[stt] discarding stale buffer (%.2fs since speaking ended < %.2fs guard)",
                        since_ended, BUFFER_MUTE_GUARD
                    )
                    audio_buffer = []
                    current_buffer_gen = None
                    continue

            trimmed = _trim_trailing_silence(audio_buffer)
            audio_buffer = []
            turn_gen = current_buffer_gen if current_buffer_gen is not None else gen_id
            current_buffer_gen = None

            if not trimmed:
                continue

            audio_data = np.concatenate(trimmed)
            duration = len(audio_data) / SAMPLE_RATE

            if duration < MIN_AUDIO_DURATION:
                logger.debug(
                    "[stt] rejecting short audio (duration=%.2fs < %.2fs)",
                    duration, MIN_AUDIO_DURATION
                )
                continue

            # Energy guard: reject near-silence or ambient room hiss
            rms = float(np.sqrt(np.mean(audio_data ** 2)))
            if rms < MIN_AUDIO_ENERGY:
                logger.debug(
                    "[stt] rejecting low energy audio (rms=%.4f < %.4f)",
                    rms, MIN_AUDIO_ENERGY
                )
                continue

            # Check staleness before transcription
            if turn_gen != turn_controller.current and playback_state.interrupted:
                logger.debug("[stt] turn %d was superseded by barge-in before transcription", turn_gen)
                continue

            # Create fresh per-turn latency instance
            latency = TurnLatency(user_stopped_speaking_at=silence_received_at)
            stt_start = time.monotonic()

            try:
                transcript = await asyncio.to_thread(_transcribe_local, model, audio_data)
            except Exception as e:
                logger.error("[stt] local Whisper error for gen %d: %s", turn_gen, e)
                continue

            latency.stt_done_at = time.monotonic()
            stt_took = latency.stt_done_at - stt_start
            logger.info("[stt] gen=%d took %.2fs (audio=%.2fs) — transcript: '%s'", turn_gen, stt_took, duration, transcript)

            # Check staleness after transcription: only discard if interrupted by barge-in
            if turn_gen != turn_controller.current and playback_state.interrupted:
                logger.info(
                    "[stt] discarding transcript for gen %d — superseded by %d via barge-in",
                    turn_gen, turn_controller.current
                )
                continue

            clean_text = transcript.strip()
            # Guard against spurious Whisper silence/noise artifacts on short or faint audio
            _lower = clean_text.lower().rstrip(".!?,")
            if _lower in {"you", "thank you", "bye", "thanks for watching", "subtitles by"} and duration < 1.2:
                logger.debug("[stt] discarding Whisper noise artifact '%s' (duration=%.2fs)", clean_text, duration)
                continue

            if clean_text:
                await queues.text_queue.put((turn_gen, clean_text, latency))

        else:
            if current_buffer_gen is None:
                current_buffer_gen = gen_id
            audio_buffer.append(chunk)