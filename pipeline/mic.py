import asyncio
from collections import deque
import time
import numpy as np
import sounddevice as sd

from config.settings import (
    SAMPLE_RATE, CHUNK_SIZE, CHANNELS,
    SILENCE_DURATION, ENERGY_THRESHOLD, VAD_THRESHOLD,
    VAD_CONTINUATION_THRESHOLD,
    HALF_DUPLEX, ECHO_DECAY_PAD, BARGE_IN_FRAMES,
    POST_SPEECH_MUTE, MIC_DEVICE, MIN_SPEECH_FRAMES,
)
import core.queues as queues
from core.state import playback_state
from core.turn import turn_controller
from core.sentinel import SILENCE_MARKER
from core.logger import get_logger

logger = get_logger("anima.mic")

try:
    import torch
    from silero_vad import load_silero_vad
    _silero_vad_model = load_silero_vad()
except Exception as _vad_err:
    _silero_vad_model = None


async def microphone_stream():
    """Capture mic audio with sounddevice, detect speech/silence via Silero VAD (with RMS fallback),
    preserve complete sentence audio across natural pauses, and tag chunks with monotonically
    increasing generation IDs from turn_controller.
    """
    try:
        in_dev = sd.query_devices(MIC_DEVICE, kind="input")
        logger.info(
            "[mic] using input device #%s: %s (sample_rate=%d, channels=%d)",
            in_dev.get("index", "default"), in_dev.get("name"), SAMPLE_RATE, CHANNELS
        )
    except Exception as e:
        logger.warning("[mic] could not query input device %s: %s", MIC_DEVICE, e)

    loop = asyncio.get_running_loop()
    raw_q = asyncio.Queue(maxsize=100)

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.debug("[mic] sounddevice status: %s", status)
        chunk = indata[:, 0].copy()
        try:
            loop.call_soon_threadsafe(raw_q.put_nowait, chunk)
        except asyncio.QueueFull:
            pass

    silence_chunks = 0
    silence_limit  = max(1, int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE))
    speech_frames  = 0
    candidate_chunks: list[np.ndarray] = []
    # Ring buffer of recent pre-speech audio frames (~190ms) to ensure leading phonemes are preserved
    preroll_buffer: deque[np.ndarray] = deque(maxlen=6)
    is_in_turn = False

    vad_backend = "Silero VAD" if _silero_vad_model is not None else "RMS Energy"
    logger.info(
        "[mic] listening with sounddevice (%s, half_duplex=%s, silence_duration=%.1fs)...",
        vad_backend, HALF_DUPLEX, SILENCE_DURATION
    )

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        device=MIC_DEVICE,
        callback=audio_callback,
    ):
        while True:
            chunk = await raw_q.get()
            now = time.monotonic()

            # ── Assistant is currently speaking ───────────────────────────────
            if playback_state.speaking.is_set():
                if HALF_DUPLEX:
                    # In half-duplex mode: mute mic audio entirely during speech.
                    # This completely eliminates room echo and prevents self-interruption.
                    silence_chunks = 0
                    speech_frames = 0
                    candidate_chunks = []
                    preroll_buffer.clear()
                    is_in_turn = False
                    if _silero_vad_model is not None:
                        try:
                            _silero_vad_model.reset_states()
                        except Exception:
                            pass
                    continue
                else:
                    # Fallback acoustic barge-in heuristic for full-duplex testing
                    rms = np.sqrt(np.mean(chunk ** 2))
                    if rms > ENERGY_THRESHOLD:
                        speech_frames += 1
                        elapsed = now - playback_state.started_at
                        cooldown = playback_state.phrase_duration + ECHO_DECAY_PAD
                        if elapsed > cooldown and speech_frames >= BARGE_IN_FRAMES:
                            logger.info("[mic] acoustic barge-in detected! Bumping turn.")
                            await turn_controller.bump()
                            playback_state.mark_interrupted()
                    else:
                        speech_frames = 0
                    continue

            # ── Post-speech mute cooldown (if not half-duplex) ─────────────────
            if not HALF_DUPLEX:
                since_ended = now - playback_state.ended_at
                if since_ended < POST_SPEECH_MUTE:
                    continue

            # ── Voice Activity Detection (Silero VAD with hysteresis) ─────────
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            active_thresh = VAD_CONTINUATION_THRESHOLD if is_in_turn else VAD_THRESHOLD

            if _silero_vad_model is not None:
                try:
                    prob = _silero_vad_model(torch.from_numpy(chunk), SAMPLE_RATE).item()
                    is_speech = prob > active_thresh
                except Exception:
                    is_speech = rms > (ENERGY_THRESHOLD * 0.7 if is_in_turn else ENERGY_THRESHOLD)
            else:
                is_speech = rms > (ENERGY_THRESHOLD * 0.7 if is_in_turn else ENERGY_THRESHOLD)

            if is_speech:
                silence_chunks = 0
                speech_frames += 1

                if not is_in_turn:
                    candidate_chunks.append(chunk)
                    # Require sustained energy across MIN_SPEECH_FRAMES before declaring real speech onset
                    if speech_frames >= MIN_SPEECH_FRAMES:
                        # Only start a new turn generation if previous turn completed or barge-in
                        if turn_controller.current == 0 or playback_state.last_turn_completed or playback_state.speaking.is_set():
                            new_gen = await turn_controller.bump()
                            playback_state.last_turn_completed = False
                            playback_state.interrupted = False
                            logger.info(
                                "[mic] speech onset confirmed (rms=%.4f, frames=%d) — turn gen %d started",
                                rms, speech_frames, new_gen
                            )
                        else:
                            new_gen = turn_controller.current
                            logger.info(
                                "[mic] speech continued (rms=%.4f, frames=%d) — continuing gen %d",
                                rms, speech_frames, new_gen
                            )
                        is_in_turn = True

                        # Forward pre-roll audio frames so the initial syllable is never clipped
                        for pre_chunk in preroll_buffer:
                            await queues.audio_queue.put((new_gen, pre_chunk))
                        preroll_buffer.clear()

                        # Forward candidate chunks that confirmed speech onset
                        for buffered_chunk in candidate_chunks:
                            await queues.audio_queue.put((new_gen, buffered_chunk))
                        candidate_chunks = []
                else:
                    current_gen = turn_controller.current
                    await queues.audio_queue.put((current_gen, chunk))

            else:
                if not is_in_turn:
                    # Ambient noise or brief click that did not sustain speech
                    speech_frames = 0
                    candidate_chunks = []
                    silence_chunks = 0
                    preroll_buffer.append(chunk)
                else:
                    # User is currently in a turn but paused between words or clauses
                    silence_chunks += 1
                    current_gen = turn_controller.current

                    # CRITICAL: Forward silence chunks during an active turn so that inter-word
                    # pauses and soft trailing phonemes remain naturally in the recorded stream.
                    await queues.audio_queue.put((current_gen, chunk))

                    if silence_chunks >= silence_limit:
                        logger.info(
                            "[mic] speech ended for gen %d (silence frames=%d) — emitting SILENCE_MARKER",
                            current_gen, silence_chunks
                        )
                        await queues.audio_queue.put((current_gen, SILENCE_MARKER))
                        silence_chunks = 0
                        speech_frames = 0
                        is_in_turn = False
                        candidate_chunks = []
                        preroll_buffer.clear()
                        if _silero_vad_model is not None:
                            try:
                                _silero_vad_model.reset_states()
                            except Exception:
                                pass
