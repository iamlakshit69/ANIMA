import asyncio
import os
import io
import time
import wave
import numpy as np
from kokoro_onnx import Kokoro

from config.settings import (
    KOKORO_MODEL_PATH, KOKORO_VOICES_PATH, KOKORO_VOICE,
    KOKORO_SPEED, KOKORO_LANG,
    MIN_PHRASE_CHARS, MAX_PHRASE_CHARS, SAMPLE_RATE,
)
import core.queues as queues
from core.state import TurnLatency
from core.turn import turn_controller
from core.sentinel import END_OF_RESPONSE, END_OF_SPEECH
from core.retry import generate_error_tone
from core.logger import get_logger

logger = get_logger("anima.tts")


def _wav_bytes_to_numpy(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Convert raw WAV bytes to float32 numpy array and sample rate."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def _synthesize_local_say(text: str) -> tuple[np.ndarray, int]:
    """Fallback TTS using macOS built-in 'say' command when Kokoro TTS is unavailable."""
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        cmd = ["say", "-o", tmp_path, "--data-format=LEI16@16000", text]
        res = subprocess.run(cmd, capture_output=True, timeout=10)
        if res.returncode != 0:
            raise RuntimeError(f"say command failed: {res.stderr.decode()}")
        with wave.open(tmp_path, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sr
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def _accumulator(phrase_queue: asyncio.Queue):
    """Accumulate streamed tokens into phrases bounded by punctuation or length."""
    phrase_buffer = ""
    current_gen = None

    while True:
        item = await queues.token_queue.get()

        if isinstance(item, tuple):
            if len(item) == 3:
                gen_id, token, latency = item
            else:
                gen_id, token = item
                latency = TurnLatency()
        else:
            gen_id = turn_controller.current
            token = item
            latency = TurnLatency()

        # Staleness check: if generation was superseded, drop buffer immediately
        if gen_id != turn_controller.current:
            phrase_buffer = ""
            current_gen = None
            continue

        current_gen = gen_id

        if token is END_OF_RESPONSE:
            if phrase_buffer.strip() and gen_id == turn_controller.current:
                await phrase_queue.put((gen_id, phrase_buffer, latency))
            phrase_buffer = ""
            await phrase_queue.put((gen_id, END_OF_RESPONSE, latency))
            current_gen = None
            continue

        phrase_buffer += token

        # Split at natural clause/sentence boundaries or max char limit
        should_synthesize = (
            len(phrase_buffer) >= MIN_PHRASE_CHARS and phrase_buffer[-1] in ".!?,:"
        ) or len(phrase_buffer) >= MAX_PHRASE_CHARS

        if should_synthesize:
            await phrase_queue.put((gen_id, phrase_buffer, latency))
            phrase_buffer = ""


async def _synthesizer(kokoro: Kokoro, phrase_queue: asyncio.Queue):
    """Synthesize phrases with Kokoro TTS and enqueue audio chunks for playback."""
    first_phrase = True
    last_gen = None

    while True:
        item = await phrase_queue.get()

        if isinstance(item, tuple):
            if len(item) == 3:
                gen_id, phrase, latency = item
            else:
                gen_id, phrase = item
                latency = TurnLatency()
        else:
            gen_id = turn_controller.current
            phrase = item
            latency = TurnLatency()

        if gen_id != last_gen:
            first_phrase = True
            last_gen = gen_id

        # Staleness check: discard superseded turn
        if gen_id != turn_controller.current:
            first_phrase = True
            continue

        if phrase is END_OF_RESPONSE:
            await queues.tts_queue.put((gen_id, END_OF_SPEECH, latency))
            first_phrase = True
            continue

        logger.info("[tts] gen=%d synthesizing: '%s'", gen_id, phrase.strip())
        tts_start = time.monotonic()

        try:
            samples, sample_rate = await asyncio.to_thread(
                kokoro.create,
                phrase,
                voice=KOKORO_VOICE,
                speed=KOKORO_SPEED,
                lang=KOKORO_LANG,
            )
        except Exception as e:
            logger.warning("[tts] Kokoro TTS error (%s). Falling back to macOS speech synthesis...", e)
            try:
                samples, sample_rate = await asyncio.to_thread(_synthesize_local_say, phrase)
                logger.info("[tts] local speech synthesis succeeded for gen %d", gen_id)
            except Exception as fallback_err:
                logger.error("[tts] local fallback also failed: %s", fallback_err)
                if first_phrase and gen_id == turn_controller.current:
                    error_tone = generate_error_tone(sample_rate=SAMPLE_RATE)
                    await queues.tts_queue.put((gen_id, (error_tone, SAMPLE_RATE), latency))
                    first_phrase = False
                continue

        # Staleness check after synthesis: discard if user interrupted during synthesis
        if gen_id != turn_controller.current:
            logger.info("[tts] gen=%d superseded by %d during synthesis — discarding audio", gen_id, turn_controller.current)
            first_phrase = True
            continue

        if first_phrase:
            latency.tts_first_phrase_done_at = time.monotonic()
            tts_latency = latency.tts_first_phrase_done_at - tts_start
            logger.info("[tts] gen=%d first phrase synthesis in %.2fs", gen_id, tts_latency)
            first_phrase = False

        await queues.tts_queue.put((gen_id, (samples, sample_rate), latency))


async def tts_stream():
    """Manage Kokoro initialization, warmup, and parallel phrase accumulator/synthesizer."""
    if not os.path.exists(KOKORO_MODEL_PATH) or not os.path.exists(KOKORO_VOICES_PATH):
        logger.error(
            "[tts] Kokoro model (%s) or voices (%s) not found!",
            KOKORO_MODEL_PATH, KOKORO_VOICES_PATH
        )
        raise FileNotFoundError(f"Missing Kokoro model files: {KOKORO_MODEL_PATH}, {KOKORO_VOICES_PATH}")

    kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)

    # Warmup — ONNX runtime lazy-initializes on the first call.
    # Warming up eliminates first-turn latency spike.
    try:
        await asyncio.to_thread(
            kokoro.create, "Warmup.",
            voice=KOKORO_VOICE,
            speed=KOKORO_SPEED,
            lang=KOKORO_LANG,
        )
    except Exception as e:
        logger.warning("[tts] warmup failed: %s", e)

    logger.info("[tts] ready (engine: Kokoro ONNX, voice: %s, speed: %.2f)...", KOKORO_VOICE, KOKORO_SPEED)

    phrase_queue = asyncio.Queue(maxsize=4)
    await asyncio.gather(
        _accumulator(phrase_queue),
        _synthesizer(kokoro, phrase_queue),
    )
