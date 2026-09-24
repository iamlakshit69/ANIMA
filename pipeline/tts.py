import asyncio
import os
import time
import urllib.request
import numpy as np
from piper import PiperVoice

from config.settings import (
    PIPER_VOICE_NAME,
    PIPER_MODEL_PATH,
    PIPER_CONFIG_PATH,
    MIN_PHRASE_CHARS,
    MAX_PHRASE_CHARS,
)
from core.queues import token_queue, tts_queue
from core.events import interrupt_event
import core.events as ev
from core.sentinel import END_OF_RESPONSE, END_OF_SPEECH


def _ensure_voice_downloaded():
    """Ensure the configured Piper voice model and config files exist."""
    if os.path.exists(PIPER_MODEL_PATH) and os.path.exists(PIPER_CONFIG_PATH):
        return

    os.makedirs(os.path.dirname(PIPER_MODEL_PATH), exist_ok=True)
    parts = PIPER_VOICE_NAME.split("-")
    lang_code = parts[0]  # e.g. "en_US"
    voice_name = parts[1] # e.g. "lessac"
    quality = parts[2]    # e.g. "low" or "medium"
    lang_family = lang_code.split("_")[0]  # e.g. "en"

    base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_family}/{lang_code}/{voice_name}/{quality}"
    model_url = f"{base_url}/{PIPER_VOICE_NAME}.onnx"
    config_url = f"{base_url}/{PIPER_VOICE_NAME}.onnx.json"

    print(f"[tts] downloading Piper voice {PIPER_VOICE_NAME}...")
    urllib.request.urlretrieve(model_url, PIPER_MODEL_PATH)
    urllib.request.urlretrieve(config_url, PIPER_CONFIG_PATH)
    print(f"[tts] downloaded {PIPER_VOICE_NAME}")


def _synthesize_phrase(voice, phrase):
    """Run Piper synthesis synchronously on a worker thread."""
    chunks = [chunk.audio_float_array for chunk in voice.synthesize(phrase)]
    if not chunks:
        return np.zeros(0, dtype=np.float32), voice.config.sample_rate
    if len(chunks) == 1:
        return chunks[0], voice.config.sample_rate
    return np.concatenate(chunks), voice.config.sample_rate


async def _accumulator(phrase_queue):
    """
    Reads tokens from token_queue and builds phrases.
    Puts complete phrases into phrase_queue for the synthesizer.
    Never blocks on synthesis — runs continuously regardless of synthesis speed.
    """
    phrase_buffer = ""

    while True:
        token = await token_queue.get()

        if token is END_OF_RESPONSE:
            if phrase_buffer.strip() and not interrupt_event.is_set():
                await phrase_queue.put(phrase_buffer)
            phrase_buffer = ""
            await phrase_queue.put(END_OF_RESPONSE)
            continue

        if interrupt_event.is_set():
            phrase_buffer = ""
            continue

        phrase_buffer += token

        # Punctuation path — already at a clean sentence/clause boundary.
        if len(phrase_buffer) >= MIN_PHRASE_CHARS and phrase_buffer[-1] in ".!?,:":
            await phrase_queue.put(phrase_buffer)
            phrase_buffer = ""

        # Length ceiling path — walk back to the nearest word boundary so
        # TTS never receives a mid-word fragment.
        elif len(phrase_buffer) >= MAX_PHRASE_CHARS:
            last_space = phrase_buffer.rfind(" ")
            if last_space > MIN_PHRASE_CHARS:
                await phrase_queue.put(phrase_buffer[:last_space])
                phrase_buffer = phrase_buffer[last_space + 1:]
            else:
                await phrase_queue.put(phrase_buffer)
                phrase_buffer = ""


async def _synthesizer(voice, phrase_queue):
    """
    Reads complete phrases from phrase_queue and runs Piper synthesis.
    Runs concurrently with _accumulator.
    """
    first_phrase = True

    while True:
        phrase = await phrase_queue.get()

        if phrase is END_OF_RESPONSE:
            await tts_queue.put(END_OF_SPEECH)
            first_phrase = True  # reset for next turn
            continue

        if interrupt_event.is_set():
            first_phrase = True
            continue

        if not phrase.strip():
            continue

        print(f"[tts] synthesizing: {phrase.strip()}")

        tts_start = time.monotonic()

        samples, sample_rate = await asyncio.to_thread(_synthesize_phrase, voice, phrase)

        # Stamp and print only for the first phrase of each turn
        if first_phrase:
            ev.tts_first_phrase_done_at = time.monotonic()
            tts_took = ev.tts_first_phrase_done_at - tts_start
            print(f"[tts] first phrase in {tts_took:.3f}s ({tts_took*1000:.1f}ms)")
            first_phrase = False

        if not interrupt_event.is_set():
            await tts_queue.put((samples, sample_rate))


async def tts_stream():
    _ensure_voice_downloaded()
    voice = PiperVoice.load(PIPER_MODEL_PATH, config_path=PIPER_CONFIG_PATH)

    # Warmup — ONNX runtime lazy-initializes on the first real call.
    await asyncio.to_thread(_synthesize_phrase, voice, "Hello.")

    print(f"[tts] ready... (piper: {PIPER_VOICE_NAME})")

    phrase_queue = asyncio.Queue(maxsize=4)

    await asyncio.gather(
        _accumulator(phrase_queue),
        _synthesizer(voice, phrase_queue),
    )