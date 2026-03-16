import asyncio
import time
from kokoro_onnx import Kokoro

from config.settings import (
    KOKORO_VOICE,
    KOKORO_SPEED,
    # Bug #12 fix: KOKORO_SAMPLE_RATE removed — it was imported but never used.
    # The actual sample rate is taken from kokoro.create()'s return value
    # (samples, sample_rate). Keeping a stale import here would silently hide
    # any mismatch between config and what Kokoro actually returns.
    MIN_PHRASE_CHARS,
    MAX_PHRASE_CHARS,
)
from core.queues import token_queue, tts_queue
from core.events import interrupt_event
import core.events as ev
from core.sentinel import END_OF_RESPONSE, END_OF_SPEECH


async def _accumulator(phrase_queue):
    """
    Reads tokens from token_queue and builds phrases.
    Puts complete phrases into phrase_queue for the synthesizer.
    Never blocks on Kokoro — runs continuously regardless of synthesis speed.
    """
    phrase_buffer = ""

    while True:
        token = await token_queue.get()

        if token is END_OF_RESPONSE:
            # Bug #7 fix: guard the phrase_buffer flush with an interrupt check.
            # Previously this branch ran before the interrupt check below,
            # so partial content accumulated before a barge-in was always
            # flushed unconditionally into phrase_queue. The synthesizer would
            # skip it, but it wasted a slot in the 4-deep queue and caused an
            # unnecessary wake-check-discard cycle.
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
        # Kokoro never receives a mid-word fragment like "Ep" or "min".
        # Previously a hard slice at MAX_PHRASE_CHARS caused exactly that.
        elif len(phrase_buffer) >= MAX_PHRASE_CHARS:
            last_space = phrase_buffer.rfind(" ")
            if last_space > MIN_PHRASE_CHARS:
                # Split cleanly at the last space
                await phrase_queue.put(phrase_buffer[:last_space])
                phrase_buffer = phrase_buffer[last_space + 1:]
            else:
                # No space found in a valid range — flush as-is rather than
                # accumulating indefinitely (handles pathological no-space input)
                await phrase_queue.put(phrase_buffer)
                phrase_buffer = ""


async def _synthesizer(kokoro, phrase_queue):
    """
    Reads complete phrases from phrase_queue and runs Kokoro synthesis.
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

        print(f"[tts] synthesizing: {phrase.strip()}")

        tts_start = time.monotonic()

        samples, sample_rate = await asyncio.to_thread(
            kokoro.create,
            phrase,
            voice=KOKORO_VOICE,
            speed=KOKORO_SPEED,
            lang="en-us",
        )

        # Stamp and print only for the first phrase of each turn
        if first_phrase:
            ev.tts_first_phrase_done_at = time.monotonic()
            tts_took = ev.tts_first_phrase_done_at - tts_start
            print(f"[tts] first phrase in {tts_took:.2f}s")
            first_phrase = False

        if not interrupt_event.is_set():
            await tts_queue.put((samples, sample_rate))


async def tts_stream():
    kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")

    # Warmup — ONNX runtime lazy-initializes on the first real call.
    await asyncio.to_thread(kokoro.create, "Hello.", voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us")

    print("[tts] ready...")

    phrase_queue = asyncio.Queue(maxsize=4)

    await asyncio.gather(
        _accumulator(phrase_queue),
        _synthesizer(kokoro, phrase_queue),
    )