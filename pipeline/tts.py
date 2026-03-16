import asyncio
from kokoro_onnx import Kokoro

from config.settings import (
    KOKORO_VOICE,
    KOKORO_SPEED,
    KOKORO_SAMPLE_RATE,
    MIN_PHRASE_CHARS,
    MAX_PHRASE_CHARS,
)
from core.queues import token_queue, tts_queue
from core.events import interrupt_event
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

        # Sentinel first — unconditionally, no interrupt gate
        if token is END_OF_RESPONSE:
            if phrase_buffer.strip():
                await phrase_queue.put(phrase_buffer)
            phrase_buffer = ""
            await phrase_queue.put(END_OF_RESPONSE)  # signal synthesizer to send END_OF_SPEECH
            continue

        if interrupt_event.is_set():
            phrase_buffer = ""
            continue

        phrase_buffer += token

        should_synthesize = (
            len(phrase_buffer) >= MIN_PHRASE_CHARS
            and phrase_buffer[-1] in ".!?,:"
        ) or len(phrase_buffer) >= MAX_PHRASE_CHARS

        if should_synthesize:
            await phrase_queue.put(phrase_buffer)
            phrase_buffer = ""


async def _synthesizer(kokoro, phrase_queue):
    """
    Reads complete phrases from phrase_queue and runs Kokoro synthesis.
    Runs concurrently with _accumulator — while this is synthesizing phrase 1,
    _accumulator is already collecting phrase 2.
    """
    while True:
        phrase = await phrase_queue.get()

        # Sentinel — always handle unconditionally
        if phrase is END_OF_RESPONSE:
            await tts_queue.put(END_OF_SPEECH)
            continue

        # Skip synthesis if interrupted — but still drain the queue
        if interrupt_event.is_set():
            continue

        print(f"[tts] synthesizing: {phrase.strip()}")

        samples, sample_rate = await asyncio.to_thread(
            kokoro.create,
            phrase,
            voice=KOKORO_VOICE,
            speed=KOKORO_SPEED,
            lang="en-us",
        )

        # Check again after synthesis — interrupt may have fired while Kokoro was running
        if not interrupt_event.is_set():
            await tts_queue.put((samples, sample_rate))


async def tts_stream():
    kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")

    # Warmup — ONNX runtime lazy-initializes on the first real call.
    # Without this the first response is 200-500ms slower than all subsequent ones.
    await asyncio.to_thread(kokoro.create, "Hello.", voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us")

    print("[tts] ready...")

    # Internal queue between accumulator and synthesizer.
    # Small size (4) creates natural backpressure — synthesizer can't fall
    # too far behind accumulator, and stale phrases don't pile up.
    phrase_queue = asyncio.Queue(maxsize=4)

    # Run both concurrently — this is the key improvement.
    # Previously synthesis blocked token accumulation entirely.
    await asyncio.gather(
        _accumulator(phrase_queue),
        _synthesizer(kokoro, phrase_queue),
    )