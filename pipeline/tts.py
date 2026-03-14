import asyncio
import numpy as np
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


async def tts_stream():
    kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")
    phrase_buffer = ""

    print("[tts] ready...")

    while True:
        token = await token_queue.get()

        if interrupt_event.is_set():
            phrase_buffer = ""
            continue

        if token is END_OF_RESPONSE:
            if phrase_buffer.strip():
                await synthesize_and_enqueue(kokoro, phrase_buffer)
                phrase_buffer = ""
            await tts_queue.put(END_OF_SPEECH)
            continue

        phrase_buffer += token

        should_synthesize = (
            len(phrase_buffer) >= MIN_PHRASE_CHARS
            and phrase_buffer[-1] in ".!?,:"
        ) or len(phrase_buffer) >= MAX_PHRASE_CHARS

        if should_synthesize:
            await synthesize_and_enqueue(kokoro, phrase_buffer)
            phrase_buffer = ""


async def synthesize_and_enqueue(kokoro, text):
    print(f"[tts] synthesizing: {text.strip()}")

    samples, sample_rate = await asyncio.to_thread(
        kokoro.create,
        text,
        voice=KOKORO_VOICE,
        speed=KOKORO_SPEED,
        lang="en-us",
    )

    await tts_queue.put((samples, sample_rate))