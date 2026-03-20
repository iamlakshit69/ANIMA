import asyncio
import io
import time
import wave
import numpy as np
from groq import AsyncGroq

from config.settings import (
    GROQ_API_KEY, GROQ_TTS_MODEL, GROQ_TTS_VOICE,
    MIN_PHRASE_CHARS, MAX_PHRASE_CHARS,
)
from core.queues import token_queue, tts_queue
from core.events import interrupt_event
from core.sentinel import END_OF_RESPONSE, END_OF_SPEECH


def _wav_bytes_to_numpy(wav_bytes):
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


async def _synthesize_groq(client, text):
    async with client.audio.speech.with_streaming_response.create(
        model=GROQ_TTS_MODEL,
        voice=GROQ_TTS_VOICE,
        input=text,
        response_format="wav",
    ) as response:
        wav_bytes = await response.read()
    return await asyncio.to_thread(_wav_bytes_to_numpy, wav_bytes)


async def _accumulator(phrase_queue):
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
        should_synthesize = (
            len(phrase_buffer) >= MIN_PHRASE_CHARS and phrase_buffer[-1] in ".!?,:"
        ) or len(phrase_buffer) >= MAX_PHRASE_CHARS
        if should_synthesize:
            await phrase_queue.put(phrase_buffer)
            phrase_buffer = ""


async def _synthesizer(client, phrase_queue):
    first_phrase = True
    while True:
        phrase = await phrase_queue.get()
        if phrase is END_OF_RESPONSE:
            await tts_queue.put(END_OF_SPEECH)
            first_phrase = True
            continue
        if interrupt_event.is_set():
            first_phrase = True
            continue
        print(f"[tts] synthesizing: {phrase.strip()}")
        tts_start = time.monotonic()
        try:
            samples, sample_rate = await _synthesize_groq(client, phrase)
        except Exception as e:
            print(f"[tts] groq error: {e}")
            continue
        if first_phrase:
            import core.events as ev_mod; ev_mod.tts_first_phrase_done_at = __import__("time").monotonic()
            print(f"[tts] first phrase in {time.monotonic() - tts_start:.2f}s")
            first_phrase = False
        if not interrupt_event.is_set():
            await tts_queue.put((samples, sample_rate))


async def tts_stream():
    client = AsyncGroq(api_key=GROQ_API_KEY)
    print("[tts] ready...")
    phrase_queue = asyncio.Queue(maxsize=4)
    await asyncio.gather(
        _accumulator(phrase_queue),
        _synthesizer(client, phrase_queue),
    )
