import asyncio
import numpy as np
import sounddevice as sd

from config.settings import KOKORO_SAMPLE_RATE
from core.queues import tts_queue
from core.events import interrupt_event, assistant_speaking
from core.sentinel import END_OF_SPEECH


async def speaker_stream():
    print("[speaker] ready...")

    while True:
        item = await tts_queue.get()

        if interrupt_event.is_set():
            assistant_speaking.clear()
            interrupt_event.clear()
            continue

        if item is END_OF_SPEECH:
            assistant_speaking.clear()
            continue

        samples, sample_rate = item
        assistant_speaking.set()

        await asyncio.to_thread(
            sd.play,
            samples,
            samplerate=sample_rate,
            blocking=True,
        )

        if interrupt_event.is_set():
            sd.stop()
            assistant_speaking.clear()
            interrupt_event.clear()