import asyncio
import time
import numpy as np
import sounddevice as sd

from config.settings import KOKORO_SAMPLE_RATE
from core.queues import tts_queue
from core.events import interrupt_event, assistant_speaking
import core.events as ev
from core.sentinel import END_OF_SPEECH


async def speaker_stream():
    print("[speaker] ready...")

    while True:
        item = await tts_queue.get()

        if item is END_OF_SPEECH:
            assistant_speaking.clear()
            interrupt_event.clear()   # safe to clear here - response is fully done
            continue

        samples, sample_rate = item

        # Stamp both the start time AND the duration of this phrase.
        # mic.py uses duration to set a dynamic cooldown so that long phrases
        # don't have their own echo falsely trigger a barge-in interrupt.
        ev.speaking_started_at = time.monotonic()
        ev.current_phrase_duration = len(samples) / sample_rate
        assistant_speaking.set()

        await asyncio.to_thread(
            sd.play,
            samples,
            samplerate=sample_rate,
            blocking=True,
        )

        # Stop hardware playback if interrupted mid-chunk, but do NOT skip
        # remaining queued chunks - they will drain naturally until END_OF_SPEECH.
        if interrupt_event.is_set():
            sd.stop()