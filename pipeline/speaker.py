import asyncio
import time
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
            ev.speaking_ended_at = time.monotonic()  # stamp when speaking fully ends
            interrupt_event.clear()
            continue

        samples, sample_rate = item

        if ev.user_stopped_speaking_at > 0:
            latency = time.monotonic() - ev.user_stopped_speaking_at
            print(f"[latency] {latency:.2f}s  (silence -> first audio)")
            ev.user_stopped_speaking_at = 0.0

        ev.speaking_started_at = time.monotonic()
        ev.current_phrase_duration = len(samples) / sample_rate
        assistant_speaking.set()

        await asyncio.to_thread(
            sd.play,
            samples,
            samplerate=sample_rate,
            blocking=True,
        )

        if interrupt_event.is_set():
            sd.stop()