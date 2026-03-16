import asyncio
import time
import sounddevice as sd

from core.queues import tts_queue, audio_queue
from core.events import interrupt_event, assistant_speaking
import core.events as ev
from core.sentinel import END_OF_SPEECH

POLL_INTERVAL = 0.02  # seconds between interrupt checks during playback (20 ms)


async def speaker_stream():
    print("[speaker] ready...")

    while True:
        item = await tts_queue.get()

        if item is END_OF_SPEECH:
            assistant_speaking.clear()
            ev.speaking_ended_at = time.monotonic()
            interrupt_event.clear()

            # Flush any echo chunks already sitting in audio_queue.
            # These were queued during the tiny gap between sd.play() finishing
            # and END_OF_SPEECH being processed — they are pure echo, not user speech.
            flushed = 0
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                    flushed += 1
                except asyncio.QueueEmpty:
                    break
            if flushed > 0:
                print(f"[speaker] flushed {flushed} stale echo chunks from audio_queue")
            continue

        samples, sample_rate = item

        # Print full per-step breakdown on first audio chunk of each turn
        if ev.user_stopped_speaking_at > 0:
            now = time.monotonic()

            stt_time  = ev.stt_done_at              - ev.user_stopped_speaking_at
            llm_time  = ev.llm_first_token_at       - ev.stt_done_at
            tts_time  = ev.tts_first_phrase_done_at - ev.llm_first_token_at
            total     = now                         - ev.user_stopped_speaking_at

            print(
                f"[latency] total {total:.2f}s  |  "
                f"STT {stt_time:.2f}s  |  "
                f"LLM {llm_time:.2f}s  |  "
                f"TTS {tts_time:.2f}s"
            )

            ev.user_stopped_speaking_at = 0.0

        ev.speaking_started_at = time.monotonic()
        ev.current_phrase_duration = len(samples) / sample_rate
        assistant_speaking.set()

        # Use non-blocking play so we can poll for barge-in mid-phrase.
        # Previously blocking=True meant sd.stop() was called only AFTER
        # the phrase had already finished playing — making barge-in a no-op.
        sd.play(samples, samplerate=sample_rate)

        while sd.get_stream().active:
            if interrupt_event.is_set():
                sd.stop()
                print("[speaker] playback interrupted (barge-in)")
                break
            await asyncio.sleep(POLL_INTERVAL)