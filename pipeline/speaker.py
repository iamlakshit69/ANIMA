import asyncio
import time
import numpy as np
import pyaudio

from core.queues import tts_queue, audio_queue
from core.events import interrupt_event, assistant_speaking
import core.events as ev
from core.sentinel import END_OF_SPEECH

CHUNK_SIZE = 1024  # audio frames written per iteration (~64ms at 16kHz)


async def speaker_stream():
    print("[speaker] ready...")
    audio = pyaudio.PyAudio()
    current_stream = None
    current_rate = None

    def _get_stream(rate):
        nonlocal current_stream, current_rate
        if current_stream is not None and current_rate == rate:
            return current_stream
        if current_stream is not None:
            try:
                current_stream.stop_stream()
                current_stream.close()
            except Exception:
                pass
        current_stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=rate,
            output=True,
        )
        current_rate = rate
        return current_stream

    try:
        while True:
            item = await tts_queue.get()

            if item is END_OF_SPEECH:
                assistant_speaking.clear()
                ev.speaking_ended_at = time.monotonic()
                interrupt_event.clear()

                # Flush any echo chunks already sitting in audio_queue
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

            stream = _get_stream(sample_rate)
            samples_bytes = samples.astype(np.float32).tobytes()
            bytes_per_chunk = CHUNK_SIZE * 4  # 4 bytes per float32 sample

            for offset in range(0, len(samples_bytes), bytes_per_chunk):
                if interrupt_event.is_set():
                    try:
                        stream.stop_stream()
                        stream.start_stream()
                    except Exception:
                        pass
                    print("[speaker] playback interrupted (barge-in)")
                    break

                chunk_bytes = samples_bytes[offset:offset + bytes_per_chunk]
                await asyncio.to_thread(stream.write, chunk_bytes)

    finally:
        if current_stream is not None:
            try:
                current_stream.stop_stream()
                current_stream.close()
            except Exception:
                pass
        audio.terminate()