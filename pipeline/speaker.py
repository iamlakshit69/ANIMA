import asyncio
import time
import sounddevice as sd

from config.settings import SPEAKER_DEVICE
import core.queues as queues
from core.state import playback_state, TurnLatency
from core.turn import turn_controller
from core.sentinel import END_OF_SPEECH
from core.logger import get_logger

logger = get_logger("anima.speaker")

POLL_INTERVAL = 0.02  # seconds between interrupt/generation checks during playback (20 ms)


async def speaker_stream():
    """Continuously receive synthesized audio chunks tagged with generation IDs,
    play them with sounddevice, log per-turn latency breakdowns, and support
    instant cancellation if superseded by a newer generation ID.
    """
    try:
        out_dev = sd.query_devices(SPEAKER_DEVICE, kind="output")
        logger.info(
            "[speaker] using output device #%s: %s",
            out_dev.get("index", "default"), out_dev.get("name")
        )
    except Exception as e:
        logger.warning("[speaker] could not query output device %s: %s", SPEAKER_DEVICE, e)

    logger.info("[speaker] ready...")

    while True:
        item = await queues.tts_queue.get()

        if isinstance(item, tuple):
            if len(item) == 3:
                gen_id, payload, latency = item
            else:
                gen_id, payload = item
                latency = TurnLatency()
        else:
            gen_id = turn_controller.current
            payload = item
            latency = TurnLatency()

        # Staleness check: discard stale audio chunk or stale END_OF_SPEECH
        # directly eliminating the race condition from Finding 2
        if gen_id != turn_controller.current:
            logger.debug(
                "[speaker] dropping stale item for gen %d (current=%d)",
                gen_id, turn_controller.current
            )
            continue

        if payload is END_OF_SPEECH:
            playback_state.mark_ended()
            logger.info("[speaker] gen=%d finished speech", gen_id)
            continue

        samples, sample_rate = payload

        # Print full per-step latency breakdown on the first audio chunk of this turn
        if latency and latency.user_stopped_speaking_at > 0:
            now = time.monotonic()
            stt_time = max(0.0, latency.stt_done_at - latency.user_stopped_speaking_at) if latency.stt_done_at > 0 else 0.0
            llm_time = max(0.0, latency.llm_first_token_at - latency.stt_done_at) if (latency.llm_first_token_at > 0 and latency.stt_done_at > 0) else 0.0
            tts_time = max(0.0, latency.tts_first_phrase_done_at - latency.llm_first_token_at) if (latency.tts_first_phrase_done_at > 0 and latency.llm_first_token_at > 0) else 0.0
            total    = max(0.0, now - latency.user_stopped_speaking_at)

            logger.info(
                "[latency] gen=%d total %.2fs | STT %.2fs | LLM %.2fs | TTS %.2fs",
                gen_id, total, stt_time, llm_time, tts_time
            )
            latency.user_stopped_speaking_at = 0.0

        playback_state.mark_started(len(samples) / sample_rate)

        # Non-blocking play + polling loop so we can stop mid-phrase on barge-in
        try:
            sd.play(samples, samplerate=sample_rate, device=SPEAKER_DEVICE)

            while sd.get_stream().active:
                if gen_id != turn_controller.current or playback_state.interrupted:
                    sd.stop()
                    playback_state.mark_ended()
                    logger.info("[speaker] gen=%d playback interrupted (superseded by %d)", gen_id, turn_controller.current)
                    break
                await asyncio.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.error("[speaker] error playing audio chunk for gen %d: %s", gen_id, e)
            playback_state.mark_ended()