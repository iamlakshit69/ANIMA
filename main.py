import asyncio
import sys
from config.settings import GROQ_API_KEY, LOG_LEVEL, LOG_FILE
from core.logger import setup_logging, get_logger
from core.turn import turn_controller
from core.state import playback_state
from pipeline.mic import microphone_stream
from pipeline.stt import speech_to_text_stream
from pipeline.llm import llm_stream
from pipeline.tts import tts_stream
from pipeline.speaker import speaker_stream

# Initialize central logging
logger = setup_logging(LOG_LEVEL, LOG_FILE)


async def _supervise(name: str, coro_fn, *args, initial_backoff: float = 1.0, max_backoff: float = 30.0):
    """Supervising task loop with exponential backoff on crashes (Finding 4.1).
    
    If a stage raises an unhandled exception, it catches, logs with full traceback,
    and restarts that specific task without taking down the rest of the assistant.
    """
    backoff = initial_backoff
    while True:
        try:
            await coro_fn(*args)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] crashed — restarting in %.1fs", name, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        else:
            logger.error("[%s] exited its main loop without an exception", name)
            return


async def _terminal_barge_in_listener():
    """Optional interactive barge-in listener: press Enter in terminal to interrupt assistant."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    try:
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        logger.info("[main] interactive barge-in enabled: press [Enter] anytime to interrupt playback")
        while True:
            await reader.readline()
            new_gen = await turn_controller.bump()
            playback_state.mark_interrupted()
            logger.info("[main] manual barge-in triggered via Enter key — bumped generation to %d", new_gen)
    except Exception:
        # Non-interactive or unsupported stdin environment (e.g. piped or GUI)
        pass


async def main():
    if not GROQ_API_KEY:
        print("\n[error] GROQ_API_KEY is not set!")
        print("Please add your Groq API key in your .env file:")
        print("  GROQ_API_KEY=gsk_...\n")
        print("You can get a free API key at: https://console.groq.com/keys\n")
        sys.exit(1)

    logger.info("[main] starting ANIMA voice assistant...")

    stages = [
        ("mic", microphone_stream),
        ("stt", speech_to_text_stream),
        ("llm", llm_stream),
        ("tts", tts_stream),
        ("speaker", speaker_stream),
    ]

    tasks = [_supervise(name, fn) for name, fn in stages]

    if sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        tasks.append(_terminal_barge_in_listener())

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[main] shutting down...")
    except SystemExit:
        pass
    except Exception as e:
        logger.exception("[main] fatal error: %s", e)
        raise