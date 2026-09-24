import asyncio
import sys
from config.settings import GROQ_API_KEY
from pipeline.mic import microphone_stream
from pipeline.stt import speech_to_text_stream
from pipeline.llm import llm_stream
from pipeline.tts import tts_stream
from pipeline.speaker import speaker_stream


async def main():
    if not GROQ_API_KEY:
        print("\n[error] GROQ_API_KEY is not set!")
        print("Please add your Groq API key in your .env file:")
        print("  GROQ_API_KEY=gsk_...\n")
        print("You can get a free API key at: https://console.groq.com/keys\n")
        sys.exit(1)

    print("[main] starting voice assistant...")

    await asyncio.gather(
        microphone_stream(),
        speech_to_text_stream(),
        llm_stream(),
        tts_stream(),
        speaker_stream(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[main] shutting down...")
    except SystemExit:
        pass
    except Exception as e:
        print(f"\n[main] crashed: {e}")
        raise