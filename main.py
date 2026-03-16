import asyncio
from pipeline.mic import microphone_stream
from pipeline.stt import speech_to_text_stream
from pipeline.llm import llm_stream
from pipeline.tts import tts_stream
from pipeline.speaker import speaker_stream


async def main():
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
    except Exception as e:
        print(f"\n[main] crashed: {e}")
        raise