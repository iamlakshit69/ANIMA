import asyncio
import time

interrupt_event    = asyncio.Event()
assistant_speaking = asyncio.Event()

# Phrase playback timing (used by mic.py for echo cooldown)
speaking_started_at: float = 0.0
current_phrase_duration: float = 0.0

# Initialised to now rather than 0.0 — prevents the very first mic chunk
# from passing the POST_SPEECH_MUTE / BUFFER_MUTE_GUARD checks on startup
# before the assistant has ever spoken.
speaking_ended_at: float = time.monotonic()

# Per-turn latency breakdown — stamped by each pipeline stage and read by
# speaker.py to print the full STT | LLM | TTS | total breakdown on the
# first audio chunk of each turn.
user_stopped_speaking_at: float = 0.0   # stamped in stt.py when SILENCE_MARKER is dequeued
stt_done_at: float = 0.0               # stamped in stt.py after Groq Whisper returns
llm_first_token_at: float = 0.0        # stamped in llm.py when first token arrives
tts_first_phrase_done_at: float = 0.0  # stamped in tts.py when first Kokoro synthesis finishes