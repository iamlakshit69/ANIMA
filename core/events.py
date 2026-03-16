import asyncio
import time

interrupt_event = asyncio.Event()
assistant_speaking = asyncio.Event()

# Phrase playback timing (used by mic.py for echo cooldown)
speaking_started_at: float = 0.0
current_phrase_duration: float = 0.0
speaking_ended_at: float = time.monotonic()  # not 0.0 — prevents startup echo

# Per-turn latency breakdown
user_stopped_speaking_at: float = 0.0   # stamped when SILENCE_MARKER hits stt.py
stt_done_at: float = 0.0                # stamped when Whisper finishes transcribing
llm_first_token_at: float = 0.0         # stamped when first token arrives from LLM
tts_first_phrase_done_at: float = 0.0   # stamped when first Kokoro synthesis finishes