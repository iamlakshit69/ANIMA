import asyncio
import time

interrupt_event = asyncio.Event()
assistant_speaking = asyncio.Event()
speaking_started_at: float = 0.0        # stamped when each phrase starts playing
current_phrase_duration: float = 0.0    # duration of the phrase currently playing
user_stopped_speaking_at: float = 0.0   # stamped when SILENCE_MARKER hits stt.py
speaking_ended_at: float = 0.0          # stamped when END_OF_SPEECH clears assistant_speaking