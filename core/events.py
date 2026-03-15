import asyncio
import time

interrupt_event = asyncio.Event()
assistant_speaking = asyncio.Event()
speaking_started_at: float = 0.0       # stamped when each phrase starts playing
current_phrase_duration: float = 0.0   # duration of the phrase currently playing