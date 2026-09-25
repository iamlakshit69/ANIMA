import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class PlaybackState:
    """Explicit owned state for audio playback and echo management."""
    speaking: asyncio.Event = field(default_factory=asyncio.Event)
    started_at: float = 0.0
    phrase_duration: float = 0.0
    ended_at: float = field(default_factory=time.monotonic)
    interrupted: bool = False
    last_turn_completed: bool = True

    def mark_started(self, duration: float) -> None:
        self.started_at = time.monotonic()
        self.phrase_duration = duration
        self.speaking.set()
        self.interrupted = False
        self.last_turn_completed = False

    def mark_ended(self) -> None:
        self.speaking.clear()
        self.ended_at = time.monotonic()
        self.last_turn_completed = True

    def mark_interrupted(self) -> None:
        self.interrupted = True
        self.mark_ended()


@dataclass
class TurnLatency:
    """Per-turn latency breakdown stamped at each pipeline boundary.
    
    Created fresh per turn and passed through queues alongside generation_id,
    eliminating all shared global timestamps and TOCTOU races across awaits.
    """
    user_stopped_speaking_at: float = 0.0   # stamped in stt.py when SILENCE_MARKER is dequeued
    stt_done_at: float = 0.0               # stamped in stt.py after Whisper returns
    llm_first_token_at: float = 0.0        # stamped in llm.py when first token arrives
    tts_first_phrase_done_at: float = 0.0  # stamped in tts.py when first synthesis finishes


playback_state = PlaybackState()
