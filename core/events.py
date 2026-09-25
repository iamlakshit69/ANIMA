"""
core/events.py
==============
Backward-compatibility shim.
All new code should import from `core.state` and `core.turn`.
"""

import asyncio
from core.state import playback_state, TurnLatency
from core.turn import turn_controller

# Backward-compatibility aliases
interrupt_event = asyncio.Event()
assistant_speaking = playback_state.speaking

# Legacy floats mapped to playback_state
def __getattr__(name):
    if name == "speaking_started_at":
        return playback_state.started_at
    if name == "current_phrase_duration":
        return playback_state.phrase_duration
    if name == "speaking_ended_at":
        return playback_state.ended_at
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")