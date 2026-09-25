import asyncio


class TurnController:
    """Monotonically increasing generation / turn controller.
    
    Owned by a single instance, threaded through every message in the pipeline.
    Any pipeline message whose generation_id != turn_controller.current is
    automatically discarded as stale, preventing race conditions during barge-in.
    """

    def __init__(self):
        self._gen = 0
        self._lock = asyncio.Lock()

    async def bump(self) -> int:
        """Call this exactly once at the moment a barge-in or new user utterance
        invalidates the previous turn. Returns the new generation id."""
        async with self._lock:
            self._gen += 1
            return self._gen

    @property
    def current(self) -> int:
        return self._gen

    def reset_for_test(self, gen: int = 0) -> None:
        """Reset generation id for testing."""
        self._gen = gen


turn_controller = TurnController()
