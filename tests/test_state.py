import time
import pytest
from core.state import PlaybackState, TurnLatency


def test_playback_state_lifecycle():
    ps = PlaybackState()
    assert not ps.speaking.is_set()
    assert not ps.interrupted

    ps.mark_started(duration=1.5)
    assert ps.speaking.is_set()
    assert ps.phrase_duration == 1.5
    assert not ps.interrupted
    assert ps.started_at > 0

    ps.mark_ended()
    assert not ps.speaking.is_set()
    assert ps.ended_at >= ps.started_at

    ps.mark_started(duration=2.0)
    ps.mark_interrupted()
    assert not ps.speaking.is_set()
    assert ps.interrupted


def test_turn_latency_isolation():
    """Verify TurnLatency instances are independent and not shared globals."""
    t1 = TurnLatency(user_stopped_speaking_at=100.0)
    t2 = TurnLatency(user_stopped_speaking_at=200.0)

    t1.stt_done_at = 101.5
    t2.stt_done_at = 201.2

    assert t1.user_stopped_speaking_at == 100.0
    assert t2.user_stopped_speaking_at == 200.0
    assert t1.stt_done_at == 101.5
    assert t2.stt_done_at == 201.2
