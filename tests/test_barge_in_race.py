import asyncio
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from core.turn import turn_controller
from core.state import playback_state, TurnLatency
from core.sentinel import END_OF_SPEECH
import core.queues as queues
from pipeline.speaker import speaker_stream


@pytest.fixture(autouse=True)
def reset_state():
    from core.queues import reset_queues
    reset_queues()
    turn_controller.reset_for_test(0)
    playback_state.mark_ended()


@pytest.mark.asyncio
async def test_barge_in_race_condition_eliminated():
    """Directly test the race condition described in Finding 2:
    
    1. Turn 1 is active (gen 1).
    2. User barges in, bumping generation to 2.
    3. Trailing END_OF_SPEECH from Turn 1 arrives at speaker.
    4. Assert that trailing Turn 1 END_OF_SPEECH is dropped and does NOT
       corrupt or prematurely reset Turn 2's active state.
    """
    g1 = await turn_controller.bump()  # gen 1
    assert g1 == 1

    played_generations = []

    with patch("pipeline.speaker.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_sd.get_stream.return_value = mock_stream

        def fake_play(samples, samplerate, device=None):
            # Record playback call
            pass
        mock_sd.play.side_effect = fake_play

        speaker_task = asyncio.create_task(speaker_stream())

        # Barge-in happens: user starts turn 2!
        g2 = await turn_controller.bump()  # gen 2
        assert g2 == 2

        # Turn 2 starts playback
        turn2_audio = np.ones(480, dtype=np.float32)
        latency2 = TurnLatency(user_stopped_speaking_at=100.0)

        # In interleaved order:
        # First, a trailing END_OF_SPEECH from turn 1 arrives!
        await queues.tts_queue.put((g1, END_OF_SPEECH, TurnLatency()))
        # Then, valid turn 2 audio arrives!
        await queues.tts_queue.put((g2, (turn2_audio, 16000), latency2))
        # Finally, turn 2 END_OF_SPEECH arrives!
        await queues.tts_queue.put((g2, END_OF_SPEECH, latency2))

        await asyncio.sleep(0.1)
        speaker_task.cancel()
        try:
            await speaker_task
        except asyncio.CancelledError:
            pass

        # Verify: sd.play was called for gen 2
        assert mock_sd.play.call_count == 1
        args, kwargs = mock_sd.play.call_args
        np.testing.assert_array_equal(args[0], turn2_audio)

        # Verify: Playback state ended cleanly at the end of gen 2
        assert not playback_state.speaking.is_set()
