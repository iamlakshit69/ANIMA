import asyncio
import pytest
from core.sentinel import SILENCE_MARKER


@pytest.mark.asyncio
async def test_silence_marker_emitted_only_once_at_transition():
    """
    Test that SILENCE_MARKER is emitted exactly once when transitioning
    from speech to silence, and NOT continuously during ongoing silence.
    """
    out_queue = asyncio.Queue()
    silence_limit = 5
    silence_chunks = 0
    is_user_speaking = False

    # Simulate sequence:
    # 3 chunks of speech
    # followed by 20 chunks of silence
    speech_frames_input = [True, True, True] + [False] * 20

    emitted_markers = 0

    for is_speech in speech_frames_input:
        if is_speech:
            is_user_speaking = True
            silence_chunks = 0
            await out_queue.put("speech_chunk")
        else:
            if is_user_speaking:
                silence_chunks += 1
                if silence_chunks >= silence_limit:
                    await out_queue.put(SILENCE_MARKER)
                    emitted_markers += 1
                    is_user_speaking = False
                    silence_chunks = 0
            else:
                # Still silent, do nothing
                pass

    # Verify that SILENCE_MARKER was emitted exactly once!
    assert emitted_markers == 1, f"Expected 1 SILENCE_MARKER, but got {emitted_markers}"

    # Verify queue contents: 3 speech chunks, then 1 SILENCE_MARKER, and nothing else
    items = []
    while not out_queue.empty():
        items.append(await out_queue.get())

    assert len(items) == 4
    assert items[-1] is SILENCE_MARKER
