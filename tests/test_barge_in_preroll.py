import collections
import pytest
import numpy as np
import asyncio
from config.settings import SAMPLE_RATE, CHUNK_SIZE, BARGE_IN_PREROLL_SECONDS, BARGE_IN_FRAMES
from core.sentinel import INTERRUPT


@pytest.mark.asyncio
async def test_barge_in_preroll_preserves_initial_audio():
    """
    Test that the rolling pre-roll buffer captures chunks prior to barge-in confirmation,
    ensuring initial speech (e.g. the first 250ms+) is not discarded.
    """
    preroll_limit = max(1, int(BARGE_IN_PREROLL_SECONDS * SAMPLE_RATE / CHUNK_SIZE))
    assert preroll_limit >= 15  # 0.5s * 16000 / 512 ~ 15 chunks

    preroll_buffer = collections.deque(maxlen=preroll_limit)
    out_queue = asyncio.Queue()

    # Simulate 20 audio chunks captured while assistant was speaking
    # Chunks 0..11 are speech onset (below or leading up to BARGE_IN_FRAMES)
    chunks = [np.full(CHUNK_SIZE, fill_value=i, dtype=np.float32) for i in range(25)]

    speech_frames = 0
    barge_in_triggered = False

    for i, chunk in enumerate(chunks):
        preroll_buffer.append(chunk)

        # Simulate speech detected from chunk 10 onwards
        is_speech = i >= 10
        if is_speech:
            speech_frames += 1
            if speech_frames >= BARGE_IN_FRAMES and not barge_in_triggered:
                barge_in_triggered = True
                # Trigger barge-in: emit INTERRUPT and flush pre-roll
                await out_queue.put(INTERRUPT)
                while preroll_buffer:
                    await out_queue.put(preroll_buffer.popleft())
                continue

    assert barge_in_triggered, "Barge-in should have been triggered"

    # Verify queue contents: first item is INTERRUPT sentinel
    first_item = await out_queue.get()
    assert first_item is INTERRUPT

    # Verify that pre-roll chunks were preserved and chronological
    flushed_chunks = []
    while not out_queue.empty():
        flushed_chunks.append(await out_queue.get())

    # Pre-roll should have contained preroll_limit chunks up to the trigger point
    assert len(flushed_chunks) == preroll_limit
    # Verify values are in strict ascending order (chronological)
    chunk_ids = [int(c[0]) for c in flushed_chunks]
    assert chunk_ids == list(range(chunk_ids[0], chunk_ids[0] + preroll_limit))
    # Crucially: chunk 10 (the very first speech frame) MUST be preserved in the pre-roll!
    assert 10 in chunk_ids, "Initial speech frame 10 must be preserved in pre-roll"
