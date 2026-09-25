import asyncio
import pytest
import core.queues as q
import core.events as ev


@pytest.fixture(autouse=True)
def reset_global_async_state():
    """
    Reset event loop binding and clear all shared queues and events
    between pytest test runs to prevent 'bound to a different event loop' errors.
    """
    for queue in [q.audio_queue, q.text_queue, q.token_queue, q.tts_queue]:
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        queue._loop = None

    ev.interrupt_event._loop = None
    ev.interrupt_event.clear()

    ev.assistant_speaking._loop = None
    ev.assistant_speaking.clear()

    ev.interrupt_counter = 0
    ev.speaking_started_at = 0.0
    ev.current_phrase_duration = 0.0
    ev.speaking_ended_at = 0.0
    ev.user_stopped_speaking_at = 0.0
    ev.stt_done_at = 0.0
    ev.llm_first_token_at = 0.0
    ev.tts_first_phrase_done_at = 0.0

    yield

    for queue in [q.audio_queue, q.text_queue, q.token_queue, q.tts_queue]:
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        queue._loop = None

    ev.interrupt_event._loop = None
    ev.interrupt_event.clear()
    ev.assistant_speaking._loop = None
    ev.assistant_speaking.clear()
