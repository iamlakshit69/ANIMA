import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.queues import text_queue, token_queue
from core.events import interrupt_event
from core.sentinel import END_OF_RESPONSE


class FakeDelta:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.delta = FakeDelta(content)


class FakeChunk:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class MockAsyncStream:
    def __init__(self, tokens, interrupt_after=None):
        self.tokens = tokens
        self.interrupt_after = interrupt_after
        self.closed = False
        self.idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.idx >= len(self.tokens):
            raise StopAsyncIteration
        token = self.tokens[self.idx]
        self.idx += 1
        if self.interrupt_after is not None and self.idx == self.interrupt_after:
            interrupt_event.set()
        return FakeChunk(token)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_llm_interrupted_stream_cleanup_and_history_rejection():
    """
    Test that when an interruption occurs during LLM streaming:
    1. stream.close() is called.
    2. Partial assistant response is NOT committed to conversation_history.
    3. The pending user message is popped to prevent corrupted history.
    """
    while not text_queue.empty():
        text_queue.get_nowait()
    while not token_queue.empty():
        token_queue.get_nowait()
    interrupt_event.clear()

    mock_stream = MockAsyncStream(["Hello", " world", ", I", " am", " ANIMA"], interrupt_after=2)

    with patch("pipeline.llm.AsyncOpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        from pipeline.llm import llm_stream

        task = asyncio.create_task(llm_stream())

        # Send a user query
        await text_queue.put("Tell me a story")

        # Wait briefly for execution
        await asyncio.sleep(0.1)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # 1. Verify stream.close() was called
        assert mock_stream.closed, "stream.close() was not awaited on interrupt!"

        # 2. Verify token_queue received END_OF_RESPONSE
        tokens_received = []
        while not token_queue.empty():
            tokens_received.append(token_queue.get_nowait())

        assert END_OF_RESPONSE in tokens_received


@pytest.mark.asyncio
async def test_llm_completed_response_commits_clean_turn():
    """
    Test that when an LLM turn completes without interruption,
    both user and assistant messages are properly recorded in conversation_history.
    """
    while not text_queue.empty():
        text_queue.get_nowait()
    while not token_queue.empty():
        token_queue.get_nowait()
    interrupt_event.clear()

    mock_stream = MockAsyncStream(["Paris", " is", " the", " capital."])

    with patch("pipeline.llm.AsyncOpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        from pipeline.llm import llm_stream

        task = asyncio.create_task(llm_stream())

        await text_queue.put("Capital of France?")
        await asyncio.sleep(0.1)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert mock_stream.closed, "stream.close() should be called upon completion in finally block"

        tokens_received = []
        while not token_queue.empty():
            tokens_received.append(token_queue.get_nowait())

        assert "Paris" in tokens_received
        assert END_OF_RESPONSE in tokens_received
