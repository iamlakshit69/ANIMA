import asyncio
from config.settings import QUEUE_MAX_SIZE

audio_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
text_queue: asyncio.Queue  = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
token_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
tts_queue: asyncio.Queue   = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)


def reset_queues():
    """Reset all queues to fresh instances bound to the current running event loop."""
    global audio_queue, text_queue, token_queue, tts_queue
    audio_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    text_queue  = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    token_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    tts_queue   = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)