import asyncio
from config.settings import QUEUE_MAX_SIZE

audio_queue  = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
text_queue   = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
token_queue  = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)

# Previously hardcoded to 5 — tuning QUEUE_MAX_SIZE in config had no effect
# on tts_queue, silently leaving it at 5 regardless of what was set.
# Now consistent with every other queue.
tts_queue    = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)