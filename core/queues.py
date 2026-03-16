import asyncio
from config.settings import QUEUE_MAX_SIZE

audio_queue  = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
text_queue   = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
token_queue  = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
# Bug #11 fix: was hardcoded to 5, now uses QUEUE_MAX_SIZE like every other
# queue. Previously, tuning QUEUE_MAX_SIZE in config had no effect on
# tts_queue, silently leaving it at 5 regardless.
tts_queue    = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)