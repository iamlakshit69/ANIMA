import asyncio
from config.settings import QUEUE_MAX_SIZE

audio_queue  = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
text_queue   = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
token_queue  = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
tts_queue    = asyncio.Queue(maxsize=5)