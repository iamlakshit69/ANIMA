import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
CHUNK_SIZE       = 512
CHANNELS         = 1

# ── VAD ──────────────────────────────────────────────────────────────────────
VAD_THRESHOLD    = 0.5   # lower = catches softer speech onsets earlier
# At 16000 Hz / 512 chunk size each chunk is 32 ms.
# 0.3 s = only 9 chunks — natural mid-sentence pauses (breaths, commas,
# thinking) are often longer than that, causing Whisper to fire on fragments.
# 0.8 s is the practical minimum for clean full-sentence captures.
SILENCE_DURATION = 0.8   # seconds of silence before firing SILENCE_MARKER

# ── STT ──────────────────────────────────────────────────────────────────────
WHISPER_LANGUAGE = "en"

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MAX_TOKENS   = 100
LLM_TEMPERATURE  = 0.4
SYSTEM_PROMPT = (
    "You are a voice assistant. "
    "You are having a real spoken conversation with a human — "
    "everything you say will be read aloud, so write exactly as you would speak. "
    "Keep responses short, natural, and warm. "
    "Never use bullet points, lists, markdown, or symbols. "
    "Never start a response with a filler like 'Certainly!' or 'Of course!'. "
    "Just answer directly, like a knowledgeable friend would in conversation."
)

# ── TTS ──────────────────────────────────────────────────────────────────────
KOKORO_VOICE       = "bf_emma"
KOKORO_SPEED       = 1.15
KOKORO_SAMPLE_RATE = 24000

# ── Pipeline ─────────────────────────────────────────────────────────────────
MIN_PHRASE_CHARS = 15
MAX_PHRASE_CHARS = 200
QUEUE_MAX_SIZE   = 100