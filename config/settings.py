import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHUNK_SIZE  = 512
CHANNELS    = 1

# ── VAD ──────────────────────────────────────────────────────────────────────
VAD_THRESHOLD    = 0.5

# At 16000 Hz / 512 chunk size each chunk is 32 ms.
# 0.4 s (original) = only 12 chunks — natural mid-sentence pauses (breaths,
# commas, thinking) are often longer, causing Groq Whisper to fire on
# fragments and transcribe incomplete sentences.
# 0.8 s is the practical minimum for clean full-sentence captures.
SILENCE_DURATION = 0.8   # seconds of silence before firing SILENCE_MARKER

# ── STT ──────────────────────────────────────────────────────────────────────
WHISPER_LANGUAGE   = "en"

GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_MAX_TOKENS  = 120   # raised from 100 — gives the model a touch more room
                          # to complete a 2-sentence answer without truncation
GROQ_TEMPERATURE = 0.4

SYSTEM_PROMPT = (
    "You are a voice assistant. "
    "You are having a real spoken conversation with a human — "
    "everything you say will be read aloud, so write exactly as you would speak. "
    "STRICT RULE: Reply in 2 sentences maximum. Never more. "
    "If a topic needs more explanation, summarise it in 2 sentences and stop. "
    "Never use bullet points, lists, markdown, numbers, or symbols. "
    "Never start a response with a filler like 'Certainly!' or 'Of course!'. "
    "Just answer directly, like a knowledgeable friend would in conversation."
)

# ── TTS ──────────────────────────────────────────────────────────────────────
KOKORO_VOICE       = "bf_emma"
KOKORO_SPEED       = 1.15
KOKORO_SAMPLE_RATE = 24000  # kept for reference — tts.py reads sample_rate
                             # from kokoro.create()'s return value directly

# ── Pipeline ─────────────────────────────────────────────────────────────────
MIN_PHRASE_CHARS = 15

# Lowered from 200 — shorter ceiling means Kokoro gets cleaner chunks and
# the word-boundary split in tts.py has less distance to walk back.
MAX_PHRASE_CHARS = 120

# Minimum RMS of the full audio buffer before sending to Groq Whisper.
# Buffers below this threshold are near-silence (echo, noise) — Whisper
# hallucinates plausible phrases on quiet input ("Thanks for watching!", "you").
MIN_AUDIO_ENERGY = 0.025

QUEUE_MAX_SIZE   = 100