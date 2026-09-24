import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
CHUNK_SIZE       = 512
CHANNELS         = 1

# ── VAD ──────────────────────────────────────────────────────────────────────
VAD_THRESHOLD    = 0.5   
SILENCE_DURATION = 0.5   
# ── STT ──────────────────────────────────────────────────────────────────────
WHISPER_LANGUAGE   = "en"
WHISPER_MODEL_SIZE = "base.en"

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MAX_TOKENS   = 120
LLM_TEMPERATURE  = 0.4
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
# Options: "en_US-lessac-low" (~35ms latency, 16kHz) or "en_US-lessac-medium" (~85ms latency, 22.05kHz)
PIPER_VOICE_NAME   = "en_US-lessac-low"
PIPER_MODEL_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "piper_models", f"{PIPER_VOICE_NAME}.onnx")
PIPER_CONFIG_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "piper_models", f"{PIPER_VOICE_NAME}.onnx.json")

# Kokoro (legacy fallback)
KOKORO_VOICE       = "bf_emma"
KOKORO_SPEED       = 1.15
KOKORO_SAMPLE_RATE = 24000

# ── Pipeline ─────────────────────────────────────────────────────────────────
MIN_PHRASE_CHARS = 15
MAX_PHRASE_CHARS = 120
QUEUE_MAX_SIZE   = 100