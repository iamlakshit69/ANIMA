import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

SAMPLE_RATE = 16000
CHUNK_SIZE  = 480
CHANNELS    = 1


SILENCE_DURATION   = 0.4

WHISPER_LANGUAGE    = "en"
GROQ_WHISPER_MODEL  = "whisper-large-v3-turbo"

GROQ_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_MAX_TOKENS  = 100
GROQ_TEMPERATURE = 0.4
SYSTEM_PROMPT = (
    "You are a fast, helpful voice assistant. "
    "You are speaking out loud to a human, not writing text. "
    "Whatever you write will be spoken exactly as written. "
    "No bullet points, no markdown, no lists. "
    "Be concise, warm, and conversational at all times."
)

GROQ_TTS_MODEL       = "canopylabs/orpheus-v1-english"
GROQ_TTS_VOICE       = "diana"
GROQ_TTS_SAMPLE_RATE = 48000

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MIN_PHRASE_CHARS = 15
MAX_PHRASE_CHARS = 200
QUEUE_MAX_SIZE   = 100
