import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
CHANNELS = 1

VAD_THRESHOLD = 0.5
SILENCE_DURATION = 0.8

WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_LANGUAGE = "en"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MAX_TOKENS = 150
GROQ_TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are a fast, helpful voice assistant. "
    "Keep all responses short and conversational — "
    "no bullet points, no markdown, no long explanations. "
    "Speak like a human, not a document."
)

KOKORO_VOICE = "af_sarah"
KOKORO_SPEED = 1.0
KOKORO_SAMPLE_RATE = 24000

MIN_PHRASE_CHARS = 20
MAX_PHRASE_CHARS = 200
QUEUE_MAX_SIZE = 100