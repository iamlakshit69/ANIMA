import os
from dotenv import load_dotenv

# Load .env from project root, config directory, or current directory
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_root_dir, ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

# Audio Hardware & Format
SAMPLE_RATE = 16000
CHUNK_SIZE  = 512   # 32ms frames at 16kHz (optimal for Silero VAD and sounddevice)
CHANNELS    = 1

# Explicit device selection: None uses system default; integer index or string name
_mic_env = os.getenv("MIC_DEVICE", None)
MIC_DEVICE: int | str | None = int(_mic_env) if (_mic_env is not None and _mic_env.isdigit()) else _mic_env

_spk_env = os.getenv("SPEAKER_DEVICE", None)
SPEAKER_DEVICE: int | str | None = int(_spk_env) if (_spk_env is not None and _spk_env.isdigit()) else _spk_env

# Echo Cancellation & Barge-in mode
# HALF_DUPLEX: mutes mic during assistant speech. Completely prevents speaker echo
# from leaking into the microphone without fragile heuristic timers.
HALF_DUPLEX: bool = True

# Speech Detection & Silence Timing
VAD_THRESHOLD              = float(os.getenv("VAD_THRESHOLD", "0.5"))                # Silero VAD speech onset threshold
VAD_CONTINUATION_THRESHOLD = float(os.getenv("VAD_CONTINUATION_THRESHOLD", "0.25")) # Silero VAD continuation threshold during active speech
ENERGY_THRESHOLD           = float(os.getenv("ENERGY_THRESHOLD", "0.015"))           # RMS fallback threshold
MIN_SPEECH_FRAMES          = int(os.getenv("MIN_SPEECH_FRAMES", "2"))                # consecutive speech frames required before confirming speech onset
SILENCE_DURATION           = float(os.getenv("SILENCE_DURATION", "1.5"))             # seconds of silence to finalize utterance (1.5s allows full natural sentences)
MIN_AUDIO_DURATION         = 0.5   # minimum audio duration in seconds
SILENCE_RMS                = 0.005 # RMS below which a chunk is dead silence
MIN_AUDIO_ENERGY           = 0.01  # minimum RMS of buffer

# Cooldown guards (used for legacy or non-half-duplex mode)
ECHO_DECAY_PAD   = 0.5
BARGE_IN_FRAMES  = 8
POST_SPEECH_MUTE = 0.5
BUFFER_MUTE_GUARD = 0.5

# Cross-file invariant check
assert BUFFER_MUTE_GUARD >= POST_SPEECH_MUTE, (
    f"BUFFER_MUTE_GUARD ({BUFFER_MUTE_GUARD}) must be >= POST_SPEECH_MUTE ({POST_SPEECH_MUTE}) "
    "or echo will leak through"
)

# Local STT (Faster-Whisper — blazing fast on-device inference via int8 NEON)
WHISPER_LANGUAGE     = "en"
LOCAL_WHISPER_MODEL  = os.getenv("LOCAL_WHISPER_MODEL", "base.en")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_DEVICE       = os.getenv("WHISPER_DEVICE", "cpu")

# Groq LLM
GROQ_MODEL       = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_MAX_TOKENS  = 100    # concise conversational responses to prevent monopolizing audio
GROQ_TEMPERATURE = 0.4
SYSTEM_PROMPT = (
    "You are a helpful spoken voice assistant named ANIMA. "
    "You are speaking out loud to a human, not writing text. "
    "Answer directly in 1 to 2 short sentences unless the user explicitly requests more detail. "
    "Never use markdown, lists, bullet points, asterisks, or formatting."
)

# Kokoro TTS (Local, Ultra-fast, High-fidelity Neural Speech)
KOKORO_MODEL_PATH  = os.getenv("KOKORO_MODEL_PATH", "kokoro-v0_19.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "voices.bin")
KOKORO_VOICE       = os.getenv("KOKORO_VOICE", "af_bella")  # natural, warm, expressive female voice
KOKORO_SPEED       = float(os.getenv("KOKORO_SPEED", "1.05"))
KOKORO_LANG        = os.getenv("KOKORO_LANG", "en-us")

# Auth
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Pipeline limits
MIN_PHRASE_CHARS = 15
MAX_PHRASE_CHARS = 200
QUEUE_MAX_SIZE   = 100

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = os.getenv("LOG_FILE", "anima.log")
