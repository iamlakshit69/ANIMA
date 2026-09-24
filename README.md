# ANIMA 🎙️

A real-time voice assistant running on the `cloud` branch, powered by Groq's low-latency APIs.

## Architecture (Cloud Branch)
- **VAD**: Energy-based Voice Activity Detection (`pipeline/mic.py`)
- **STT**: Groq Whisper Large v3 Turbo (`pipeline/stt.py`)
- **LLM**: Groq Meta LLaMA 4 Scout 17b (`pipeline/llm.py`)
- **TTS**: Groq CanopyLabs Orpheus (`pipeline/tts.py`)
- **Audio Output**: Sounddevice playback with barge-in support (`pipeline/speaker.py`)

---

## Quickstart

### 1. Set your API Key
Open `.env` in the root folder and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
> If you don't have a key, get one for free at [console.groq.com/keys](https://console.groq.com/keys).

### 2. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 3. Run the Assistant
```bash
python main.py
```
Or directly without activating:
```bash
./venv/bin/python main.py
```

Press `Ctrl+C` to stop the assistant.
