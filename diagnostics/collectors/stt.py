"""
diagnostics/collectors/stt.py
==============================
Benchmarks Groq Whisper STT:
  - Transcription latency at multiple audio lengths (1s, 3s, 5s)
  - Multiple runs per length for min/mean/max stats
  - Real-time factor (latency / audio_duration)
  - Accuracy check on known phrases via Kokoro → Groq round trip
  - WAV size at each length
  - Model availability check
"""

import asyncio
import io
import os
import sys
import time
import wave

import numpy as np
from groq import AsyncGroq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import (
    GROQ_API_KEY,
    GROQ_WHISPER_MODEL,
    WHISPER_LANGUAGE,
    SAMPLE_RATE,
)

# ── Config ────────────────────────────────────────────────────────────────────
RUNS_PER_LENGTH = 3
TEST_LENGTHS_S  = [1.0, 3.0, 5.0]

ACCURACY_PHRASES = [
    "Hello, how are you today?",
    "The weather is nice outside.",
    "My name is John and I am thirty years old.",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_test_audio(duration_s: float, freq: float = 440.0) -> tuple[bytes, int]:
    """Generate a sine tone WAV of given duration for latency benchmarking."""
    n_samples = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32768).astype(np.int16).tobytes())
    wav_bytes = buf.getvalue()
    return wav_bytes, len(wav_bytes)


def _make_speech_audio_from_kokoro(kokoro, phrase: str, voice: str, speed: float) -> bytes:
    """Synthesize real speech WAV for accuracy testing."""
    samples, sr = kokoro.create(phrase, voice=voice, speed=speed, lang="en-us")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((samples * 32768).astype(np.int16).tobytes())
    return buf.getvalue()


async def _transcribe(client: AsyncGroq, wav_bytes: bytes, filename: str = "audio.wav") -> tuple[str, float]:
    """Send WAV bytes to Groq Whisper, return (transcript, latency_s)."""
    buf = io.BytesIO(wav_bytes)
    t0  = time.monotonic()
    result = await client.audio.transcriptions.create(
        model=GROQ_WHISPER_MODEL,
        file=(filename, buf, "audio/wav"),
        language=WHISPER_LANGUAGE,
    )
    return result.text.strip(), time.monotonic() - t0


# ── Sub-collectors ────────────────────────────────────────────────────────────

async def _benchmark_latency(client: AsyncGroq) -> dict:
    """Benchmark transcription latency at 1s, 3s, 5s audio lengths."""
    results = {}

    for duration in TEST_LENGTHS_S:
        wav_bytes, wav_size = _make_test_audio(duration)
        latencies = []

        print(f"[stt] benchmarking {duration:.0f}s audio "
              f"({wav_size/1024:.1f}KB) x{RUNS_PER_LENGTH} runs...")

        for run in range(RUNS_PER_LENGTH):
            _, latency = await _transcribe(client, wav_bytes, f"test_{duration}s.wav")
            latencies.append(latency)
            print(f"[stt]   run {run+1}: {latency:.3f}s")
            await asyncio.sleep(0.1)

        arr = np.array(latencies)
        results[f"{duration:.0f}s"] = {
            "audio_duration_s": duration,
            "wav_size_kb":      round(wav_size / 1024, 1),
            "runs":             RUNS_PER_LENGTH,
            "min_s":            round(float(arr.min()),  3),
            "mean_s":           round(float(arr.mean()), 3),
            "max_s":            round(float(arr.max()),  3),
            "std_s":            round(float(arr.std()),  3),
            # Real-time factor: how many seconds of processing per second of audio
            # < 0.1 is excellent, > 0.5 means Groq is struggling
            "realtime_factor":  round(float(arr.mean()) / duration, 3),
        }

    return results


async def _benchmark_accuracy(client: AsyncGroq) -> dict:
    """Synthesize known phrases via Kokoro, transcribe via Groq, check accuracy."""
    try:
        from kokoro_onnx import Kokoro
        from config.settings import KOKORO_VOICE, KOKORO_SPEED
    except ImportError:
        return {"error": "kokoro_onnx not available for accuracy test"}

    print("[stt] running accuracy test with real Kokoro speech...")
    kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")
    await asyncio.to_thread(
        kokoro.create, "Hello.",
        voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us",
    )

    results = []
    for phrase in ACCURACY_PHRASES:
        print(f"[stt]   phrase: '{phrase}'")
        wav_bytes = await asyncio.to_thread(
            _make_speech_audio_from_kokoro, kokoro,
            phrase, KOKORO_VOICE, KOKORO_SPEED,
        )
        transcript, latency = await _transcribe(client, wav_bytes)

        # Word-level accuracy (simple, not full WER)
        def _clean(s):
            return s.lower().replace(",", "").replace(".", "").replace("?", "").split()

        expected_words   = _clean(phrase)
        transcript_words = _clean(transcript)
        matches  = sum(1 for w in transcript_words if w in expected_words)
        accuracy = round(matches / max(len(expected_words), 1) * 100, 1)

        results.append({
            "expected":     phrase,
            "got":          transcript,
            "latency_s":    round(latency, 3),
            "accuracy_pct": accuracy,
            "exact_match":  _clean(phrase) == _clean(transcript),
        })
        await asyncio.sleep(0.1)

    overall = round(
        sum(r["accuracy_pct"] for r in results) / len(results), 1
    ) if results else 0.0

    return {
        "overall_accuracy_pct": overall,
        "phrases": results,
    }


async def _check_model_info(client: AsyncGroq) -> dict:
    """Confirm configured Whisper model is available on Groq."""
    try:
        t0     = time.monotonic()
        models = await client.models.list()
        latency = time.monotonic() - t0
        available      = [m.id for m in models.data]
        whisper_models = [m for m in available if "whisper" in m.lower()]
        return {
            "configured_model": GROQ_WHISPER_MODEL,
            "model_available":  GROQ_WHISPER_MODEL in available,
            "whisper_models":   whisper_models,
            "api_latency_ms":   round(latency * 1000, 2),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Main entry point ──────────────────────────────────────────────────────────

async def collect_async() -> dict:
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set — skipping STT benchmark"}

    client = AsyncGroq(api_key=GROQ_API_KEY)

    print("[stt] checking model availability...")
    model_info = await _check_model_info(client)

    print("[stt] benchmarking transcription latency...")
    latency = await _benchmark_latency(client)

    print("[stt] running accuracy test...")
    accuracy = await _benchmark_accuracy(client)

    mean_latencies = [v["mean_s"] for v in latency.values()]
    avg_latency    = round(sum(mean_latencies) / len(mean_latencies), 3) if mean_latencies else None

    return {
        "model_info":        model_info,
        "latency_by_length": latency,
        "accuracy":          accuracy,
        "summary": {
            "avg_latency_s": avg_latency,
            "model":         GROQ_WHISPER_MODEL,
            "language":      WHISPER_LANGUAGE,
            "status": (
                "good"    if avg_latency and avg_latency < 0.5 else
                "warning" if avg_latency and avg_latency < 1.0 else
                "bad"
            ),
        },
    }


def collect() -> dict:
    return asyncio.run(collect_async())


if __name__ == "__main__":
    import json
    results = collect()
    print(json.dumps(results, indent=2))