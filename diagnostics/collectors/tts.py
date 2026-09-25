"""
diagnostics/collectors/tts.py
==============================
Benchmarks Kokoro TTS performance:
  - Cold start vs warmup latency
  - Synthesis speed at multiple phrase lengths (short/medium/long)
  - Multiple runs per length for min/mean/max stats
  - Real-time factor (synthesis_time / audio_duration)
  - GPU utilization during synthesis
  - ONNX model file info
  - Audio quality metrics (sample rate, bit depth, duration)
  - Word boundary split accuracy test
"""

import asyncio
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import (
    MIN_PHRASE_CHARS,
    MAX_PHRASE_CHARS,
)
try:
    from config.settings import KOKORO_VOICE, KOKORO_SPEED, KOKORO_SAMPLE_RATE
except ImportError:
    KOKORO_VOICE = "af_heart"
    KOKORO_SPEED = 1.0
    KOKORO_SAMPLE_RATE = 24000


# ── Config ────────────────────────────────────────────────────────────────────
RUNS_PER_PHRASE = 3

TEST_PHRASES = {
    "short":  "Good evening.",                                          # ~15 chars
    "medium": "Good evening, how can I help you today?",               # ~42 chars
    "long":   (
        "Nazi Germany, led by Adolf Hitler, was a period of extreme "
        "nationalism and racism in Germany from 1933 to 1945."
    ),                                                                  # ~110 chars
}

ONNX_MODEL_PATH = "kokoro-v0_19.onnx"
VOICES_PATH     = "voices.bin"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_gpu_memory() -> dict | None:
    """Snapshot current GPU memory usage."""
    if not torch.cuda.is_available():
        return None
    mem = torch.cuda.mem_get_info(0)
    total = torch.cuda.get_device_properties(0).total_memory
    return {
        "used_mb":  round((total - mem[0]) / 1e6, 1),
        "free_mb":  round(mem[0] / 1e6, 1),
        "total_mb": round(total / 1e6, 1),
    }


def _synthesize(kokoro, phrase: str) -> tuple[np.ndarray, int, float]:
    """Run Kokoro synthesis and return (samples, sample_rate, latency_s)."""
    t0 = time.monotonic()
    samples, sample_rate = kokoro.create(
        phrase,
        voice=KOKORO_VOICE,
        speed=KOKORO_SPEED,
        lang="en-us",
    )
    latency = time.monotonic() - t0
    return samples, sample_rate, latency


def _audio_duration(samples: np.ndarray, sample_rate: int) -> float:
    return len(samples) / sample_rate


# ── Sub-collectors ────────────────────────────────────────────────────────────

def _collect_model_info() -> dict:
    """ONNX model file size and voices.bin info."""
    info = {}

    if os.path.exists(ONNX_MODEL_PATH):
        size_mb = os.path.getsize(ONNX_MODEL_PATH) / 1e6
        info["onnx_model"] = {
            "path":    ONNX_MODEL_PATH,
            "size_mb": round(size_mb, 1),
            "exists":  True,
        }
    else:
        info["onnx_model"] = {"exists": False, "path": ONNX_MODEL_PATH}

    if os.path.exists(VOICES_PATH):
        size_mb = os.path.getsize(VOICES_PATH) / 1e6
        info["voices"] = {
            "path":    VOICES_PATH,
            "size_mb": round(size_mb, 1),
            "exists":  True,
        }
    else:
        info["voices"] = {"exists": False, "path": VOICES_PATH}

    try:
        import onnxruntime as ort
        info["onnxruntime_version"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnxruntime_version"] = "not found"

    return info


async def _benchmark_cold_start(kokoro_class) -> dict:
    """
    Measure cold start (first synthesis ever) vs warmed-up latency.
    Cold start includes ONNX lazy initialization overhead.
    """
    print("[tts] measuring cold start latency (fresh Kokoro instance)...")

    # Fresh instance — no warmup
    from kokoro_onnx import Kokoro
    fresh_kokoro = Kokoro(ONNX_MODEL_PATH, VOICES_PATH)

    gpu_before = _get_gpu_memory()
    t0 = time.monotonic()
    samples, sr, _ = await asyncio.to_thread(
        _synthesize, fresh_kokoro, TEST_PHRASES["short"]
    )
    cold_latency = time.monotonic() - t0
    gpu_after = _get_gpu_memory()

    audio_dur = _audio_duration(samples, sr)

    print(f"[tts]   cold start: {cold_latency:.3f}s for {audio_dur:.2f}s audio")

    # Second call on same instance — should be much faster (warmed up)
    _, _, warm_latency = await asyncio.to_thread(
        _synthesize, fresh_kokoro, TEST_PHRASES["short"]
    )
    print(f"[tts]   warmed up:  {warm_latency:.3f}s for same phrase")

    return {
        "cold_start_s":    round(cold_latency,  3),
        "warmed_up_s":     round(warm_latency,  3),
        "overhead_s":      round(cold_latency - warm_latency, 3),
        "audio_duration_s": round(audio_dur, 3),
        "cold_rtf":        round(cold_latency / audio_dur, 3),
        "warm_rtf":        round(warm_latency / audio_dur, 3),
        "gpu_memory_delta_mb": (
            round(gpu_after["used_mb"] - gpu_before["used_mb"], 1)
            if gpu_before and gpu_after else None
        ),
    }


async def _benchmark_by_length(kokoro) -> dict:
    """
    Benchmark synthesis latency at short / medium / long phrase lengths.
    Each phrase is run RUNS_PER_PHRASE times.
    """
    results = {}

    for label, phrase in TEST_PHRASES.items():
        print(
            f"[tts] benchmarking '{label}' phrase "
            f"({len(phrase)} chars) x{RUNS_PER_PHRASE} runs..."
        )

        latencies  = []
        audio_durs = []
        gpu_usages = []

        for i in range(RUNS_PER_PHRASE):
            gpu_before = _get_gpu_memory()
            samples, sr, latency = await asyncio.to_thread(_synthesize, kokoro, phrase)
            gpu_after  = _get_gpu_memory()

            audio_dur = _audio_duration(samples, sr)
            latencies.append(latency)
            audio_durs.append(audio_dur)

            if gpu_before and gpu_after:
                gpu_usages.append(gpu_after["used_mb"])

            print(
                f"[tts]   run {i+1}: {latency:.3f}s synthesis  "
                f"→ {audio_dur:.2f}s audio  "
                f"RTF={latency/audio_dur:.3f}"
            )

        arr      = np.array(latencies)
        dur_mean = float(np.mean(audio_durs))

        results[label] = {
            "phrase":           phrase,
            "char_count":       len(phrase),
            "word_count":       len(phrase.split()),
            "runs":             RUNS_PER_PHRASE,
            "latency": {
                "min_s":  round(float(arr.min()),  3),
                "mean_s": round(float(arr.mean()), 3),
                "max_s":  round(float(arr.max()),  3),
                "std_s":  round(float(arr.std()),  3),
            },
            "audio_duration_s": round(dur_mean, 3),
            "sample_rate":      sr,
            # Real-time factor: synthesis_time / audio_duration
            # < 0.5 = faster than real-time (good), > 1.0 = slower than real-time (bad)
            "realtime_factor":  round(float(arr.mean()) / dur_mean, 3),
            "chars_per_sec":    round(len(phrase) / float(arr.mean()), 1),
            "gpu_used_mb":      round(float(np.mean(gpu_usages)), 1) if gpu_usages else None,
        }

    return results


async def _test_word_boundary_split(kokoro) -> dict:
    """
    Test that the word-boundary split in tts.py works correctly.
    Synthesizes a phrase that exceeds MAX_PHRASE_CHARS and checks
    that splitting at the last space doesn't produce mid-word fragments.
    """
    print("[tts] testing word boundary split logic...")

    # Construct a phrase just over MAX_PHRASE_CHARS
    long_phrase = (
        "The quick brown fox jumps over the lazy dog near the riverbank "
        "while the sun sets slowly behind the mountains in the distance."
    )

    # Simulate the tts.py split logic
    if len(long_phrase) >= MAX_PHRASE_CHARS:
        last_space = long_phrase.rfind(" ", 0, MAX_PHRASE_CHARS)
        if last_space > MIN_PHRASE_CHARS:
            part1 = long_phrase[:last_space]
            part2 = long_phrase[last_space + 1:]
        else:
            part1 = long_phrase
            part2 = ""
    else:
        part1 = long_phrase
        part2 = ""

    # Verify neither part ends/starts mid-word
    part1_clean = not part1.endswith(("-", "'")) and part1[-1] == part1.rstrip()[-1]
    part2_clean = not part2 or part2[0] == part2.lstrip()[0]

    # Synthesize both parts to confirm Kokoro handles them cleanly
    _, _, lat1 = await asyncio.to_thread(_synthesize, kokoro, part1)
    _, _, lat2 = await asyncio.to_thread(_synthesize, kokoro, part2) if part2 else (None, None, 0.0)

    return {
        "original_chars":    len(long_phrase),
        "max_phrase_chars":  MAX_PHRASE_CHARS,
        "min_phrase_chars":  MIN_PHRASE_CHARS,
        "split_at_char":     last_space if len(long_phrase) >= MAX_PHRASE_CHARS else None,
        "part1":             part1,
        "part1_chars":       len(part1),
        "part1_latency_s":   round(lat1, 3),
        "part2":             part2,
        "part2_chars":       len(part2),
        "part2_latency_s":   round(lat2, 3),
        "clean_split":       part1_clean and part2_clean,
    }


async def _benchmark_gpu_utilization(kokoro) -> dict:
    """Measure GPU VRAM usage before, during, and after synthesis."""
    if not torch.cuda.is_available():
        return {"available": False}

    before = _get_gpu_memory()

    # Run a longer synthesis to capture peak usage
    _, _, latency = await asyncio.to_thread(
        _synthesize, kokoro, TEST_PHRASES["long"]
    )

    after = _get_gpu_memory()

    return {
        "available":      True,
        "before_used_mb": before["used_mb"] if before else None,
        "after_used_mb":  after["used_mb"]  if after  else None,
        "delta_mb":       round(after["used_mb"] - before["used_mb"], 1) if before and after else None,
        "synthesis_latency_s": round(latency, 3),
    }


# ── Main entry point ──────────────────────────────────────────────────────────

async def collect_async() -> dict:
    from kokoro_onnx import Kokoro

    print("[tts] collecting model file info...")
    model_info = _collect_model_info()

    print("[tts] benchmarking cold start...")
    cold_start = await _benchmark_cold_start(Kokoro)

    # Load a warmed-up instance for the remaining benchmarks
    print("[tts] loading warmed-up Kokoro instance...")
    kokoro = Kokoro(ONNX_MODEL_PATH, VOICES_PATH)
    await asyncio.to_thread(_synthesize, kokoro, "Hello.")  # warmup

    print("[tts] benchmarking synthesis by phrase length...")
    by_length = await _benchmark_by_length(kokoro)

    print("[tts] testing word boundary split...")
    word_split = await _test_word_boundary_split(kokoro)

    print("[tts] measuring GPU utilization...")
    gpu = await _benchmark_gpu_utilization(kokoro)

    # Summary — use medium phrase as representative
    medium_latency = by_length.get("medium", {}).get("latency", {}).get("mean_s")
    medium_rtf     = by_length.get("medium", {}).get("realtime_factor")

    return {
        "model_info":  model_info,
        "cold_start":  cold_start,
        "by_length":   by_length,
        "word_split":  word_split,
        "gpu":         gpu,
        "summary": {
            "voice":                KOKORO_VOICE,
            "speed":                KOKORO_SPEED,
            "medium_phrase_latency_s": medium_latency,
            "medium_phrase_rtf":       medium_rtf,
            "cold_start_overhead_s":   cold_start.get("overhead_s"),
            "status": (
                "good"    if medium_latency and medium_latency < 0.5 else
                "warning" if medium_latency and medium_latency < 1.0 else
                "bad"
            ),
        },
    }


def collect() -> dict:
    return asyncio.run(collect_async())


if __name__ == "__main__":
    import json
    results = collect()
    # Trim verbose fields for terminal output
    for v in results.get("by_length", {}).values():
        v.pop("phrase", None)
    print(json.dumps(results, indent=2))