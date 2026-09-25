"""
diagnostics/collectors/audio.py
================================
Collects audio pipeline diagnostics:
  - Mic noise floor (RMS of silence)
  - Echo decay curve (plays test audio, records mic simultaneously)
  - VAD false positive rate on silence
  - VAD detection latency on real speech onset
  - PipeWire reported device latency
  - Recommended guard values based on measured decay
"""

import asyncio
import time
import numpy as np
import pyaudio
import sounddevice as sd
import torch
from silero_vad import load_silero_vad
from kokoro_onnx import Kokoro

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import (
    SAMPLE_RATE, CHUNK_SIZE, CHANNELS,
)
try:
    from config.settings import VAD_THRESHOLD, KOKORO_VOICE, KOKORO_SPEED
except ImportError:
    VAD_THRESHOLD = 0.5
    KOKORO_VOICE = "af_heart"
    KOKORO_SPEED = 1.0


# ── Config ────────────────────────────────────────────────────────────────────
SILENCE_MEASURE_SECS  = 3.0   # how long to record silence for noise floor
POST_PLAY_RECORD_SECS = 5.0   # how long to record after playback ends
ECHO_DECAY_THRESHOLD  = 0.02  # MIN_AUDIO_ENERGY equivalent
CONSECUTIVE_CHUNKS    = 5     # chunks below threshold to declare "decayed"
VAD_SILENCE_SECS      = 5.0   # how long to feed silence to VAD for false positive test

TEST_PHRASE = (
    "Good evening, how can I help you today? "
    "I am here to assist you with anything you need. "
    "Feel free to ask me anything at all."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_mic_blocking(stream):
    raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _chunk_rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk ** 2)))


def _open_mic(pa: pyaudio.PyAudio):
    return pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )


# ── Sub-collectors ────────────────────────────────────────────────────────────

async def _measure_noise_floor(pa: pyaudio.PyAudio) -> dict:
    """
    Record silence and measure mic noise floor RMS.

    Two sources of contamination are guarded against:
      1. PyAudio stream open transient — the first ~10 chunks always have
         elevated RMS as the ADC stabilises. Discarded via WARMUP_CHUNKS.
      2. GPU/fan noise from Kokoro warmup — caller must ensure a settling
         delay of at least 1.0s before calling this function.
    """
    print("[audio] measuring mic noise floor (stay quiet)...")

    WARMUP_CHUNKS = 10  # discard first N chunks (stream open transient)

    stream   = _open_mic(pa)
    chunks   = []
    n_chunks = int(SILENCE_MEASURE_SECS * SAMPLE_RATE / CHUNK_SIZE) + WARMUP_CHUNKS

    for i in range(n_chunks):
        chunk = await asyncio.to_thread(_read_mic_blocking, stream)
        if i >= WARMUP_CHUNKS:   # skip transient
            chunks.append(_chunk_rms(chunk))

    stream.stop_stream()
    stream.close()

    rms_values = np.array(chunks)
    return {
        "mean_rms":   round(float(rms_values.mean()), 5),
        "max_rms":    round(float(rms_values.max()),  5),
        "min_rms":    round(float(rms_values.min()),  5),
        "std_rms":    round(float(rms_values.std()),  5),
        # mean + 3σ gives a threshold that is statistically above the noise
        # floor with ~99.7% confidence while still being well below real speech
        "recommended_min_audio_energy": round(float(rms_values.mean() + 3 * rms_values.std()), 4),
    }


async def _measure_echo_decay(pa: pyaudio.PyAudio, kokoro: Kokoro) -> dict:
    """Play test phrase and record mic simultaneously to measure echo decay."""
    print("[audio] synthesizing echo test phrase...")
    samples, sample_rate = await asyncio.to_thread(
        kokoro.create, TEST_PHRASE,
        voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us",
    )
    playback_duration = len(samples) / sample_rate
    total_record_secs = playback_duration + POST_PLAY_RECORD_SECS

    print(f"[audio] playing {playback_duration:.1f}s phrase + recording {POST_PLAY_RECORD_SECS:.1f}s after...")
    stream = _open_mic(pa)

    rms_timeline  = []   # list of (timestamp, rms)
    record_start  = time.monotonic()
    playback_end_t = None

    sd.play(samples, samplerate=sample_rate)

    while True:
        now     = time.monotonic()
        elapsed = now - record_start

        if playback_end_t is None and not sd.get_stream().active:
            playback_end_t = elapsed

        if elapsed >= total_record_secs:
            break

        chunk = await asyncio.to_thread(_read_mic_blocking, stream)
        rms_timeline.append((elapsed, _chunk_rms(chunk)))

    sd.stop()
    stream.stop_stream()
    stream.close()

    if playback_end_t is None:
        playback_end_t = playback_duration

    # Analyse post-playback decay
    post = [(t, r) for t, r in rms_timeline if t > playback_end_t]

    peak_echo = max((r for _, r in post), default=0.0)
    mean_echo = float(np.mean([r for _, r in post])) if post else 0.0

    # Find decay point — first run of CONSECUTIVE_CHUNKS all below threshold
    decay_time = None
    for i in range(len(post) - CONSECUTIVE_CHUNKS):
        window = [r for _, r in post[i:i + CONSECUTIVE_CHUNKS]]
        if all(r < ECHO_DECAY_THRESHOLD for r in window):
            decay_time = post[i][0] - playback_end_t
            break

    recommended_buffer_mute_guard  = round((decay_time or POST_PLAY_RECORD_SECS) + 0.5, 1)
    recommended_post_speech_mute   = round((decay_time or POST_PLAY_RECORD_SECS) + 0.3, 1)

    return {
        "playback_duration_s":          round(playback_duration, 2),
        "post_playback_peak_rms":       round(peak_echo, 4),
        "post_playback_mean_rms":       round(mean_echo, 4),
        "echo_decay_time_s":            round(decay_time, 2) if decay_time else None,
        "echo_fully_decayed":           decay_time is not None,
        "recommended_buffer_mute_guard": recommended_buffer_mute_guard,
        "recommended_post_speech_mute":  recommended_post_speech_mute,
        "rms_timeline":                 [(round(t, 3), round(r, 5)) for t, r in rms_timeline],
        "playback_end_t":               round(playback_end_t, 3),
    }


async def _measure_vad_false_positives(pa: pyaudio.PyAudio, vad_model) -> dict:
    """
    Feed silence into Silero VAD and count how often it fires above VAD_THRESHOLD.
    A high false positive rate means VAD_THRESHOLD needs raising or mic gain is too high.
    """
    print("[audio] testing VAD false positive rate on silence (stay quiet)...")
    stream   = _open_mic(pa)
    n_chunks = int(VAD_SILENCE_SECS * SAMPLE_RATE / CHUNK_SIZE)
    probs    = []

    for _ in range(n_chunks):
        chunk = await asyncio.to_thread(_read_mic_blocking, stream)
        prob  = vad_model(torch.from_numpy(chunk), SAMPLE_RATE).item()
        probs.append(prob)

    stream.stop_stream()
    stream.close()

    probs_arr      = np.array(probs)
    false_positives = int((probs_arr > VAD_THRESHOLD).sum())
    fp_rate         = round(false_positives / len(probs) * 100, 1)

    return {
        "chunks_tested":      len(probs),
        "false_positives":    false_positives,
        "false_positive_pct": fp_rate,
        "mean_vad_prob":      round(float(probs_arr.mean()), 4),
        "max_vad_prob":       round(float(probs_arr.max()),  4),
        "vad_threshold":      VAD_THRESHOLD,
        "status": (
            "good"    if fp_rate == 0   else
            "warning" if fp_rate < 5.0  else
            "bad"
        ),
    }


async def _measure_vad_detection_latency(pa: pyaudio.PyAudio, vad_model) -> dict:
    """
    Measure how many chunks VAD needs before it crosses VAD_THRESHOLD
    on a synthetic tone burst — approximates real speech onset detection speed.
    """
    print("[audio] measuring VAD detection latency on tone burst...")

    # Generate a 440Hz tone at moderate volume
    duration   = 1.0
    t          = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    tone       = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    tone_chunks = [
        tone[i:i + CHUNK_SIZE]
        for i in range(0, len(tone) - CHUNK_SIZE, CHUNK_SIZE)
    ]

    detection_chunk = None
    probs = []
    for i, chunk in enumerate(tone_chunks):
        prob = vad_model(torch.from_numpy(chunk), SAMPLE_RATE).item()
        probs.append(prob)
        if prob > VAD_THRESHOLD and detection_chunk is None:
            detection_chunk = i

    chunk_duration_ms = CHUNK_SIZE / SAMPLE_RATE * 1000

    return {
        "detection_chunk":     detection_chunk,
        "detection_latency_ms": round(detection_chunk * chunk_duration_ms, 1) if detection_chunk else None,
        "chunk_duration_ms":   round(chunk_duration_ms, 1),
        "tone_vad_probs":      [round(p, 4) for p in probs],
    }


def _measure_device_latency() -> dict:
    """Query sounddevice for reported input/output device latency."""
    try:
        default_in  = sd.query_devices(kind="input")
        default_out = sd.query_devices(kind="output")
        return {
            "input_device":           default_in["name"],
            "input_latency_ms":       round(default_in["default_low_input_latency"]  * 1000, 2),
            "output_device":          default_out["name"],
            "output_latency_ms":      round(default_out["default_low_output_latency"] * 1000, 2),
            "input_sample_rate":      int(default_in["default_samplerate"]),
            "output_sample_rate":     int(default_out["default_samplerate"]),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Main entry point ──────────────────────────────────────────────────────────

async def collect_async() -> dict:
    print("[audio] loading VAD model...")
    vad_model = load_silero_vad()

    print("[audio] loading Kokoro for echo test...")
    kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")
    await asyncio.to_thread(
        kokoro.create, "Hello.",
        voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us",
    )

    # Allow GPU fan and system to settle after Kokoro warmup synthesis.
    # Without this the noise floor measurement picks up fan spin-up noise
    # and returns RMS ~0.6 instead of the true ~0.015.
    print("[audio] settling after Kokoro warmup (1.5s)...")
    await asyncio.sleep(1.5)

    pa = pyaudio.PyAudio()

    try:
        device_latency = _measure_device_latency()
        noise_floor    = await _measure_noise_floor(pa)
        vad_fp         = await _measure_vad_false_positives(pa, vad_model)
        vad_latency    = await _measure_vad_detection_latency(pa, vad_model)
        echo_decay     = await _measure_echo_decay(pa, kokoro)
    finally:
        pa.terminate()

    return {
        "device_latency": device_latency,
        "noise_floor":    noise_floor,
        "vad_false_positives": vad_fp,
        "vad_detection_latency": vad_latency,
        "echo_decay":     echo_decay,
    }


def collect() -> dict:
    return asyncio.run(collect_async())


if __name__ == "__main__":
    import json
    results = collect()
    # Don't print rms_timeline — too verbose
    results["echo_decay"].pop("rms_timeline", None)
    print(json.dumps(results, indent=2))