"""
diagnostics/report/json_report.py
===================================
Generates a machine-readable JSON diagnostic report from all collector results.
Includes metadata (timestamp, git hash, hostname) for regression tracking.

Output: diagnostics/results/diagnostic_report_<timestamp>.json
"""

import os
import json
import datetime
import socket
import subprocess
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _build_meta() -> dict:
    now = datetime.datetime.now()
    return {
        "timestamp":      now.isoformat(),
        "timestamp_unix": now.timestamp(),
        "hostname":       socket.gethostname(),
        "git_hash":       _git_hash(),
        "git_branch":     _git_branch(),
        "python_version": sys.version.split()[0],
    }


def _build_summary(results: dict) -> dict:
    """
    Flattened summary of the most important metrics.
    Useful for quick comparisons across multiple JSON reports
    without having to dig into nested structures.
    """
    stt  = results.get("stt",      {})
    llm  = results.get("llm",      {})
    tts  = results.get("tts",      {})
    net  = results.get("network",  {})
    hw   = results.get("hardware", {})
    audio = results.get("audio",   {})

    gpu_name = None
    gpu_vram = None
    if hw.get("gpu", {}).get("available") and hw["gpu"].get("gpus"):
        g = hw["gpu"]["gpus"][0]
        gpu_name = g.get("name")
        gpu_vram = g.get("vram_total_gb")

    return {
        # Pipeline latency (seconds)
        "pipeline": {
            "stt_mean_latency_s":    stt.get("summary", {}).get("avg_latency_s"),
            "llm_short_ttft_s":      llm.get("summary", {}).get("short_prompt_ttft_s"),
            "tts_medium_latency_s":  tts.get("summary", {}).get("medium_phrase_latency_s"),
            "tts_medium_rtf":        tts.get("summary", {}).get("medium_phrase_rtf"),
        },
        # Network
        "network": {
            "groq_ping_mean_ms":  net.get("ping", {}).get("mean_ms"),
            "groq_ping_jitter_ms":net.get("ping", {}).get("jitter_ms"),
            "upload_mbps":        net.get("upload_bandwidth",   {}).get("speed_mbps"),
            "download_mbps":      net.get("download_bandwidth", {}).get("speed_mbps"),
            "api_key_valid":      net.get("api_key", {}).get("valid", False),
        },
        # Hardware
        "hardware": {
            "gpu_name":        gpu_name,
            "gpu_vram_gb":     gpu_vram,
            "ram_total_gb":    hw.get("ram", {}).get("total_gb"),
            "ram_usage_pct":   hw.get("ram", {}).get("usage_percent"),
            "disk_read_mbs":   hw.get("disk", {}).get("read_speed_mbs"),
            "cpu_cores":       hw.get("cpu", {}).get("physical_cores"),
        },
        # Audio
        "audio": {
            "noise_floor_mean_rms":      audio.get("noise_floor", {}).get("mean_rms"),
            "echo_decay_time_s":         audio.get("echo_decay",  {}).get("echo_decay_time_s"),
            "echo_fully_decayed":        audio.get("echo_decay",  {}).get("echo_fully_decayed"),
            "vad_false_positive_pct":    audio.get("vad_false_positives", {}).get("false_positive_pct"),
            "recommended_min_audio_energy": audio.get("noise_floor", {}).get("recommended_min_audio_energy"),
            "recommended_buffer_mute_guard": audio.get("echo_decay", {}).get("recommended_buffer_mute_guard"),
            "recommended_post_speech_mute":  audio.get("echo_decay", {}).get("recommended_post_speech_mute"),
        },
        # Health flags
        "health": {
            "gpu_available":       hw.get("gpu", {}).get("available", False),
            "groq_reachable":      net.get("groq_reachable", False),
            "api_key_valid":       net.get("api_key", {}).get("valid", False),
            "stt_status":          stt.get("summary", {}).get("status"),
            "llm_status":          llm.get("summary", {}).get("status"),
            "tts_status":          tts.get("summary", {}).get("status"),
            "echo_clean":          audio.get("echo_decay", {}).get("echo_fully_decayed", False),
            "vad_clean":           audio.get("vad_false_positives", {}).get("status") == "good",
            "system_prompt_compliance_pct": llm.get("compliance", {}).get("compliance_score"),
        },
    }


def _strip_rms_timeline(results: dict) -> dict:
    """
    Remove rms_timeline from the results before serializing —
    it's hundreds of floats and is already captured in the plot.
    Keep everything else intact.
    """
    import copy
    out = copy.deepcopy(results)
    try:
        out["audio"]["echo_decay"].pop("rms_timeline", None)
    except (KeyError, AttributeError):
        pass
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(results: dict, output_dir: str = OUTPUT_DIR) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename  = f"diagnostic_report_{timestamp}.json"
    filepath  = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    meta    = _build_meta()
    summary = _build_summary(results)
    cleaned = _strip_rms_timeline(results)

    report = {
        "meta":    meta,
        "summary": summary,
        "results": cleaned,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[json] saved → {filepath}")
    return filepath


def load(filepath: str) -> dict:
    """Load a previously saved JSON report for comparison."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(report_a: dict, report_b: dict) -> dict:
    """
    Compare two JSON reports and return a diff of key summary metrics.
    Useful for tracking regressions between runs.
    """
    def _safe_diff(a, b):
        if a is None or b is None:
            return None
        try:
            return round(float(b) - float(a), 4)
        except (TypeError, ValueError):
            return None

    sum_a = report_a.get("summary", {})
    sum_b = report_b.get("summary", {})

    diff = {}
    for section in ["pipeline", "network", "audio"]:
        diff[section] = {}
        for key in sum_a.get(section, {}):
            val_a = sum_a[section].get(key)
            val_b = sum_b[section].get(key)
            delta = _safe_diff(val_a, val_b)
            diff[section][key] = {
                "before": val_a,
                "after":  val_b,
                "delta":  delta,
                "improved": delta < 0 if delta is not None and isinstance(delta, float) else None,
            }

    return {
        "meta_a":  report_a.get("meta", {}),
        "meta_b":  report_b.get("meta", {}),
        "diff":    diff,
    }


if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Compare two reports: python json_report.py report_a.json report_b.json
        report_a = load(sys.argv[1])
        report_b = load(sys.argv[2])
        diff = compare(report_a, report_b)
        print(json.dumps(diff, indent=2))
    else:
        # Smoke test
        path = generate({})
        print(f"Smoke test passed → {path}")