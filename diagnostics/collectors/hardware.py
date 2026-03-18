"""
diagnostics/collectors/hardware.py
====================================
Collects hardware information:
  - CPU (model, cores, frequency, current usage)
  - RAM (total, available, usage %)
  - GPU (name, VRAM total/used, driver version)
  - Disk (read/write speed relevant to ONNX model load)
  - OS / Python / key library versions
  - ONNX runtime info
  - PipeWire / PulseAudio info
"""

import os
import sys
import time
import platform
import subprocess
import importlib.metadata

import psutil
import torch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> str:
    """Run a shell command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        return ""


def _get_package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


# ── Collectors ────────────────────────────────────────────────────────────────

def collect_cpu() -> dict:
    freq = psutil.cpu_freq()
    return {
        "model":          platform.processor() or _run(["lscpu"]).split("\n")[0],
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores":  psutil.cpu_count(logical=True),
        "freq_mhz_current": round(freq.current, 1) if freq else None,
        "freq_mhz_max":     round(freq.max, 1)     if freq else None,
        "usage_percent":  psutil.cpu_percent(interval=0.5),
    }


def collect_ram() -> dict:
    mem = psutil.virtual_memory()
    return {
        "total_gb":     round(mem.total     / 1e9, 2),
        "available_gb": round(mem.available / 1e9, 2),
        "used_gb":      round(mem.used      / 1e9, 2),
        "usage_percent": mem.percent,
    }


def collect_gpu() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem   = torch.cuda.mem_get_info(i)  # (free, total) in bytes
        gpus.append({
            "index":          i,
            "name":           props.name,
            "vram_total_gb":  round(props.total_memory / 1e9, 2),
            "vram_free_gb":   round(mem[0] / 1e9, 2),
            "vram_used_gb":   round((props.total_memory - mem[0]) / 1e9, 2),
            "vram_usage_pct": round((1 - mem[0] / props.total_memory) * 100, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessors": props.multi_processor_count,
        })

    return {
        "available": True,
        "cuda_version": torch.version.cuda,
        "driver_version": _run(["nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader"]).split("\n")[0],
        "gpus": gpus,
    }


def collect_disk() -> dict:
    """
    Measure sequential read speed on the project directory.
    This approximates ONNX model load time which is disk-read-bound on cold start.
    """
    test_file = "/tmp/_diag_disk_test"
    block_size = 64 * 1024 * 1024  # 64 MB

    # Write
    data = os.urandom(block_size)
    t0 = time.monotonic()
    with open(test_file, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    write_speed = block_size / (time.monotonic() - t0) / 1e6  # MB/s

    # Read
    t0 = time.monotonic()
    with open(test_file, "rb") as f:
        _ = f.read()
    read_speed = block_size / (time.monotonic() - t0) / 1e6  # MB/s

    os.unlink(test_file)

    # Disk usage for project partition
    usage = psutil.disk_usage(os.getcwd())

    return {
        "read_speed_mbs":  round(read_speed,  1),
        "write_speed_mbs": round(write_speed, 1),
        "partition_total_gb": round(usage.total / 1e9, 1),
        "partition_free_gb":  round(usage.free  / 1e9, 1),
        "partition_usage_pct": usage.percent,
    }


def collect_audio_system() -> dict:
    """PipeWire / PulseAudio server info and mic device details."""
    pw_info = _run(["pw-cli", "info", "0"])
    pa_info = _run(["pactl", "info"])

    # Extract server name and version
    server_name    = ""
    server_version = ""
    for line in pa_info.splitlines():
        if "Server Name" in line:
            server_name = line.split(":", 1)[-1].strip()
        if "Server Version" in line:
            server_version = line.split(":", 1)[-1].strip()

    # Default source (mic) info
    default_source = ""
    source_volume  = ""
    source_mute    = ""
    in_source      = False
    for line in _run(["pactl", "list", "sources"]).splitlines():
        if "alsa_input" in line and "Name:" in line:
            default_source = line.split(":", 1)[-1].strip()
            in_source = True
        if in_source and "Volume:" in line and "%" in line:
            source_volume = line.strip()
            break
        if in_source and "Mute:" in line:
            source_mute = line.strip()

    return {
        "server_name":    server_name,
        "server_version": server_version,
        "default_source": default_source,
        "source_volume":  source_volume,
        "source_mute":    source_mute,
    }


def collect_versions() -> dict:
    """Key library and runtime versions."""
    return {
        "python":       sys.version.split()[0],
        "os":           f"{platform.system()} {platform.release()}",
        "torch":        torch.__version__,
        "cuda":         torch.version.cuda or "n/a",
        "groq":         _get_package_version("groq"),
        "faster_whisper": _get_package_version("faster-whisper"),
        "kokoro_onnx":  _get_package_version("kokoro-onnx"),
        "silero_vad":   _get_package_version("silero-vad"),
        "pyaudio":      _get_package_version("pyaudio"),
        "sounddevice":  _get_package_version("sounddevice"),
        "onnxruntime":  _get_package_version("onnxruntime-gpu") or
                        _get_package_version("onnxruntime"),
        "numpy":        _get_package_version("numpy"),
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def collect() -> dict:
    print("[hardware] collecting CPU...")
    cpu = collect_cpu()

    print("[hardware] collecting RAM...")
    ram = collect_ram()

    print("[hardware] collecting GPU...")
    gpu = collect_gpu()

    print("[hardware] collecting disk speed...")
    disk = collect_disk()

    print("[hardware] collecting audio system info...")
    audio_system = collect_audio_system()

    print("[hardware] collecting library versions...")
    versions = collect_versions()

    return {
        "cpu":          cpu,
        "ram":          ram,
        "gpu":          gpu,
        "disk":         disk,
        "audio_system": audio_system,
        "versions":     versions,
    }


if __name__ == "__main__":
    import json
    results = collect()
    print(json.dumps(results, indent=2))