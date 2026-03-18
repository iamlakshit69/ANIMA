"""
diagnostics/collectors/network.py
===================================
Collects network diagnostics relevant to Groq API performance:
  - Ping latency to Groq API endpoint (min/mean/max over multiple samples)
  - HTTPS connection establishment time (TCP + TLS handshake)
  - API key validity check
  - Estimated upload bandwidth (relevant for Whisper WAV uploads)
  - Estimated download bandwidth (relevant for LLM token streaming)
  - DNS resolution time for api.groq.com
"""

import asyncio
import os
import socket
import ssl
import time
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import GROQ_API_KEY

GROQ_HOST     = "api.groq.com"
GROQ_BASE_URL = f"https://{GROQ_HOST}"
PING_SAMPLES  = 10   # number of HTTPS pings to average
BANDWIDTH_MB  = 2    # payload size for bandwidth estimate (MB)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dns_lookup(host: str) -> dict:
    """Measure DNS resolution time."""
    t0 = time.monotonic()
    try:
        ip = socket.gethostbyname(host)
        latency_ms = (time.monotonic() - t0) * 1000
        return {
            "host":       host,
            "ip":         ip,
            "latency_ms": round(latency_ms, 2),
            "success":    True,
        }
    except Exception as e:
        return {
            "host":    host,
            "error":   str(e),
            "success": False,
        }


def _tls_handshake_time(host: str, port: int = 443) -> dict:
    """Measure raw TCP + TLS handshake time (no HTTP)."""
    t0 = time.monotonic()
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=host):
                latency_ms = (time.monotonic() - t0) * 1000
        return {
            "latency_ms": round(latency_ms, 2),
            "success":    True,
        }
    except Exception as e:
        return {
            "error":   str(e),
            "success": False,
        }


async def _https_ping(client: httpx.AsyncClient, url: str) -> float | None:
    """Single HTTPS GET round trip time in ms."""
    try:
        t0 = time.monotonic()
        r  = await client.get(url, timeout=10)
        return (time.monotonic() - t0) * 1000
    except Exception:
        return None


async def _measure_ping(n: int = PING_SAMPLES) -> dict:
    """Measure HTTPS ping to Groq over multiple samples."""
    # Ping the openapi spec endpoint — lightweight, no auth needed
    url = f"{GROQ_BASE_URL}/openai/v1/models"
    samples = []

    async with httpx.AsyncClient() as client:
        # Warmup — first request includes connection setup
        await _https_ping(client, url)

        for _ in range(n):
            ms = await _https_ping(client, url)
            if ms is not None:
                samples.append(ms)
            await asyncio.sleep(0.05)

    if not samples:
        return {"success": False, "error": "all pings failed"}

    import numpy as np
    arr = np.array(samples)
    return {
        "success":    True,
        "samples":    n,
        "min_ms":     round(float(arr.min()),  2),
        "mean_ms":    round(float(arr.mean()), 2),
        "max_ms":     round(float(arr.max()),  2),
        "std_ms":     round(float(arr.std()),  2),
        "jitter_ms":  round(float(arr.std()),  2),  # std dev ≈ jitter
    }


async def _check_api_key() -> dict:
    """Verify GROQ_API_KEY is set and accepted by the API."""
    if not GROQ_API_KEY:
        return {
            "valid":   False,
            "error":   "GROQ_API_KEY not set in environment / .env",
            "latency_ms": None,
        }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{GROQ_BASE_URL}/openai/v1/models",
                headers=headers,
                timeout=10,
            )
        latency_ms = (time.monotonic() - t0) * 1000

        if r.status_code == 200:
            models = [m["id"] for m in r.json().get("data", [])]
            return {
                "valid":       True,
                "latency_ms":  round(latency_ms, 2),
                "status_code": r.status_code,
                "models_available": len(models),
                "models": models[:10],  # first 10
            }
        elif r.status_code == 401:
            return {
                "valid":       False,
                "error":       "Invalid API key (401 Unauthorized)",
                "latency_ms":  round(latency_ms, 2),
                "status_code": r.status_code,
            }
        else:
            return {
                "valid":       False,
                "error":       f"Unexpected status {r.status_code}",
                "latency_ms":  round(latency_ms, 2),
                "status_code": r.status_code,
            }
    except Exception as e:
        return {
            "valid":   False,
            "error":   str(e),
            "latency_ms": None,
        }


async def _estimate_upload_bandwidth() -> dict:
    """
    Estimate upload bandwidth by POST-ing a payload to a public echo endpoint.
    Relevant because Groq STT requires uploading a WAV file on every turn.
    Typical WAV sizes: 1s=32KB, 3s=96KB, 5s=160KB at 16kHz 16-bit mono.
    """
    payload = os.urandom(BANDWIDTH_MB * 1024 * 1024)

    try:
        t0 = time.monotonic()
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://httpbin.org/post",
                content=payload,
                headers={"Content-Type": "application/octet-stream"},
                timeout=30,
            )
        elapsed = time.monotonic() - t0
        speed_mbps = (BANDWIDTH_MB * 8) / elapsed  # megabits per second

        return {
            "success":         True,
            "payload_mb":      BANDWIDTH_MB,
            "elapsed_s":       round(elapsed, 2),
            "speed_mbps":      round(speed_mbps, 2),
            "speed_MBs":       round(BANDWIDTH_MB / elapsed, 2),
            # Estimate time to upload a typical STT WAV
            "est_1s_wav_ms":   round(32  / 1024 / (BANDWIDTH_MB / elapsed) * 1000, 1),
            "est_3s_wav_ms":   round(96  / 1024 / (BANDWIDTH_MB / elapsed) * 1000, 1),
            "est_5s_wav_ms":   round(160 / 1024 / (BANDWIDTH_MB / elapsed) * 1000, 1),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _estimate_download_bandwidth() -> dict:
    """
    Estimate download bandwidth — relevant for LLM token streaming throughput.
    """
    url = "https://speed.cloudflare.com/__down?bytes=2097152"  # 2MB

    try:
        t0 = time.monotonic()
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=30)
            _ = r.content
        elapsed    = time.monotonic() - t0
        size_mb    = len(r.content) / 1e6
        speed_mbps = (size_mb * 8) / elapsed

        return {
            "success":    True,
            "size_mb":    round(size_mb, 2),
            "elapsed_s":  round(elapsed, 2),
            "speed_mbps": round(speed_mbps, 2),
            "speed_MBs":  round(size_mb / elapsed, 2),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Main entry point ──────────────────────────────────────────────────────────

async def collect_async() -> dict:
    print("[network] DNS lookup...")
    dns = _dns_lookup(GROQ_HOST)

    print("[network] TLS handshake timing...")
    tls = _tls_handshake_time(GROQ_HOST)

    print(f"[network] HTTPS ping ({PING_SAMPLES} samples)...")
    ping = await _measure_ping()

    print("[network] checking API key...")
    api_key = await _check_api_key()

    print("[network] estimating upload bandwidth...")
    upload = await _estimate_upload_bandwidth()

    print("[network] estimating download bandwidth...")
    download = await _estimate_download_bandwidth()

    # Summarise
    groq_reachable = dns["success"] and tls["success"] and ping["success"]

    return {
        "groq_reachable":   groq_reachable,
        "dns":              dns,
        "tls_handshake":    tls,
        "ping":             ping,
        "api_key":          api_key,
        "upload_bandwidth": upload,
        "download_bandwidth": download,
    }


def collect() -> dict:
    return asyncio.run(collect_async())


if __name__ == "__main__":
    import json
    results = collect()
    print(json.dumps(results, indent=2))