"""
diagnostics/report/markdown.py
================================
Generates a human-readable Markdown diagnostic report from all collector results.

Output: diagnostics/results/diagnostic_report_<timestamp>.md
"""

import os
import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _status_emoji(status: str) -> str:
    return {"good": "✅", "warning": "⚠️", "bad": "❌"}.get(status, "ℹ️")


def _bool_emoji(val: bool) -> str:
    return "✅" if val else "❌"


def _section(title: str, level: int = 2) -> str:
    prefix = "#" * level
    return f"\n{prefix} {title}\n"


def _table(headers: list[str], rows: list[list]) -> str:
    col_widths = [
        max(len(str(headers[i])), max((len(str(row[i])) for row in rows), default=0))
        for i in range(len(headers))
    ]
    sep  = "| " + " | ".join("-" * w for w in col_widths) + " |"
    head = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    )
    return f"{head}\n{sep}\n{body}\n"


def _kv(label: str, value, unit: str = "") -> str:
    return f"- **{label}:** {value}{(' ' + unit) if unit else ''}\n"


# ── Section renderers ─────────────────────────────────────────────────────────

def _render_header(results: dict, timestamp: str) -> str:
    git_hash = results.get("meta", {}).get("git_hash", "unknown")
    out  = f"# Voice Assistant — Diagnostic Report\n\n"
    out += f"> Generated: {timestamp}  \n"
    out += f"> Git commit: `{git_hash}`\n\n"

    out += "## Health Summary\n\n"

    checks = [
        ("GPU available",          results.get("hardware", {}).get("gpu", {}).get("available", False)),
        ("Groq API reachable",     results.get("network",  {}).get("groq_reachable", False)),
        ("API key valid",          results.get("network",  {}).get("api_key", {}).get("valid", False)),
        ("STT latency good",       results.get("stt", {}).get("summary", {}).get("status") == "good"),
        ("LLM TTFT good",          results.get("llm", {}).get("summary", {}).get("status") == "good"),
        ("TTS latency good",       results.get("tts", {}).get("summary", {}).get("status") == "good"),
        ("Echo decays cleanly",    results.get("audio", {}).get("echo_decay", {}).get("echo_fully_decayed", False)),
        ("VAD clean on silence",   results.get("audio", {}).get("vad_false_positives", {}).get("status") == "good"),
        ("System prompt compliant",results.get("llm", {}).get("compliance", {}).get("compliance_score", 0) >= 75),
    ]

    for label, ok in checks:
        out += f"- {_bool_emoji(ok)} {label}\n"

    return out + "\n"


def _render_hardware(hw: dict) -> str:
    if not hw:
        return _section("Hardware") + "_No data collected._\n"

    out  = _section("Hardware")
    cpu  = hw.get("cpu",  {})
    ram  = hw.get("ram",  {})
    gpu  = hw.get("gpu",  {})
    disk = hw.get("disk", {})
    vers = hw.get("versions", {})
    audio_sys = hw.get("audio_system", {})

    out += _section("CPU", 3)
    out += _kv("Model",          cpu.get("model", "?"))
    out += _kv("Cores",          f"{cpu.get('physical_cores', '?')}p / {cpu.get('logical_cores', '?')}t")
    out += _kv("Max frequency",  cpu.get("freq_mhz_max", "?"), "MHz")
    out += _kv("Current usage",  cpu.get("usage_percent", "?"), "%")

    out += _section("RAM", 3)
    out += _kv("Total",     cpu.get("total_gb",     ram.get("total_gb",     "?")), "GB")
    out += _kv("Used",      ram.get("used_gb",      "?"), "GB")
    out += _kv("Available", ram.get("available_gb", "?"), "GB")
    out += _kv("Usage",     ram.get("usage_percent","?"), "%")

    out += _section("GPU", 3)
    if gpu.get("available") and gpu.get("gpus"):
        g = gpu["gpus"][0]
        out += _kv("Name",           g.get("name", "?"))
        out += _kv("VRAM total",     g.get("vram_total_gb", "?"), "GB")
        out += _kv("VRAM used",      g.get("vram_used_gb",  "?"), "GB")
        out += _kv("VRAM usage",     g.get("vram_usage_pct","?"), "%")
        out += _kv("Compute cap",    g.get("compute_capability", "?"))
        out += _kv("CUDA version",   gpu.get("cuda_version",   "?"))
        out += _kv("Driver version", gpu.get("driver_version", "?"))
    else:
        out += "_GPU not available — running in CPU mode._\n"

    out += _section("Disk", 3)
    out += _kv("Read speed",   disk.get("read_speed_mbs",  "?"), "MB/s")
    out += _kv("Write speed",  disk.get("write_speed_mbs", "?"), "MB/s")
    out += _kv("Free space",   disk.get("partition_free_gb","?"), "GB")

    out += _section("Audio System", 3)
    out += _kv("Server",       audio_sys.get("server_name",    "?"))
    out += _kv("Version",      audio_sys.get("server_version", "?"))
    out += _kv("Mic device",   audio_sys.get("default_source", "?"))
    out += _kv("Mic volume",   audio_sys.get("source_volume",  "?"))

    out += _section("Library Versions", 3)
    rows = [[k, v] for k, v in vers.items()]
    out += _table(["Library", "Version"], rows)

    return out


def _render_audio(audio: dict) -> str:
    if not audio:
        return _section("Audio") + "_No data collected._\n"

    out   = _section("Audio")
    noise = audio.get("noise_floor", {})
    echo  = audio.get("echo_decay",  {})
    vad_fp = audio.get("vad_false_positives", {})
    vad_lat = audio.get("vad_detection_latency", {})
    dev   = audio.get("device_latency", {})

    out += _section("Device Latency", 3)
    out += _kv("Input device",    dev.get("input_device",    "?"))
    out += _kv("Input latency",   dev.get("input_latency_ms","?"),  "ms")
    out += _kv("Output device",   dev.get("output_device",   "?"))
    out += _kv("Output latency",  dev.get("output_latency_ms","?"), "ms")

    out += _section("Mic Noise Floor", 3)
    out += _kv("Mean RMS",   noise.get("mean_rms", "?"))
    out += _kv("Max RMS",    noise.get("max_rms",  "?"))
    out += _kv("Std RMS",    noise.get("std_rms",  "?"))
    out += _kv("Recommended `MIN_AUDIO_ENERGY`",
               noise.get("recommended_min_audio_energy", "?"))

    out += _section("Echo Decay", 3)
    out += _kv("Playback duration",      echo.get("playback_duration_s",    "?"), "s")
    out += _kv("Peak RMS after playback",echo.get("post_playback_peak_rms", "?"))
    out += _kv("Mean RMS after playback",echo.get("post_playback_mean_rms", "?"))
    out += _kv("Echo decay time",        echo.get("echo_decay_time_s",      "N/A"), "s")
    out += _kv("Echo fully decayed",     _bool_emoji(echo.get("echo_fully_decayed", False)))
    out += "\n> **Recommendations:**\n"
    out += f"> - `BUFFER_MUTE_GUARD = {echo.get('recommended_buffer_mute_guard', '?')}` seconds\n"
    out += f"> - `POST_SPEECH_MUTE = {echo.get('recommended_post_speech_mute',  '?')}` seconds\n"

    out += _section("VAD False Positive Rate (on silence)", 3)
    fp_status = vad_fp.get("status", "?")
    out += _kv("Status",             f"{_status_emoji(fp_status)} {fp_status}")
    out += _kv("False positives",    f"{vad_fp.get('false_positives', '?')} / {vad_fp.get('chunks_tested', '?')} chunks")
    out += _kv("False positive rate",vad_fp.get("false_positive_pct", "?"), "%")
    out += _kv("Mean VAD prob",      vad_fp.get("mean_vad_prob", "?"))
    out += _kv("Max VAD prob",       vad_fp.get("max_vad_prob",  "?"))
    out += _kv("VAD threshold",      vad_fp.get("vad_threshold", "?"))

    out += _section("VAD Detection Latency", 3)
    det_ms = vad_lat.get("detection_latency_ms")
    out += _kv("Detection latency", f"{det_ms:.1f}" if det_ms else "not detected", "ms")
    out += _kv("Chunk duration",    vad_lat.get("chunk_duration_ms", "?"), "ms")

    return out


def _render_network(net: dict) -> str:
    if not net:
        return _section("Network") + "_No data collected._\n"

    out  = _section("Network")
    dns  = net.get("dns",  {})
    tls  = net.get("tls_handshake", {})
    ping = net.get("ping", {})
    api  = net.get("api_key", {})
    ul   = net.get("upload_bandwidth",   {})
    dl   = net.get("download_bandwidth", {})

    out += _kv("Groq reachable", _bool_emoji(net.get("groq_reachable", False)))

    out += _section("DNS", 3)
    out += _kv("Host",       dns.get("host", "?"))
    out += _kv("Resolved IP",dns.get("ip",   "?"))
    out += _kv("Latency",    dns.get("latency_ms", "?"), "ms")

    out += _section("TLS Handshake", 3)
    out += _kv("Latency", tls.get("latency_ms", "?"), "ms")

    out += _section("HTTPS Ping to Groq API", 3)
    if ping.get("success"):
        out += _table(
            ["Metric", "Value"],
            [
                ["Samples",  ping.get("samples",  "?")],
                ["Min",      f"{ping.get('min_ms',    '?')} ms"],
                ["Mean",     f"{ping.get('mean_ms',   '?')} ms"],
                ["Max",      f"{ping.get('max_ms',    '?')} ms"],
                ["Jitter",   f"{ping.get('jitter_ms', '?')} ms"],
            ]
        )
    else:
        out += "_Ping failed._\n"

    out += _section("API Key", 3)
    out += _kv("Valid",            _bool_emoji(api.get("valid", False)))
    out += _kv("Models available", api.get("models_available", "?"))
    if api.get("error"):
        out += _kv("Error", api["error"])

    out += _section("Bandwidth", 3)
    if ul.get("success"):
        out += _kv("Upload speed",     ul.get("speed_mbps", "?"), "Mbps")
        out += _kv("Est. 1s WAV upload", ul.get("est_1s_wav_ms", "?"), "ms")
        out += _kv("Est. 3s WAV upload", ul.get("est_3s_wav_ms", "?"), "ms")
        out += _kv("Est. 5s WAV upload", ul.get("est_5s_wav_ms", "?"), "ms")
    if dl.get("success"):
        out += _kv("Download speed",   dl.get("speed_mbps", "?"), "Mbps")

    return out


def _render_stt(stt: dict) -> str:
    if not stt:
        return _section("STT (Groq Whisper)") + "_No data collected._\n"

    out     = _section("STT (Groq Whisper)")
    summary = stt.get("summary", {})
    model   = stt.get("model_info", {})
    by_len  = stt.get("latency_by_length", {})
    acc     = stt.get("accuracy", {})

    status = summary.get("status", "?")
    out += _kv("Status",  f"{_status_emoji(status)} {status}")
    out += _kv("Model",   summary.get("model",   "?"))
    out += _kv("Language",summary.get("language","?"))
    out += _kv("Avg latency", summary.get("avg_latency_s", "?"), "s")
    out += _kv("Model available", _bool_emoji(model.get("model_available", False)))

    out += _section("Latency by Audio Length", 3)
    rows = []
    for length, v in by_len.items():
        rows.append([
            length,
            f"{v.get('wav_size_kb', '?')} KB",
            f"{v.get('min_s',  '?')}s",
            f"{v.get('mean_s', '?')}s",
            f"{v.get('max_s',  '?')}s",
            f"{v.get('std_s',  '?')}s",
            f"{v.get('realtime_factor', '?')}x",
        ])
    out += _table(["Length", "WAV size", "Min", "Mean", "Max", "Std", "RTF"], rows)

    out += _section("Accuracy (Kokoro → Groq round trip)", 3)
    overall = acc.get("overall_accuracy_pct", "?")
    out += _kv("Overall accuracy", overall, "%")
    for phrase in acc.get("phrases", []):
        out += f"\n- **Expected:** `{phrase.get('expected', '?')}`  \n"
        out += f"  **Got:** `{phrase.get('got', '?')}`  \n"
        out += f"  Accuracy: {phrase.get('accuracy_pct', '?')}%  |  "
        out += f"Exact match: {_bool_emoji(phrase.get('exact_match', False))}  |  "
        out += f"Latency: {phrase.get('latency_s', '?')}s\n"

    return out


def _render_llm(llm: dict) -> str:
    if not llm:
        return _section("LLM (Groq)") + "_No data collected._\n"

    out      = _section("LLM (Groq)")
    summary  = llm.get("summary",  {})
    by_comp  = llm.get("by_complexity", {})
    history  = llm.get("history_overhead", {})
    compliance = llm.get("compliance", {})

    status = summary.get("status", "?")
    out += _kv("Status",        f"{_status_emoji(status)} {status}")
    out += _kv("Model",         summary.get("model",               "?"))
    out += _kv("Short TTFT",    summary.get("short_prompt_ttft_s", "?"), "s")
    out += _kv("Max tokens",    summary.get("max_tokens",          "?"))
    out += _kv("Temperature",   summary.get("temperature",         "?"))

    out += _section("Latency by Prompt Complexity", 3)
    rows = []
    for label, v in by_comp.items():
        if "error" in v:
            rows.append([label, "ERROR", "-", "-", "-", "-", "-"])
            continue
        rows.append([
            label,
            v.get("prompt_words", "?"),
            f"{v.get('ttft', {}).get('min_s',  '?')}s",
            f"{v.get('ttft', {}).get('mean_s', '?')}s",
            f"{v.get('ttft', {}).get('max_s',  '?')}s",
            f"{v.get('throughput', {}).get('mean_tokens_per_sec', '?')} tok/s",
            v.get("avg_tokens_generated", "?"),
        ])
    out += _table(["Complexity", "Words", "TTFT min", "TTFT mean", "TTFT max", "Throughput", "Avg tokens"], rows)

    out += _section("Conversation History Overhead", 3)
    rows = []
    for key in ["0_turns", "5_turns", "10_turns"]:
        v = history.get(key, {})
        if v:
            rows.append([
                v.get("history_turns", "?"),
                v.get("approx_history_tokens", "?"),
                f"{v.get('mean_ttft_s', '?')}s",
                f"{v.get('min_ttft_s',  '?')}s",
                f"{v.get('max_ttft_s',  '?')}s",
            ])
    if rows:
        out += _table(["Turns", "~Tokens", "Mean TTFT", "Min TTFT", "Max TTFT"], rows)
    overhead = history.get("overhead_per_turn_s")
    if overhead is not None:
        out += f"\n> **History overhead:** {overhead*1000:.2f}ms per additional turn\n"

    out += _section("System Prompt Compliance", 3)
    score = compliance.get("compliance_score", "?")
    out += _kv("Compliance score", score, "%")
    out += _kv("Word count",       compliance.get("word_count", "?"))
    out += _kv("Sentence count",   compliance.get("sentence_count", "?"))
    for check, ok in compliance.get("checks", {}).items():
        out += f"- {_bool_emoji(ok)} {check.replace('_', ' ')}\n"
    response = compliance.get("response", "")
    if response:
        out += f"\n**Sample response:**\n> {response[:300]}{'...' if len(response) > 300 else ''}\n"

    return out


def _render_tts(tts: dict) -> str:
    if not tts:
        return _section("TTS (Kokoro)") + "_No data collected._\n"

    out      = _section("TTS (Kokoro)")
    summary  = tts.get("summary",    {})
    cold     = tts.get("cold_start", {})
    by_len   = tts.get("by_length",  {})
    word_split = tts.get("word_split", {})
    gpu      = tts.get("gpu",        {})
    model    = tts.get("model_info", {})

    status = summary.get("status", "?")
    out += _kv("Status",  f"{_status_emoji(status)} {status}")
    out += _kv("Voice",   summary.get("voice", "?"))
    out += _kv("Speed",   summary.get("speed", "?"))
    out += _kv("Medium phrase latency", summary.get("medium_phrase_latency_s", "?"), "s")
    out += _kv("Medium phrase RTF",     summary.get("medium_phrase_rtf", "?"))

    out += _section("Model Files", 3)
    onnx = model.get("onnx_model", {})
    out += _kv("ONNX model",         onnx.get("path", "?"))
    out += _kv("ONNX size",          onnx.get("size_mb", "?"), "MB")
    out += _kv("ONNXRuntime version", model.get("onnxruntime_version", "?"))
    out += _kv("Providers",          ", ".join(model.get("onnxruntime_providers", [])))

    out += _section("Cold Start vs Warmed Up", 3)
    out += _kv("Cold start latency", cold.get("cold_start_s", "?"), "s")
    out += _kv("Warmed up latency",  cold.get("warmed_up_s",  "?"), "s")
    out += _kv("Overhead",           cold.get("overhead_s",   "?"), "s")
    out += _kv("Cold RTF",           cold.get("cold_rtf",     "?"))
    out += _kv("Warm RTF",           cold.get("warm_rtf",     "?"))

    out += _section("Synthesis Latency by Phrase Length", 3)
    rows = []
    for label, v in by_len.items():
        rows.append([
            label,
            v.get("char_count",    "?"),
            f"{v.get('latency', {}).get('min_s',  '?')}s",
            f"{v.get('latency', {}).get('mean_s', '?')}s",
            f"{v.get('latency', {}).get('max_s',  '?')}s",
            f"{v.get('audio_duration_s', '?')}s",
            f"{v.get('realtime_factor',  '?')}x",
            f"{v.get('chars_per_sec',    '?')} c/s",
        ])
    out += _table(["Length", "Chars", "Min", "Mean", "Max", "Audio dur", "RTF", "Chars/s"], rows)

    out += _section("Word Boundary Split Test", 3)
    out += _kv("Original chars",   word_split.get("original_chars",   "?"))
    out += _kv("Split at char",    word_split.get("split_at_char",    "?"))
    out += _kv("Part 1 chars",     word_split.get("part1_chars",      "?"))
    out += _kv("Part 2 chars",     word_split.get("part2_chars",      "?"))
    out += _kv("Clean split",      _bool_emoji(word_split.get("clean_split", False)))
    out += _kv("Part 1 latency",   word_split.get("part1_latency_s",  "?"), "s")
    out += _kv("Part 2 latency",   word_split.get("part2_latency_s",  "?"), "s")

    out += _section("GPU During Synthesis", 3)
    if gpu.get("available"):
        out += _kv("VRAM before", gpu.get("before_used_mb", "?"), "MB")
        out += _kv("VRAM after",  gpu.get("after_used_mb",  "?"), "MB")
        out += _kv("VRAM delta",  gpu.get("delta_mb",       "?"), "MB")
    else:
        out += "_GPU not available._\n"

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(results: dict, output_dir: str = OUTPUT_DIR) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename  = f"diagnostic_report_{timestamp}.md"
    filepath  = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    doc  = _render_header(results, timestamp)
    doc += _render_hardware(results.get("hardware", {}))
    doc += _render_audio(results.get("audio",    {}))
    doc += _render_network(results.get("network", {}))
    doc += _render_stt(results.get("stt",      {}))
    doc += _render_llm(results.get("llm",      {}))
    doc += _render_tts(results.get("tts",      {}))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"[markdown] saved → {filepath}")
    return filepath


if __name__ == "__main__":
    # Smoke test
    path = generate({})
    print(f"Smoke test passed → {path}")