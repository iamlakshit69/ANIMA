"""
diagnostics/report/plot.py
===========================
Generates a multi-panel diagnostic plot from all collector results.

Panels:
  1. Echo decay curve (mic RMS over time with threshold lines)
  2. Per-stage latency bar chart (STT / LLM / TTS mean latency)
  3. STT latency by audio length (grouped bars, 1s/3s/5s)
  4. LLM TTFT by prompt complexity (short/medium/long)
  5. TTS synthesis latency by phrase length + real-time factor line
  6. Network ping distribution (min/mean/max/jitter)
  7. Hardware overview panel (GPU VRAM, RAM, disk speed as bar gauges)
  8. VAD false positive rate + noise floor summary

Output: diagnostics/results/diagnostic_plot.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

# ── Theme ─────────────────────────────────────────────────────────────────────
BG        = "#0d1117"
PANEL_BG  = "#161b22"
GRID      = "#21262d"
TEXT      = "#e6edf3"
SUBTEXT   = "#8b949e"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
YELLOW    = "#d29922"
RED       = "#f85149"
ORANGE    = "#db6d28"
PURPLE    = "#bc8cff"
TEAL      = "#39d353"

STAGE_COLORS = {
    "STT": ACCENT,
    "LLM": PURPLE,
    "TTS": TEAL,
}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "diagnostic_plot.png")


def _status_color(status: str) -> str:
    return {"good": GREEN, "warning": YELLOW, "bad": RED}.get(status, SUBTEXT)


def _apply_panel_style(ax, title: str = ""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    ax.spines[:].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.8)
    ax.xaxis.label.set_color(SUBTEXT)
    ax.yaxis.label.set_color(SUBTEXT)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold",
                     pad=6, loc="left")


def _bar(ax, x, height, color, width=0.5, label=None):
    bars = ax.bar(x, height, width=width, color=color, alpha=0.85,
                  label=label, zorder=3)
    return bars


# ── Panel renderers ───────────────────────────────────────────────────────────

def _panel_echo_decay(ax, audio_data: dict):
    _apply_panel_style(ax, "Echo Decay")

    echo = audio_data.get("echo_decay", {})
    timeline = echo.get("rms_timeline", [])
    playback_end = echo.get("playback_end_t", 0)

    if not timeline:
        ax.text(0.5, 0.5, "No echo data", ha="center", va="center",
                color=SUBTEXT, transform=ax.transAxes)
        return

    times = [t for t, _ in timeline]
    rms   = [r for _, r in timeline]

    ax.plot(times, rms, color=ACCENT, linewidth=0.9, zorder=4, label="Mic RMS")
    ax.axvline(playback_end, color=RED,    linewidth=1.5, linestyle="--",
               label="Playback ended", zorder=5)
    ax.axhline(0.02, color=GREEN,  linewidth=1.0, linestyle=":",
               label="MIN_AUDIO_ENERGY (0.02)", zorder=5)
    ax.axhline(0.01, color=YELLOW, linewidth=0.8, linestyle=":",
               label="SILENCE_RMS (0.01)", zorder=5)

    decay_t = echo.get("echo_decay_time_s")
    if decay_t is not None:
        ax.axvline(playback_end + decay_t, color=GREEN, linewidth=1.2,
                   linestyle="-.", label=f"Decayed at +{decay_t:.2f}s", zorder=5)

    ax.axvspan(playback_end, max(times), alpha=0.07, color=RED, zorder=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6.5, loc="upper right",
              facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)


def _panel_pipeline_latency(ax, audio_data: dict, stt_data: dict,
                             llm_data: dict, tts_data: dict):
    _apply_panel_style(ax, "Pipeline Stage Latency (mean)")

    stages  = []
    values  = []
    colors  = []
    statuses = []

    stt_lat = stt_data.get("summary", {}).get("avg_latency_s")
    llm_lat = llm_data.get("summary", {}).get("short_prompt_ttft_s")
    tts_lat = tts_data.get("summary", {}).get("medium_phrase_latency_s")

    for label, val, status_key in [
        ("STT", stt_lat, "stt"),
        ("LLM\n(TTFT)", llm_lat, "llm"),
        ("TTS", tts_lat, "tts"),
    ]:
        if val is not None:
            stages.append(label)
            values.append(val)
            colors.append(STAGE_COLORS.get(label.split("\n")[0], ACCENT))

    if not stages:
        ax.text(0.5, 0.5, "No latency data", ha="center", va="center",
                color=SUBTEXT, transform=ax.transAxes)
        return

    xs   = np.arange(len(stages))
    bars = ax.bar(xs, values, color=colors, width=0.5, alpha=0.85, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}s", ha="center", va="bottom",
                color=TEXT, fontsize=8, fontweight="bold")

    total = sum(values)
    ax.axhline(total, color=ORANGE, linewidth=1.2, linestyle="--",
               label=f"Total ~{total:.2f}s", zorder=5)

    ax.set_xticks(xs)
    ax.set_xticklabels(stages, color=TEXT, fontsize=9)
    ax.set_ylabel("Seconds")
    ax.set_ylim(bottom=0, top=max(values) * 1.4)
    ax.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)


def _panel_stt_latency(ax, stt_data: dict):
    _apply_panel_style(ax, "STT Latency by Audio Length")

    by_length = stt_data.get("latency_by_length", {})
    if not by_length:
        ax.text(0.5, 0.5, "No STT data", ha="center", va="center",
                color=SUBTEXT, transform=ax.transAxes)
        return

    labels = list(by_length.keys())
    means  = [v["mean_s"] for v in by_length.values()]
    mins   = [v["min_s"]  for v in by_length.values()]
    maxs   = [v["max_s"]  for v in by_length.values()]
    rtfs   = [v["realtime_factor"] for v in by_length.values()]

    xs = np.arange(len(labels))
    ax.bar(xs, means, color=ACCENT, width=0.5, alpha=0.85, zorder=3, label="Mean")
    ax.errorbar(xs, means,
                yerr=[np.array(means) - np.array(mins),
                      np.array(maxs) - np.array(means)],
                fmt="none", color=TEXT, capsize=4, linewidth=1.2, zorder=4)

    # RTF on secondary axis
    ax2 = ax.twinx()
    ax2.plot(xs, rtfs, color=ORANGE, marker="o", linewidth=1.5,
             markersize=5, label="Real-time factor", zorder=5)
    ax2.set_ylabel("RTF", color=ORANGE, fontsize=8)
    ax2.tick_params(colors=ORANGE, labelsize=7)
    ax2.set_facecolor(PANEL_BG)
    ax2.spines[:].set_color(GRID)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=TEXT)
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, loc="upper left")
    ax2.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, loc="upper right")


def _panel_llm_latency(ax, llm_data: dict):
    _apply_panel_style(ax, "LLM Time-to-First-Token by Complexity")

    by_complexity = llm_data.get("by_complexity", {})
    if not by_complexity:
        ax.text(0.5, 0.5, "No LLM data", ha="center", va="center",
                color=SUBTEXT, transform=ax.transAxes)
        return

    labels     = list(by_complexity.keys())
    ttft_means = [v.get("ttft", {}).get("mean_s", 0) for v in by_complexity.values()]
    ttft_mins  = [v.get("ttft", {}).get("min_s",  0) for v in by_complexity.values()]
    ttft_maxs  = [v.get("ttft", {}).get("max_s",  0) for v in by_complexity.values()]
    throughputs = [v.get("throughput", {}).get("mean_tokens_per_sec", 0)
                   for v in by_complexity.values()]

    xs = np.arange(len(labels))
    ax.bar(xs, ttft_means, color=PURPLE, width=0.5, alpha=0.85, zorder=3)
    ax.errorbar(xs, ttft_means,
                yerr=[np.array(ttft_means) - np.array(ttft_mins),
                      np.array(ttft_maxs) - np.array(ttft_means)],
                fmt="none", color=TEXT, capsize=4, linewidth=1.2, zorder=4)

    # History overhead annotation
    history = llm_data.get("history_overhead", {})
    overhead = history.get("overhead_per_turn_s")
    if overhead is not None:
        ax.text(0.98, 0.95,
                f"History overhead:\n{overhead*1000:.1f}ms / turn",
                ha="right", va="top", transform=ax.transAxes,
                color=YELLOW, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                          edgecolor=YELLOW, alpha=0.8))

    # Throughput on secondary axis
    ax2 = ax.twinx()
    ax2.plot(xs, throughputs, color=TEAL, marker="s", linewidth=1.5,
             markersize=5, label="tok/s", zorder=5)
    ax2.set_ylabel("tokens/sec", color=TEAL, fontsize=8)
    ax2.tick_params(colors=TEAL, labelsize=7)
    ax2.set_facecolor(PANEL_BG)
    ax2.spines[:].set_color(GRID)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=TEXT)
    ax.set_ylabel("TTFT (seconds)")
    ax2.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)


def _panel_tts_latency(ax, tts_data: dict):
    _apply_panel_style(ax, "TTS Synthesis Latency by Phrase Length")

    by_length = tts_data.get("by_length", {})
    if not by_length:
        ax.text(0.5, 0.5, "No TTS data", ha="center", va="center",
                color=SUBTEXT, transform=ax.transAxes)
        return

    labels  = list(by_length.keys())
    means   = [v.get("latency", {}).get("mean_s", 0) for v in by_length.values()]
    mins    = [v.get("latency", {}).get("min_s",  0) for v in by_length.values()]
    maxs    = [v.get("latency", {}).get("max_s",  0) for v in by_length.values()]
    rtfs    = [v.get("realtime_factor", 0) for v in by_length.values()]
    chars   = [v.get("char_count", 0) for v in by_length.values()]

    xs = np.arange(len(labels))
    bars = ax.bar(xs, means, color=TEAL, width=0.5, alpha=0.85, zorder=3)
    ax.errorbar(xs, means,
                yerr=[np.array(means) - np.array(mins),
                      np.array(maxs) - np.array(means)],
                fmt="none", color=TEXT, capsize=4, linewidth=1.2, zorder=4)

    # Char count labels on bars
    for bar, c in zip(bars, chars):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{c}c", ha="center", va="center",
                color=BG, fontsize=7, fontweight="bold")

    # Cold start overhead annotation
    cold = tts_data.get("cold_start", {})
    overhead = cold.get("overhead_s")
    if overhead is not None:
        ax.text(0.98, 0.95,
                f"Cold start overhead:\n+{overhead:.3f}s",
                ha="right", va="top", transform=ax.transAxes,
                color=ORANGE, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                          edgecolor=ORANGE, alpha=0.8))

    # RTF on secondary axis
    ax2 = ax.twinx()
    ax2.plot(xs, rtfs, color=ORANGE, marker="^", linewidth=1.5,
             markersize=5, label="RTF", zorder=5)
    ax2.axhline(1.0, color=RED, linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_ylabel("Real-time factor", color=ORANGE, fontsize=8)
    ax2.tick_params(colors=ORANGE, labelsize=7)
    ax2.set_facecolor(PANEL_BG)
    ax2.spines[:].set_color(GRID)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=TEXT)
    ax.set_ylabel("Seconds")
    ax2.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT)


def _panel_network(ax, network_data: dict):
    _apply_panel_style(ax, "Network — Groq API Latency")

    ping = network_data.get("ping", {})
    if not ping.get("success"):
        ax.text(0.5, 0.5, "No network data", ha="center", va="center",
                color=SUBTEXT, transform=ax.transAxes)
        return

    labels = ["Min", "Mean", "Max", "Jitter\n(std)"]
    values = [
        ping.get("min_ms",    0),
        ping.get("mean_ms",   0),
        ping.get("max_ms",    0),
        ping.get("jitter_ms", 0),
    ]
    colors = [GREEN, ACCENT, RED, YELLOW]

    xs   = np.arange(len(labels))
    bars = ax.bar(xs, values, color=colors, width=0.5, alpha=0.85, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}ms", ha="center", va="bottom",
                color=TEXT, fontsize=8)

    # Upload/download bandwidth
    upload   = network_data.get("upload_bandwidth", {})
    download = network_data.get("download_bandwidth", {})
    info_lines = []
    if upload.get("success"):
        info_lines.append(f"↑ {upload['speed_mbps']:.1f} Mbps")
        info_lines.append(f"  1s WAV: ~{upload.get('est_1s_wav_ms', 0):.0f}ms upload")
        info_lines.append(f"  5s WAV: ~{upload.get('est_5s_wav_ms', 0):.0f}ms upload")
    if download.get("success"):
        info_lines.append(f"↓ {download['speed_mbps']:.1f} Mbps")

    if info_lines:
        ax.text(0.98, 0.97, "\n".join(info_lines),
                ha="right", va="top", transform=ax.transAxes,
                color=SUBTEXT, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                          edgecolor=GRID, alpha=0.8))

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=TEXT)
    ax.set_ylabel("ms")
    ax.set_ylim(bottom=0)


def _panel_hardware(ax, hw_data: dict):
    """Hardware gauges: GPU VRAM, RAM, disk read speed."""
    _apply_panel_style(ax, "Hardware Overview")
    ax.axis("off")

    gpu  = hw_data.get("gpu", {})
    ram  = hw_data.get("ram", {})
    disk = hw_data.get("disk", {})
    cpu  = hw_data.get("cpu", {})
    vers = hw_data.get("versions", {})

    lines = []

    # CPU
    lines.append(("CPU", f"{cpu.get('physical_cores', '?')}c / "
                          f"{cpu.get('logical_cores', '?')}t  "
                          f"{cpu.get('freq_mhz_max', '?')} MHz  "
                          f"usage {cpu.get('usage_percent', '?')}%",
                  ACCENT))

    # RAM
    ram_pct = ram.get("usage_percent", 0)
    lines.append(("RAM", f"{ram.get('used_gb', '?'):.1f} / "
                          f"{ram.get('total_gb', '?'):.1f} GB  "
                          f"({ram_pct}% used)",
                  GREEN if ram_pct < 70 else YELLOW if ram_pct < 90 else RED))

    # GPU
    if gpu.get("available") and gpu.get("gpus"):
        g = gpu["gpus"][0]
        vram_pct = g.get("vram_usage_pct", 0)
        lines.append(("GPU", f"{g.get('name', '?')}  "
                              f"{g.get('vram_used_gb', '?'):.1f}/"
                              f"{g.get('vram_total_gb', '?'):.1f} GB VRAM  "
                              f"({vram_pct}%)",
                      GREEN if vram_pct < 70 else YELLOW if vram_pct < 90 else RED))
        lines.append(("CUDA", f"{gpu.get('cuda_version', '?')}  "
                               f"driver {gpu.get('driver_version', '?')}",
                      SUBTEXT))
    else:
        lines.append(("GPU", "Not available / CPU mode", RED))

    # Disk
    lines.append(("Disk R/W", f"{disk.get('read_speed_mbs', '?'):.0f} / "
                               f"{disk.get('write_speed_mbs', '?'):.0f} MB/s  "
                               f"({disk.get('partition_free_gb', '?'):.0f} GB free)",
                  ACCENT))

    # Key versions
    lines.append(("torch",      vers.get("torch", "?"),      SUBTEXT))
    lines.append(("onnxruntime", vers.get("onnxruntime", "?"), SUBTEXT))
    lines.append(("groq",       vers.get("groq", "?"),        SUBTEXT))
    lines.append(("python",     vers.get("python", "?"),       SUBTEXT))
    lines.append(("OS",         vers.get("os", "?"),           SUBTEXT))

    y = 0.97
    for label, value, color in lines:
        ax.text(0.0, y, f"{label}:", transform=ax.transAxes,
                color=SUBTEXT, fontsize=8, va="top", fontweight="bold")
        ax.text(0.22, y, value, transform=ax.transAxes,
                color=color, fontsize=8, va="top")
        y -= 0.10


def _panel_vad_noise(ax, audio_data: dict):
    _apply_panel_style(ax, "VAD & Noise Floor")

    noise  = audio_data.get("noise_floor", {})
    vad_fp = audio_data.get("vad_false_positives", {})
    vad_lat = audio_data.get("vad_detection_latency", {})

    ax.axis("off")
    lines = []

    # Noise floor
    lines.append(("Noise floor mean RMS",
                  f"{noise.get('mean_rms', '?'):.5f}",
                  ACCENT))
    lines.append(("Noise floor max RMS",
                  f"{noise.get('max_rms', '?'):.5f}",
                  ACCENT))
    lines.append(("Recommended MIN_AUDIO_ENERGY",
                  f"{noise.get('recommended_min_audio_energy', '?'):.4f}",
                  GREEN))

    # VAD false positives
    fp_pct   = vad_fp.get("false_positive_pct", "?")
    fp_status = vad_fp.get("status", "")
    fp_color  = {"good": GREEN, "warning": YELLOW, "bad": RED}.get(fp_status, SUBTEXT)
    lines.append(("VAD false positive rate",
                  f"{fp_pct}%  [{fp_status}]",
                  fp_color))
    lines.append(("VAD mean prob on silence",
                  f"{vad_fp.get('mean_vad_prob', '?'):.4f}",
                  SUBTEXT))
    lines.append(("VAD max prob on silence",
                  f"{vad_fp.get('max_vad_prob', '?'):.4f}",
                  SUBTEXT))

    # VAD detection latency
    det_ms = vad_lat.get("detection_latency_ms")
    lines.append(("VAD detection latency",
                  f"{det_ms:.1f}ms" if det_ms else "not detected",
                  GREEN if det_ms and det_ms < 100 else YELLOW))

    # Echo summary
    echo = audio_data.get("echo_decay", {})
    decay_t = echo.get("echo_decay_time_s")
    lines.append(("Echo decay time",
                  f"{decay_t:.2f}s" if decay_t else "did not decay",
                  GREEN if decay_t and decay_t < 1.0 else YELLOW if decay_t else RED))
    lines.append(("Recommended BUFFER_MUTE_GUARD",
                  f"{echo.get('recommended_buffer_mute_guard', '?')}s",
                  GREEN))
    lines.append(("Recommended POST_SPEECH_MUTE",
                  f"{echo.get('recommended_post_speech_mute', '?')}s",
                  GREEN))

    y = 0.97
    for label, value, color in lines:
        ax.text(0.0, y, f"{label}:", transform=ax.transAxes,
                color=SUBTEXT, fontsize=8, va="top")
        ax.text(0.58, y, value, transform=ax.transAxes,
                color=color, fontsize=8, va="top", fontweight="bold")
        y -= 0.115


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(results: dict, output_path: str = OUTPUT_PATH):
    """
    Generate the full diagnostic plot.
    results: the combined dict from run.py containing all collector outputs.
    """
    hw_data      = results.get("hardware", {})
    audio_data   = results.get("audio",    {})
    network_data = results.get("network",  {})
    stt_data     = results.get("stt",      {})
    llm_data     = results.get("llm",      {})
    tts_data     = results.get("tts",      {})

    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        "Voice Assistant — Full Diagnostic Report",
        color=TEXT, fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        left=0.05, right=0.97,
        top=0.94,  bottom=0.05,
    )

    # Row 0
    ax_echo     = fig.add_subplot(gs[0, 0])
    ax_pipeline = fig.add_subplot(gs[0, 1])
    ax_hw       = fig.add_subplot(gs[0, 2])

    # Row 1
    ax_stt      = fig.add_subplot(gs[1, 0])
    ax_llm      = fig.add_subplot(gs[1, 1])
    ax_tts      = fig.add_subplot(gs[1, 2])

    # Row 2
    ax_network  = fig.add_subplot(gs[2, 0])
    ax_vad      = fig.add_subplot(gs[2, 1])
    ax_status   = fig.add_subplot(gs[2, 2])

    _panel_echo_decay(ax_echo, audio_data)
    _panel_pipeline_latency(ax_pipeline, audio_data, stt_data, llm_data, tts_data)
    _panel_hardware(ax_hw, hw_data)
    _panel_stt_latency(ax_stt, stt_data)
    _panel_llm_latency(ax_llm, llm_data)
    _panel_tts_latency(ax_tts, tts_data)
    _panel_network(ax_network, network_data)
    _panel_vad_noise(ax_vad, audio_data)
    _panel_status_summary(ax_status, results)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[plot] saved → {output_path}")
    return output_path


def _panel_status_summary(ax, results: dict):
    """Traffic-light summary of each subsystem."""
    _apply_panel_style(ax, "Health Summary")
    ax.axis("off")

    checks = [
        ("Hardware",  results.get("hardware", {}).get("gpu", {}).get("available", False),
         "GPU available", "No GPU — CPU mode"),
        ("Network",   results.get("network",  {}).get("groq_reachable", False),
         "Groq reachable", "Groq unreachable"),
        ("API Key",   results.get("network",  {}).get("api_key", {}).get("valid", False),
         "API key valid", "API key invalid"),
        ("STT",       results.get("stt", {}).get("summary", {}).get("status") == "good",
         "STT latency good", "STT latency high"),
        ("LLM",       results.get("llm", {}).get("summary", {}).get("status") == "good",
         "LLM TTFT good", "LLM TTFT high"),
        ("TTS",       results.get("tts", {}).get("summary", {}).get("status") == "good",
         "TTS latency good", "TTS latency high"),
        ("Echo",      results.get("audio", {}).get("echo_decay", {}).get("echo_fully_decayed", False),
         "Echo decays cleanly", "Echo persists — check mic gain"),
        ("VAD",       results.get("audio", {}).get("vad_false_positives", {}).get("status") == "good",
         "VAD clean on silence", "VAD false positives detected"),
        ("Compliance", results.get("llm", {}).get("compliance", {}).get("compliance_score", 0) >= 75,
         "System prompt followed", "System prompt compliance low"),
    ]

    y = 0.95
    for label, ok, ok_msg, fail_msg in checks:
        dot_color = GREEN if ok else RED
        msg       = ok_msg if ok else fail_msg
        ax.text(0.02, y, "●", transform=ax.transAxes,
                color=dot_color, fontsize=12, va="top")
        ax.text(0.12, y, f"{label}:", transform=ax.transAxes,
                color=TEXT, fontsize=8, va="top", fontweight="bold")
        ax.text(0.38, y, msg, transform=ax.transAxes,
                color=dot_color, fontsize=8, va="top")
        y -= 0.105


if __name__ == "__main__":
    # Smoke test with empty data
    generate({})
    print("Smoke test passed — check diagnostics/results/diagnostic_plot.png")