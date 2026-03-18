"""
diagnostics/run.py
===================
Main entry point for the voice assistant diagnostic suite.

Usage:
    python diagnostics/run.py              # run all collectors
    python diagnostics/run.py --skip-audio # skip mic/speaker tests
    python diagnostics/run.py --only network stt llm
    python diagnostics/run.py --compare results/a.json results/b.json

Output (all saved to diagnostics/results/):
    diagnostic_report_<timestamp>.md      — human readable report
    diagnostic_report_<timestamp>.json    — machine readable report
    diagnostic_plot.png                   — multi-panel visual report
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback

# Make sure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diagnostics.collectors import hardware, audio, network, stt, llm, tts
from diagnostics.report import plot, markdown, json_report

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

ALL_COLLECTORS = ["hardware", "audio", "network", "stt", "llm", "tts"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header():
    print("\n" + "═" * 60)
    print("  Voice Assistant — Full Diagnostic Suite")
    print("═" * 60 + "\n")


def _print_section(name: str):
    print(f"\n{'─' * 60}")
    print(f"  [{name.upper()}]")
    print(f"{'─' * 60}")


def _print_summary(results: dict):
    """Print a quick health summary to the terminal after all collectors finish."""
    print("\n" + "═" * 60)
    print("  RESULTS SUMMARY")
    print("═" * 60)

    summary = json_report._build_summary(results)
    pipeline = summary.get("pipeline", {})
    health   = summary.get("health",   {})
    net      = summary.get("network",  {})
    aud      = summary.get("audio",    {})

    # Pipeline latency
    stt_lat = pipeline.get("stt_mean_latency_s")
    llm_lat = pipeline.get("llm_short_ttft_s")
    tts_lat = pipeline.get("tts_medium_latency_s")
    total   = sum(x for x in [stt_lat, llm_lat, tts_lat] if x is not None)

    print("\n  Pipeline Latency:")
    if stt_lat: print(f"    STT  {stt_lat:.3f}s")
    if llm_lat: print(f"    LLM  {llm_lat:.3f}s  (TTFT)")
    if tts_lat: print(f"    TTS  {tts_lat:.3f}s")
    if total:   print(f"    ─────────────")
    if total:   print(f"    Est  ~{total:.3f}s  end-to-end")

    # Network
    ping = net.get("groq_ping_mean_ms")
    if ping:
        print(f"\n  Network:")
        print(f"    Groq ping  {ping:.1f}ms mean")
        ul = net.get("upload_mbps")
        dl = net.get("download_mbps")
        if ul: print(f"    Upload     {ul:.1f} Mbps")
        if dl: print(f"    Download   {dl:.1f} Mbps")

    # Audio
    echo_t = aud.get("echo_decay_time_s")
    noise  = aud.get("noise_floor_mean_rms")
    print(f"\n  Audio:")
    if noise:  print(f"    Noise floor  {noise:.5f} RMS")
    if echo_t: print(f"    Echo decay   {echo_t:.2f}s")
    else:      print(f"    Echo decay   did not decay within window")

    rec_energy = aud.get("recommended_min_audio_energy")
    rec_guard  = aud.get("recommended_buffer_mute_guard")
    rec_mute   = aud.get("recommended_post_speech_mute")
    if any([rec_energy, rec_guard, rec_mute]):
        print(f"\n  Recommended settings:")
        if rec_energy: print(f"    MIN_AUDIO_ENERGY   = {rec_energy}")
        if rec_guard:  print(f"    BUFFER_MUTE_GUARD  = {rec_guard}")
        if rec_mute:   print(f"    POST_SPEECH_MUTE   = {rec_mute}")

    # Health flags
    print(f"\n  Health:")
    checks = [
        ("GPU available",    health.get("gpu_available",    False)),
        ("Groq reachable",   health.get("groq_reachable",   False)),
        ("API key valid",    health.get("api_key_valid",    False)),
        ("STT",              health.get("stt_status") == "good"),
        ("LLM",              health.get("llm_status") == "good"),
        ("TTS",              health.get("tts_status") == "good"),
        ("Echo clean",       health.get("echo_clean",       False)),
        ("VAD clean",        health.get("vad_clean",        False)),
    ]
    for label, ok in checks:
        icon = "✓" if ok else "✗"
        print(f"    {icon}  {label}")

    print()


def _safe_collect(name: str, fn) -> dict:
    """Run a collector and catch exceptions so one failure doesn't abort the run."""
    try:
        t0 = time.monotonic()
        result = fn()
        elapsed = time.monotonic() - t0
        print(f"\n[run] ✓ {name} done in {elapsed:.1f}s")
        return result
    except Exception as e:
        print(f"\n[run] ✗ {name} FAILED: {e}")
        traceback.print_exc()
        return {"error": str(e)}


# ── Compare mode ──────────────────────────────────────────────────────────────

def _run_compare(path_a: str, path_b: str):
    print(f"\n[run] comparing:\n  A: {path_a}\n  B: {path_b}\n")
    try:
        report_a = json_report.load(path_a)
        report_b = json_report.load(path_b)
    except Exception as e:
        print(f"[run] failed to load reports: {e}")
        sys.exit(1)

    diff = json_report.compare(report_a, report_b)

    meta_a = diff.get("meta_a", {})
    meta_b = diff.get("meta_b", {})
    print(f"  A: {meta_a.get('timestamp', '?')}  git={meta_a.get('git_hash', '?')}  branch={meta_a.get('git_branch', '?')}")
    print(f"  B: {meta_b.get('timestamp', '?')}  git={meta_b.get('git_hash', '?')}  branch={meta_b.get('git_branch', '?')}")
    print()

    for section, metrics in diff.get("diff", {}).items():
        print(f"  {section.upper()}")
        for key, d in metrics.items():
            before  = d.get("before")
            after   = d.get("after")
            delta   = d.get("delta")
            improved = d.get("improved")

            if delta is None:
                continue

            arrow = ""
            if improved is True:  arrow = "↓ better"
            elif improved is False: arrow = "↑ worse"

            print(
                f"    {key:<40} "
                f"{str(before):>10} → {str(after):>10}  "
                f"Δ {delta:+.4f}  {arrow}"
            )
        print()


# ── Main run ──────────────────────────────────────────────────────────────────

def run(collectors_to_run: list[str]) -> dict:
    results = {}

    collector_map = {
        "hardware": hardware.collect,
        "audio":    audio.collect,
        "network":  network.collect,
        "stt":      stt.collect,
        "llm":      llm.collect,
        "tts":      tts.collect,
    }

    for name in collectors_to_run:
        if name not in collector_map:
            print(f"[run] unknown collector '{name}' — skipping")
            continue
        _print_section(name)
        results[name] = _safe_collect(name, collector_map[name])

    return results


def generate_reports(results: dict) -> tuple[str, str, str]:
    """Generate all three report formats. Returns (md_path, json_path, plot_path)."""
    _print_section("generating reports")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    md_path   = markdown.generate(results)
    json_path = json_report.generate(results)
    plot_path = plot.generate(results)

    return md_path, json_path, plot_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Voice assistant full diagnostic suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagnostics/run.py                          # run everything
  python diagnostics/run.py --skip audio             # skip mic/speaker tests
  python diagnostics/run.py --only network stt llm   # only these collectors
  python diagnostics/run.py --no-plot                # skip plot generation
  python diagnostics/run.py --compare a.json b.json  # diff two reports
        """,
    )
    parser.add_argument(
        "--only", nargs="+", metavar="COLLECTOR",
        choices=ALL_COLLECTORS,
        help="Only run these collectors",
    )
    parser.add_argument(
        "--skip", nargs="+", metavar="COLLECTOR",
        choices=ALL_COLLECTORS,
        help="Skip these collectors",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation (faster, no matplotlib dependency needed)",
    )
    parser.add_argument(
        "--no-reports", action="store_true",
        help="Skip all report generation — just print summary to terminal",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("REPORT_A", "REPORT_B"),
        help="Compare two JSON reports instead of running diagnostics",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # Compare mode — no diagnostics, just diff
    if args.compare:
        _run_compare(args.compare[0], args.compare[1])
        return

    _print_header()

    # Determine which collectors to run
    if args.only:
        collectors = args.only
    else:
        collectors = list(ALL_COLLECTORS)
        if args.skip:
            collectors = [c for c in collectors if c not in args.skip]

    print(f"  Collectors: {', '.join(collectors)}")
    print(f"  Results dir: {RESULTS_DIR}\n")

    t_total_start = time.monotonic()

    results = run(collectors)

    _print_summary(results)

    if not args.no_reports:
        md_path, json_path, plot_path = generate_reports(results)

        print("\n  Output files:")
        print(f"    {md_path}")
        print(f"    {json_path}")
        if not args.no_plot:
            print(f"    {plot_path}")

    total_elapsed = time.monotonic() - t_total_start
    print(f"\n[run] total time: {total_elapsed:.1f}s\n")


if __name__ == "__main__":
    main()