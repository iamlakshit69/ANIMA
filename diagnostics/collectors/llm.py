"""
diagnostics/collectors/llm.py
==============================
Benchmarks Groq LLM performance:
  - Time to first token at multiple prompt complexities (short/medium/long)
  - Token throughput (tokens/sec)
  - Full response latency
  - Multiple runs per complexity for min/mean/max stats
  - Conversation history overhead (latency with 0, 5, 10 turns of history)
  - Model availability check
  - Response quality spot check (does it follow the system prompt?)
"""

import asyncio
import os
import sys
import time

import numpy as np
from groq import AsyncGroq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
    SYSTEM_PROMPT,
)

# ── Config ────────────────────────────────────────────────────────────────────
RUNS_PER_COMPLEXITY = 3

TEST_PROMPTS = {
    "short":  "Hi.",
    "medium": "What is the capital of France and what is it known for?",
    "long": (
        "Can you explain the main differences between machine learning "
        "and deep learning, and give me a practical example of each?"
    ),
}

# Synthetic conversation history for history overhead test
HISTORY_TURN = [
    {"role": "user",      "content": "What time is it?"},
    {"role": "assistant", "content": "I don't have access to real-time data like the current time."},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _stream_completion(
    client: AsyncGroq,
    messages: list[dict],
) -> dict:
    """
    Run a single streaming completion and return timing + token stats.
    """
    t_start       = time.monotonic()
    t_first_token = None
    tokens        = []

    try:
        stream = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE,
            stream=True,
        )

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue
            if t_first_token is None:
                t_first_token = time.monotonic()
            tokens.append(token)

        t_end         = time.monotonic()
        ttft          = (t_first_token - t_start) if t_first_token else None
        total_latency = t_end - t_start
        n_tokens      = len(tokens)
        throughput    = (
            n_tokens / (t_end - t_first_token)
            if t_first_token and n_tokens > 1 else 0.0
        )

        return {
            "success":               True,
            "time_to_first_token_s": round(ttft, 3) if ttft else None,
            "total_latency_s":       round(total_latency, 3),
            "tokens_generated":      n_tokens,
            "tokens_per_sec":        round(throughput, 1),
            "response":              "".join(tokens),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Sub-collectors ────────────────────────────────────────────────────────────

async def _benchmark_by_complexity(client: AsyncGroq) -> dict:
    """Benchmark TTFT and throughput at short / medium / long prompt complexity."""
    results = {}

    for label, prompt in TEST_PROMPTS.items():
        print(f"[llm] benchmarking '{label}' prompt x{RUNS_PER_COMPLEXITY} runs...")
        runs = []

        for i in range(RUNS_PER_COMPLEXITY):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]
            result = await _stream_completion(client, messages)
            if result["success"]:
                runs.append(result)
                print(
                    f"[llm]   run {i+1}: "
                    f"TTFT={result['time_to_first_token_s']:.3f}s  "
                    f"total={result['total_latency_s']:.3f}s  "
                    f"tokens={result['tokens_generated']}  "
                    f"tok/s={result['tokens_per_sec']:.1f}"
                )
            else:
                print(f"[llm]   run {i+1}: FAILED — {result.get('error')}")
            await asyncio.sleep(0.2)

        if not runs:
            results[label] = {"error": "all runs failed"}
            continue

        ttfts       = np.array([r["time_to_first_token_s"] for r in runs if r["time_to_first_token_s"]])
        totals      = np.array([r["total_latency_s"]       for r in runs])
        throughputs = np.array([r["tokens_per_sec"]        for r in runs])
        token_counts = np.array([r["tokens_generated"]     for r in runs])

        results[label] = {
            "prompt":       prompt,
            "prompt_words": len(prompt.split()),
            "runs":         len(runs),
            "ttft": {
                "min_s":  round(float(ttfts.min()),  3),
                "mean_s": round(float(ttfts.mean()), 3),
                "max_s":  round(float(ttfts.max()),  3),
                "std_s":  round(float(ttfts.std()),  3),
            },
            "total_latency": {
                "min_s":  round(float(totals.min()),  3),
                "mean_s": round(float(totals.mean()), 3),
                "max_s":  round(float(totals.max()),  3),
            },
            "throughput": {
                "mean_tokens_per_sec": round(float(throughputs.mean()), 1),
                "max_tokens_per_sec":  round(float(throughputs.max()),  1),
            },
            "avg_tokens_generated": round(float(token_counts.mean()), 1),
            "sample_response":      runs[0]["response"],
        }

    return results


async def _benchmark_history_overhead(client: AsyncGroq) -> dict:
    """
    Measure how much conversation history adds to TTFT.
    Tests with 0, 5, and 10 turns of synthetic history.
    """
    print("[llm] benchmarking conversation history overhead...")
    results = {}

    for n_turns in [0, 5, 10]:
        history = HISTORY_TURN * n_turns
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": TEST_PROMPTS["short"]},
        ]
        approx_tokens = sum(len(m["content"].split()) for m in history)

        runs = []
        for _ in range(RUNS_PER_COMPLEXITY):
            result = await _stream_completion(client, messages)
            if result["success"] and result["time_to_first_token_s"]:
                runs.append(result["time_to_first_token_s"])
            await asyncio.sleep(0.2)

        if runs:
            arr = np.array(runs)
            results[f"{n_turns}_turns"] = {
                "history_turns":         n_turns,
                "approx_history_tokens": approx_tokens,
                "mean_ttft_s":           round(float(arr.mean()), 3),
                "min_ttft_s":            round(float(arr.min()),  3),
                "max_ttft_s":            round(float(arr.max()),  3),
            }
            print(f"[llm]   {n_turns} turns (~{approx_tokens} tokens): mean TTFT={arr.mean():.3f}s")

    # Cost per additional turn
    if "0_turns" in results and "10_turns" in results:
        overhead = results["10_turns"]["mean_ttft_s"] - results["0_turns"]["mean_ttft_s"]
        results["overhead_per_turn_s"] = round(overhead / 10, 4)

    return results


async def _check_system_prompt_compliance(client: AsyncGroq) -> dict:
    """Spot check: does the model actually follow the system prompt constraints?"""
    print("[llm] checking system prompt compliance...")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "Explain quantum computing in detail with examples."},
    ]

    result = await _stream_completion(client, messages)
    if not result["success"]:
        return {"error": result.get("error")}

    response  = result["response"]
    sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    has_bullets  = any(
        line.strip().startswith(("-", "*", "•", "1.", "2."))
        for line in response.splitlines()
    )
    has_markdown  = any(c in response for c in ["**", "__", "##", "```"])
    sentence_count = len(sentences)
    word_count     = len(response.split())

    checks = {
        "no_bullet_points": not has_bullets,
        "no_markdown":      not has_markdown,
        "within_2_sentences": sentence_count <= 3,
        "concise_under_80_words": word_count < 80,
    }
    compliance_score = round(sum(checks.values()) / len(checks) * 100, 1)

    return {
        "response":        response,
        "word_count":      word_count,
        "sentence_count":  sentence_count,
        "checks":          checks,
        "compliance_score": compliance_score,
    }


async def _check_model_info(client: AsyncGroq) -> dict:
    """Confirm configured LLM model is available on Groq."""
    try:
        t0      = time.monotonic()
        models  = await client.models.list()
        latency = time.monotonic() - t0
        available = [m.id for m in models.data]
        return {
            "configured_model": GROQ_MODEL,
            "model_available":  GROQ_MODEL in available,
            "api_latency_ms":   round(latency * 1000, 2),
            "total_models":     len(available),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Main entry point ──────────────────────────────────────────────────────────

async def collect_async() -> dict:
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set — skipping LLM benchmark"}

    client = AsyncGroq(api_key=GROQ_API_KEY)

    print("[llm] checking model availability...")
    model_info = await _check_model_info(client)

    print("[llm] benchmarking by prompt complexity...")
    complexity = await _benchmark_by_complexity(client)

    print("[llm] benchmarking history overhead...")
    history = await _benchmark_history_overhead(client)

    print("[llm] checking system prompt compliance...")
    compliance = await _check_system_prompt_compliance(client)

    short_ttft = complexity.get("short", {}).get("ttft", {}).get("mean_s")

    return {
        "model_info":       model_info,
        "by_complexity":    complexity,
        "history_overhead": history,
        "compliance":       compliance,
        "summary": {
            "model":               GROQ_MODEL,
            "short_prompt_ttft_s": short_ttft,
            "max_tokens":          GROQ_MAX_TOKENS,
            "temperature":         GROQ_TEMPERATURE,
            "status": (
                "good"    if short_ttft and short_ttft < 0.3 else
                "warning" if short_ttft and short_ttft < 0.6 else
                "bad"
            ),
        },
    }


def collect() -> dict:
    return asyncio.run(collect_async())


if __name__ == "__main__":
    import json
    results = collect()
    # Truncate sample responses for cleaner terminal output
    for v in results.get("by_complexity", {}).values():
        if isinstance(v, dict) and "sample_response" in v:
            v["sample_response"] = v["sample_response"][:100] + "..."
    if "response" in results.get("compliance", {}):
        results["compliance"]["response"] = results["compliance"]["response"][:150] + "..."
    print(json.dumps(results, indent=2))