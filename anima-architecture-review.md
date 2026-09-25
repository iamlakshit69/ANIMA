# ANIMA Voice Assistant — Architecture Review & Fix Specification

**Scope reviewed:** `main.py`, `core/{events,queues,sentinel}.py`, `config/settings.py`,
`pipeline/{mic,stt,llm,tts,speaker}.py`, `requirements.txt`, `README.md` (~634 LOC).
**Not reviewed** (not present in the uploaded archive, but present on disk per `ls -lah`):
`diagnostics/`, `tests/`, `.git/`, `.pytest_cache/`, `kokoro-v0_19.onnx`, `voices.bin`,
`piper_models/`. See "Open Questions" at the end — these need to be checked before Phase 3.

This document is written to be handed directly to a coding agent. Each finding has: severity,
exact location, why it happens, why it produces the symptom you're seeing, and a concrete fix.
Section 5 is the actionable task list, ordered by priority.

---

## 1. Root Cause #1 (Critical): No real echo cancellation — timers pretending to be AEC

### What's happening
The mic stays open and actively listening **while the assistant is speaking** (this is required
for barge-in / interruption). But nothing cancels the assistant's own voice out of the mic
signal. Instead, five independent hand-tuned heuristics try to *guess* when audio is "real user
speech" vs. "the speaker echoing into the mic":

| Constant | File | Value | Guessing what |
|---|---|---|---|
| `ENERGY_THRESHOLD` | `pipeline/mic.py` | 0.01 | is this chunk loud enough to be speech at all |
| `ECHO_DECAY_PAD` | `pipeline/mic.py` | 0.5s | how long echo lingers in the room after playback "ends" |
| `POST_SPEECH_MUTE` | `pipeline/mic.py` | 1.5s | how long to ignore mic after assistant stops talking |
| `BARGE_IN_FRAMES` | `pipeline/mic.py` | 8 frames (~240ms) | how much continuous energy = a real interruption, not echo |
| `BUFFER_MUTE_GUARD` | `pipeline/stt.py` | 0.5s | discard buffers collected too soon after assistant finished |
| `MIN_AUDIO_ENERGY` | `pipeline/stt.py` | 0.02 | reject whole buffers that are "too quiet to be real" |

The comments in your own codebase document this failing in production:

> `pipeline/stt.py`: *"Previously 2.0s — raised to 4.0s because long responses need 4s+ to decay;
> a 2s guard meant the second half of the echo passed through unblocked, causing hallucinated
> transcripts ('Thanks for watching!', 'you', etc.)"*

That is Whisper transcribing your own TTS output played back through the speaker. No amount of
timer-tuning fixes this correctly — it only shifts *which* rooms/speakers/mic-gain-settings it
breaks on. A quieter speaker, a louder room, a more sensitive mic, a longer sentence — any of
these will desync the timers and either (a) cause self-interruption (assistant hears its own
echo, treats it as a barge-in) or (b) cause hallucinated transcripts to reach the LLM, or (c)
cause the mic to stay muted so long that real user speech right after the assistant finishes gets
dropped.

### Why this is the most likely cause of "it's not working"
Self-triggered barge-in and hallucinated transcripts are the two most disruptive, hard-to-debug
symptoms a user of a voice assistant will hit, and this architecture guarantees both will happen
under some conditions.

### Compounding problem: two different audio backends
`pipeline/mic.py` uses **PyAudio** for capture. `pipeline/speaker.py` uses **sounddevice** for
playback. These are two separate libraries with independent internal buffering and clocks. Real
AEC requires tightly time-aligned access to *both* the exact signal sent to the speaker and the
signal captured by the mic, on a shared clock. With two unrelated audio stacks, you cannot get
that alignment — so even a future "let's add real AEC" effort is blocked until I/O is unified.

### Fix
1. **Unify audio I/O onto one backend.** Prefer `sounddevice` for both (it's already a
   dependency, has a numpy-native interface, and supports a duplex `Stream` that captures input
   and drives output on the same callback/clock). Drop `pyaudio` from `requirements.txt`.
2. **Add real acoustic echo cancellation.** Options, cheapest to most robust:
   - `speexdsp-python` (bindings to libspeexdsp's AEC) — lightweight, works via a "far-end"
     reference frame you feed it (the exact audio you send to the speaker) plus the mic frame;
     it returns the cleaned signal.
   - `webrtc-audio-processing` (Python bindings to WebRTC's APM, which includes AEC3) — heavier
     but higher quality, same package family as `webrtcvad` (already in your requirements, but
     currently unused — see Finding 2.2).
   - If a coding agent's time is constrained: as an interim measure, implement **half-duplex
     barge-in** — mute the mic entirely while `assistant_speaking` is set, and only allow
     barge-in via a fixed hardware/software cue (e.g. holding a hotkey, or a short "wake" word),
     rather than trying to detect barge-in acoustically at all. This is a real, shippable
     fallback if AEC integration is too large a lift right now — say so explicitly to the user
     rather than leaving the current guesswork in place.
3. **Once real AEC exists, delete the timer heuristics**: `ECHO_DECAY_PAD`, `POST_SPEECH_MUTE`,
   `BUFFER_MUTE_GUARD`, and the "flush stale echo chunks from audio_queue" logic in
   `speaker.py` all exist purely to compensate for the lack of AEC. If AEC does its job, none of
   them should be needed. Keep `MIN_AUDIO_ENERGY`-style rejection only as a cheap sanity check on
   truly silent buffers, not as a primary echo defense.

---

## 2. Root Cause #2 (Critical): No turn/generation identity — a single shared boolean can't safely represent "interrupted"

### What's happening
`core/events.py` defines one process-wide `interrupt_event = asyncio.Event()` that all five
pipeline stages (`mic`, `stt`, `llm`, `tts`, `speaker`) read and write, with no concept of *which
conversation turn* a given piece of data belongs to. Every stage has its own ad-hoc logic for
deciding whether to trust or discard the data in front of it based on this one flag:

- `stt.py` discards `audio_buffer` if `interrupt_event.is_set()`, and *again* checks it after the
  Groq network round-trip returns (because the round trip takes 1–2s, plenty of time for state
  to change underneath it).
- `llm.py` drains `text_queue` and then **busy-waits** (`while interrupt_event.is_set(): await
  asyncio.sleep(0.05)`) for the flag to clear before resuming — a polling loop standing in for
  what should be an explicit "wait for this specific turn to be cancelled" signal.
- `tts.py`'s `_accumulator` and `_synthesizer` both independently check the flag at different
  points in the token/phrase stream.
- `speaker.py` clears `interrupt_event` unconditionally whenever it processes an `END_OF_SPEECH`
  marker.

### The concrete race condition this causes
1. User is mid-barge-in on turn N. `interrupt_event.set()`.
2. `llm.py` breaks out of its streaming loop for turn N but **still unconditionally sends
   `END_OF_RESPONSE`** down `token_queue` (this is correct per-se — it has to signal the pipe is
   drained — but it carries no marker saying "this belongs to the cancelled turn").
3. That stale `END_OF_RESPONSE` flows through `tts.py`'s accumulator/synthesizer, which forward
   it as `END_OF_SPEECH` regardless of interrupt state.
4. Meanwhile, `stt.py` has already picked up the *new* utterance (turn N+1), and depending on
   timing, `llm.py` may have already started processing turn N+1 and legitimately re-set
   `interrupt_event` for a **newer** reason.
5. `speaker.py` receives the **stale** `END_OF_SPEECH` from turn N and calls
   `interrupt_event.clear()` — silently cancelling the in-flight state for turn N+1 that had
   nothing to do with the message that just arrived.

This is not a hypothetical: it's a direct, traceable consequence of correlating pipeline
messages by a single shared boolean instead of an identity. It's also almost certainly the
reason the code has accumulated the string of dated, numbered patches visible in the comments
(`Bug #3 fix`, `Bug #4 fix`, `Bug #10 fix`) — those are the observable symptoms of this design
being patched reactively rather than redesigned.

### Fix
Introduce a monotonically increasing **generation / turn ID**, owned by a single place (e.g. a
small `TurnController` class), and thread it through every message that flows through the
pipeline:

```python
# core/turn.py (new file)
import asyncio

class TurnController:
    def __init__(self):
        self._gen = 0
        self._lock = asyncio.Lock()

    async def bump(self) -> int:
        """Call this exactly once, at the moment a barge-in or new user utterance
        invalidates the previous turn. Returns the new generation id."""
        async with self._lock:
            self._gen += 1
            return self._gen

    @property
    def current(self) -> int:
        return self._gen

turn_controller = TurnController()
```

Every queue payload becomes `(generation_id, payload)` instead of a bare payload. Each stage's
rule becomes uniform and simple: *"if `generation_id != turn_controller.current`, drop it — no
special-casing needed per stage."* This replaces:
- the busy-wait loop in `llm.py`,
- the "discard stale transcript" check in `stt.py`,
- the ad-hoc `first_phrase = True` resets in `tts.py`,
- the unconditional `interrupt_event.clear()` in `speaker.py` (it should instead clear only if
  the `END_OF_SPEECH` it just processed matches the generation that is *currently* considered
  interrupted — or, better, generation bumps alone are sufficient and a separate "cleared" flag
  becomes unnecessary).

This is a moderate refactor (touches all 5 pipeline files and the queue payload shape) but it
removes an entire class of race condition rather than adding another special case to catch the
next one.

---

## 3. Global mutable state with no ownership (`core/events.py`)

### What's happening
`core/events.py` exposes plain module-level floats (`speaking_started_at`,
`current_phrase_duration`, `speaking_ended_at`, `user_stopped_speaking_at`, `stt_done_at`,
`llm_first_token_at`, `tts_first_phrase_done_at`) that are read and written directly via
`import core.events as ev; ev.some_field = ...` from four different files. There is no
encapsulation, no validation, and no single owner for any of these fields.

This isn't a thread-safety problem (asyncio is single-threaded/cooperative), but it **is** a
correctness and maintainability problem:
- **TOCTOU races across `await` points.** A coroutine can read `ev.speaking_ended_at`, hit an
  `await`, and by the time it resumes, another task has already mutated that value based on
  events that happened during the await. The code partially compensates for this with guard
  comments (*"Guard stt_done_at > 0 — it's 0.0 on the very first turn"*), which is itself a sign
  the invariants aren't enforced anywhere, just remembered by convention.
- **No way to unit test in isolation.** Every stage's behavior depends on hidden global state
  mutated by other stages, so you cannot test `stt.py`'s guard logic without also simulating
  `speaker.py`'s side effects on the same module.
- **Latency instrumentation is scattered across 4 files with manual math done in a 5th
  (`speaker.py` computes `stt_time`, `llm_time`, `tts_time` by subtracting timestamps written
  elsewhere)** — fragile, and breaks silently if a stage is skipped or reordered.

### Fix
Replace the flat module with an explicit, owned state object, and prefer passing per-turn data
through the queue (as part of the `(generation_id, payload)` tuple from Section 2) rather than a
shared global wherever possible. For state that genuinely is cross-cutting (like
`assistant_speaking`), keep it, but wrap it:

```python
# core/state.py (replaces core/events.py)
import asyncio
import time
from dataclasses import dataclass, field

@dataclass
class PlaybackState:
    speaking: asyncio.Event = field(default_factory=asyncio.Event)
    started_at: float = 0.0
    phrase_duration: float = 0.0
    ended_at: float = field(default_factory=time.monotonic)

    def mark_started(self, duration: float) -> None:
        self.started_at = time.monotonic()
        self.phrase_duration = duration
        self.speaking.set()

    def mark_ended(self) -> None:
        self.speaking.clear()
        self.ended_at = time.monotonic()

@dataclass
class TurnLatency:
    """One instance per turn — created fresh, not a shared global. Pass this
    alongside the generation id through the queues instead of writing to
    module globals."""
    user_stopped_speaking_at: float = 0.0
    stt_done_at: float = 0.0
    llm_first_token_at: float = 0.0
    tts_first_phrase_done_at: float = 0.0
```

`playback_state` becomes the single owned instance for speaking/echo timing (still needed for
Finding 1's interim heuristics if AEC isn't done yet — but now at least it's got a name and a
clear owner). `TurnLatency` becomes a value created per-turn and threaded through the pipeline,
not a shared mutable global — eliminating the "guard stt_done_at > 0" class of workaround
entirely, since a fresh turn always starts with a fresh, zeroed `TurnLatency`.

---

## 4. Additional findings

### 4.1 (High) No crash supervision — one bad exception kills the whole assistant
`main.py` runs all five stages with a bare `asyncio.gather(...)`. If any single stage raises an
exception that isn't caught internally (e.g. a PyAudio device disappearing, an unexpected
attribute on a Groq response, a `KeyError`), `gather()` propagates it and **cancels all four other
running tasks**, killing the whole assistant. There's no restart policy, no isolation between
stages, and no distinction between "this stage needs to die" and "the whole process should die."

**Fix:** Wrap each stage in a supervising loop that catches, logs, and restarts that specific
task with backoff, while leaving the others running:

```python
async def _supervise(name, coro_fn, *args):
    backoff = 1.0
    while True:
        try:
            await coro_fn(*args)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("[%s] crashed — restarting in %.1fs", name, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        else:
            # a stage's while-True loop returning normally is itself a bug — log it
            logging.error("[%s] exited its main loop without an exception", name)
            return
```

Then `main()` runs `asyncio.gather(*(_supervise(name, fn) for name, fn in stages))`. This alone
would have prevented a whole class of "it just stopped working and I had to restart it" reports.

### 4.2 (High) Groq API failures silently drop the user's turn with no feedback
In `stt.py`, `llm.py`, and `tts.py`, any exception from the Groq client is caught, logged to
stdout, and the code just `continue`s — the user gets silence and has no idea their turn was
dropped (rate limit, network blip, invalid key mid-session, etc.). There's no retry with backoff,
no circuit breaker, and no user-facing fallback (e.g. a canned "sorry, I didn't catch that" TTS
line, or at minimum a distinct audio cue).

**Fix:** Add bounded retry with exponential backoff for transient errors (timeouts, 5xx, rate
limits) in all three network-calling stages, and on final failure, push a short pre-recorded or
synthesized apology phrase into `tts_queue` / `token_queue` so the user gets *some* signal rather
than dead air.

### 4.3 (Medium) VAD is a stale, misleading mismatch between comments and code
`requirements.txt` lists `webrtcvad`, and `pipeline/stt.py` has a comment referencing **"Silero
VAD"** firing `SILENCE_MARKER` — but the actual VAD implementation in `pipeline/mic.py` is a bare
RMS-energy threshold (`ENERGY_THRESHOLD = 0.01`), and neither `webrtcvad` nor Silero is imported
or used anywhere in the reviewed code. This strongly suggests a prior implementation used a real
VAD library and was reverted or half-migrated, leaving stale comments and a dead dependency
behind. This kind of drift is exactly what confuses a coding agent working on this repo later —
it will read the comment, assume Silero VAD is in use, and reason incorrectly about the system.

**Fix:** Pick one, deliberately:
- If you want cheap and already-installed: wire up `webrtcvad` (frame-based, fast, no model
  download) in `mic.py` and delete the RMS threshold and the stale comment in `stt.py`.
- If you want much better accuracy in noisy rooms: use `silero-vad` (small ONNX/torch model,
  designed exactly for this).
Either way, fix the comment in `stt.py` to describe what's actually running, and remove
`webrtcvad` from `requirements.txt` if you end up not using it.

### 4.4 (Medium) Two audio backends is also just unnecessary complexity, independent of AEC
Even setting aside AEC, using `pyaudio` for input and `sounddevice` for output means two
different device-enumeration and error-handling code paths, two different failure modes to
handle, and no shared device-selection logic (see 4.6). Consolidating onto one library (see
Finding 1, fix #1) simplifies this regardless of whether AEC ships in the same change.

### 4.5 (Medium) Config values scattered across three files instead of one source of truth
`config/settings.py` holds most tunables (`SAMPLE_RATE`, `SILENCE_DURATION`, `MIN_PHRASE_CHARS`,
etc.), but `ENERGY_THRESHOLD`, `ECHO_DECAY_PAD`, `BARGE_IN_FRAMES`, `POST_SPEECH_MUTE` are
hardcoded in `pipeline/mic.py`, and `SILENCE_RMS`, `MIN_AUDIO_ENERGY`, `BUFFER_MUTE_GUARD` are
hardcoded in `pipeline/stt.py` — despite these all being tightly coupled, interdependent
audio-detection constants (the comment in `stt.py` even says *"Must be >= POST_SPEECH_MUTE in
mic.py to be effective"*, i.e. a cross-file invariant enforced only by a code comment, not by
code). Tuning one without the other, exactly what the comment warns against, is trivially easy to
do by accident.

**Fix:** Move every one of these constants into `config/settings.py`, and where one value must be
`>=` another (like `BUFFER_MUTE_GUARD` vs `POST_SPEECH_MUTE`), enforce it in code, not a comment:

```python
assert BUFFER_MUTE_GUARD >= POST_SPEECH_MUTE, (
    "BUFFER_MUTE_GUARD must be >= POST_SPEECH_MUTE or echo will leak through"
)
```

(This whole category of constant should shrink a lot once real AEC lands per Finding 1 — but
until then, consolidate them.)

### 4.6 (Medium) No explicit audio device selection
`pyaudio.PyAudio()` / the sounddevice calls open whatever the OS considers the default input and
output device, with no configuration option and no error handling if no input device exists, or
if the "default" device is the wrong one (very common on machines with a Bluetooth headset, a
webcam mic, and a laptop mic all present simultaneously — an extremely common real-world cause of
"it doesn't seem to be listening / nothing plays"). Add a `MIC_DEVICE` / `SPEAKER_DEVICE` setting
in `config/settings.py` (nullable, defaults to system default), and on startup, log the actual
device name/index selected so a user can diagnose "it's listening to the wrong mic" instantly
instead of guessing.

### 4.7 (Low) `GROQ_MAX_TOKENS = 100` can truncate mid-sentence
The system prompt asks the model to be concise, but `max_tokens=100` is a hard cutoff — if the
model doesn't wrap up in time, the response is truncated mid-word/mid-sentence and that partial
text is sent to TTS and spoken aloud as-is. Either raise the cap and rely on prompting alone, or
detect truncation (`finish_reason == "length"`) and trim to the last complete sentence before
sending to TTS.

### 4.8 (Low) Logging is `print()`-only, no levels, no persistence
Every stage uses bare `print()` with hand-rolled tags (`[mic]`, `[stt]`, etc.). This is fine for
local dev, but there's no way to turn debug output off/on, no log file for post-mortem debugging
when a user reports "it broke" after you're no longer watching the terminal, and no structured
fields (turn id, timestamps) that could be queried later. Swap for the standard `logging` module
with a rotating file handler once the turn-id work in Finding 2 lands (so log lines can carry a
`generation_id` for correlation).

### 4.9 (Low) Repo carries ~316MB of apparently-dead model weights
Per your `ls -lah`, the working tree has `kokoro-v0_19.onnx` (310MB), `voices.bin` (5.5MB), and a
`piper_models/` directory — none of which are referenced anywhere in the reviewed pipeline code,
and the README explicitly says this is the **"cloud"** branch (Groq APIs only, no local
inference). `onnxruntime` isn't even in `requirements.txt`, confirming nothing in this branch
loads that ONNX file. This is almost certainly leftover from a prior local-inference
implementation (Kokoro for TTS, Piper for something else — STT or TTS) that was abandoned in
favor of the Groq cloud APIs, without being cleaned up.

**Why this matters architecturally, not just as tidiness:** carrying two competing, half-present
architectures in one working tree (a "local" one implied by the leftover model files, and the
actual "cloud" one in the code) is exactly the kind of thing that will confuse a coding agent
working on this repo — it may find `piper_models/` and reasonably assume it's live and load-bearing.

**Fix:** Confirm nothing imports these (a `grep -r piper_models` / `grep -r kokoro` across the
full repo, including `diagnostics/` and `tests/` which weren't in the archive I reviewed — see
Open Questions). If genuinely unused on this branch, delete them from the working tree and,
critically, purge them from git history (`git filter-repo` or BFG) if they were ever committed —
310MB blobs in git history make every clone slow permanently, not just today's checkout.

---

## 5. Task list for the coding agent (priority order)

**Phase 0 — before touching code**
- [ ] Grep the full repo (including `diagnostics/`, `tests/`, `.git` history) for
  `piper_models`, `kokoro`, `voices.bin`, `webrtcvad` to confirm Finding 4.9 and 4.3 before
  deleting anything.
- [ ] Confirm what `diagnostics/` and `tests/` currently do — they weren't included in the
  archive reviewed here and may already codify some of these constants/behaviors.

**Phase 1 — stop the bleeding (crash/data-loss fixes, no architecture change)**
- [ ] Add the `_supervise()` wrapper from Finding 4.1 around all five `asyncio.gather()` tasks in
  `main.py`.
- [ ] Add bounded retry-with-backoff to the Groq calls in `stt.py`, `llm.py`, `tts.py`
  (Finding 4.2), plus a user-facing fallback phrase on final failure.
- [ ] Add the cross-file invariant assertion from Finding 4.5 (`BUFFER_MUTE_GUARD >=
  POST_SPEECH_MUTE`) so a future config change fails loudly instead of silently leaking echo.
- [ ] Fix or remove the stale "Silero VAD" comment in `stt.py` (Finding 4.3) — accuracy of
  comments matters a lot once an agent is working from this file.

**Phase 2 — the real architecture fix**
- [ ] Consolidate audio I/O onto a single backend (`sounddevice` for both capture and playback),
  removing `pyaudio` (Finding 1, fix #1 / Finding 4.4).
- [ ] Integrate real AEC (`speexdsp-python` or `webrtc-audio-processing`) using the now-unified
  audio stream, feeding it the exact frames sent to the speaker as the far-end reference
  (Finding 1, fix #2). If this is too large a single change, ship the half-duplex fallback
  first (mute mic during playback, no acoustic barge-in) as an interim, explicitly-labeled
  mode.
- [ ] Once AEC (or the half-duplex fallback) is in and validated, delete `ECHO_DECAY_PAD`,
  `POST_SPEECH_MUTE`, `BUFFER_MUTE_GUARD`, and the echo-flush logic in `speaker.py`
  (Finding 1, fix #3).
- [ ] Introduce `TurnController` / generation IDs and thread `(generation_id, payload)` through
  every queue (`audio_queue`, `text_queue`, `token_queue`, `tts_queue`, and the internal
  `phrase_queue` in `tts.py`). Update every stage's staleness check to compare generation ids
  instead of checking the shared `interrupt_event` ad hoc (Finding 2). This touches all of
  `mic.py`, `stt.py`, `llm.py`, `tts.py`, `speaker.py`.
- [ ] Replace `core/events.py`'s flat globals with the `PlaybackState` / `TurnLatency` structure
  from Finding 3. Move per-turn latency fields off the shared module and onto a value created
  fresh per turn, passed alongside the generation id.

**Phase 3 — cleanup and hygiene**
- [ ] Move `ENERGY_THRESHOLD`, `ECHO_DECAY_PAD` (if still needed post-AEC), `BARGE_IN_FRAMES`,
  `POST_SPEECH_MUTE`, `SILENCE_RMS`, `MIN_AUDIO_ENERGY`, `BUFFER_MUTE_GUARD` all into
  `config/settings.py` (Finding 4.5).
- [ ] Add `MIC_DEVICE`/`SPEAKER_DEVICE` settings and startup device-name logging (Finding 4.6).
- [ ] Decide the VAD story (webrtcvad vs. silero-vad vs. keep energy-based-but-documented-as-such)
  and make code match comments (Finding 4.3).
- [ ] Handle `finish_reason == "length"` truncation in `llm.py` by trimming to the last complete
  sentence before it reaches TTS (Finding 4.7).
- [ ] Delete the dead model weights / `piper_models/` if Phase 0's grep confirms they're unused,
  and purge them from git history (Finding 4.9).
- [ ] Replace `print()` calls with `logging`, tagged with `generation_id` once available
  (Finding 4.8).

**Phase 4 — testing**
- [ ] Because Phase 2 removes the global-state coupling, add unit tests for the staleness logic
  in each stage using fake queues and a fake `TurnController` — this was effectively untestable
  before the refactor due to hidden global state (Finding 3).
- [ ] Add an integration test that simulates a barge-in mid-response and asserts the *new* turn's
  `interrupt_event`/generation state is never clobbered by the *old* turn's trailing messages —
  this directly tests the race condition described in Finding 2.

