import asyncio
import time
from groq import AsyncGroq

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
    SYSTEM_PROMPT,
)
from core.queues import text_queue, token_queue
from core.events import interrupt_event
import core.events as ev
from core.sentinel import END_OF_RESPONSE

# 10 complete back-and-forth turns kept in context.
# One turn = 1 user message + 1 assistant message = 2 entries in the list.
# Slicing [-MAX_HISTORY_TURNS:] would keep only 10 messages (5 turns) — wrong.
# Correct slice is [-(MAX_HISTORY_TURNS * 2):] for 10 full turns.
MAX_HISTORY_TURNS = 10


async def llm_stream():
    client = AsyncGroq(api_key=GROQ_API_KEY)
    conversation_history = []

    print(f"[llm] ready... (groq: {GROQ_MODEL})")

    while True:
        user_text = await text_queue.get()

        # Bug #3 fix: do NOT discard user_text and continue — that silently
        # drops the genuine new utterance that arrives while interrupt_event is
        # still set. interrupt_event stays set until speaker.py processes
        # END_OF_SPEECH, which can be several seconds after the barge-in fires.
        #
        # Correct logic:
        #   - Any item dequeued while interrupt is set is stale (from the
        #     interrupted turn). Discard it and drain the rest of the queue.
        #   - Wait for the interrupt to fully clear, then loop back to
        #     text_queue.get() for the genuine new utterance. That utterance
        #     won't arrive until POST_SPEECH_MUTE + Groq Whisper latency after
        #     the interrupt fired, so the wait below is typically instantaneous.
        if interrupt_event.is_set():
            while not text_queue.empty():
                try:
                    text_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            while interrupt_event.is_set():
                await asyncio.sleep(0.05)
            print("[llm] interrupt cleared — waiting for new utterance")
            continue  # back to text_queue.get() for the real new utterance

        print(f"[llm] sending: {user_text}")

        # Append user entry *before* the API call so history is consistent
        # going into the request. If the call fails, pop it back off (see
        # except block) to avoid consecutive same-role entries that break
        # future requests.
        conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + conversation_history

        try:
            stream = await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=GROQ_MAX_TOKENS,
                temperature=GROQ_TEMPERATURE,
                stream=True,
            )
        except Exception as e:
            print(f"[llm] groq error: {e}")
            # Bug #4 fix: pop the user entry on API failure so history never
            # has two consecutive 'user' role entries. Previously the assistant
            # entry was simply never added on error, permanently corrupting
            # history and causing cascading API failures on every subsequent turn.
            conversation_history.pop()
            await token_queue.put(END_OF_RESPONSE)
            continue

        response_tokens = []
        first_token = True

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue

            # Stamp when first token arrives for per-step latency breakdown.
            # Guard stt_done_at > 0 — it's 0.0 on the very first turn before
            # any STT has completed.
            if first_token:
                ev.llm_first_token_at = time.monotonic()
                if ev.stt_done_at > 0:
                    print(f"[llm] first token in {ev.llm_first_token_at - ev.stt_done_at:.2f}s")
                first_token = False

            response_tokens.append(token)

            if not interrupt_event.is_set():
                await token_queue.put(token)
            else:
                # Break immediately on interrupt — don't drain the rest of the
                # Groq stream. Consuming remaining tokens after the user has
                # already started speaking again wastes time and adds latency.
                break

        full_response = "".join(response_tokens)

        # Append assistant entry first, then trim — so the trim window always
        # contains complete turns and never cuts a turn in half.
        conversation_history.append({
            "role": "assistant",
            "content": full_response,
        })

        # Bug #10 fix: one turn = 2 messages (user + assistant).
        # [-MAX_HISTORY_TURNS:] kept only 5 turns; [-(MAX_HISTORY_TURNS * 2):]
        # correctly keeps 10 full turns.
        conversation_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]

        await token_queue.put(END_OF_RESPONSE)