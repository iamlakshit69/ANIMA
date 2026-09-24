import asyncio
import time
from openai import AsyncOpenAI

from config.settings import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
)
from core.queues import text_queue, token_queue
from core.events import interrupt_event
import core.events as ev
from core.sentinel import END_OF_RESPONSE

# 10 complete back-and-forth turns kept in context.
# One turn = 1 user message + 1 assistant message = 2 entries in the list.
# The old code sliced [-10:] which kept 10 *messages* (only 5 turns).
MAX_HISTORY_TURNS = 10
OLLAMA_MODEL = "phi3:3.8b"


async def llm_stream():
    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    conversation_history = []

    print(f"[llm] warming up {OLLAMA_MODEL}...")
    try:
        await client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
    except Exception as e:
        print(f"[llm] warmup note: {e}")

    print(f"[llm] ready... (local: {OLLAMA_MODEL})")

    while True:
        user_text = await text_queue.get()

        # Bug #3 fix: do NOT discard user_text and continue — that silently
        # dropped valid new utterances that happened to arrive while
        # interrupt_event was still set (it stays set until speaker.py
        # processes END_OF_SPEECH, which can be several seconds after the
        # barge-in fires).
        #
        # Correct logic:
        #   - Any item dequeued while interrupt is set is stale (it came from
        #     the interrupted turn). Discard it and everything else in the queue.
        #   - Wait for the interrupt to fully clear, then loop back to
        #     text_queue.get() for the genuine new utterance. That utterance
        #     won't arrive until POST_SPEECH_MUTE (2 s) + Whisper latency after
        #     the interrupt fired, so interrupt_event is always already cleared
        #     by then — the wait below will typically be instantaneous.
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

        # Append user entry *before* the API call so the history is consistent
        # going into the request — but if the call fails we pop it back off
        # (see except block below) to avoid consecutive same-role entries that
        # break future requests.
        conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + conversation_history

        try:
            stream = await client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                stream=True,
            )
        except Exception as e:
            print(f"[llm] ollama error: {e}")
            # Bug #4 fix: remove the user entry we just appended so history
            # never has two consecutive 'user' role entries.  Previously the
            # assistant entry was simply never added on error, permanently
            # corrupting the history and causing cascading API failures.
            conversation_history.pop()
            await token_queue.put(END_OF_RESPONSE)
            continue

        response_tokens = []
        first_token = True

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue

            # Stamp when first token arrives — guard against zero on first turn
            if first_token:
                ev.llm_first_token_at = time.monotonic()
                if ev.stt_done_at > 0:
                    print(f"[llm] first token in {ev.llm_first_token_at - ev.stt_done_at:.2f}s")
                first_token = False

            response_tokens.append(token)

            if not interrupt_event.is_set():
                await token_queue.put(token)
            else:
                # Break immediately on interrupt rather than draining the rest
                # of the Ollama stream — consuming remaining tokens after the
                # user has already started speaking again wastes seconds.
                break

        full_response = "".join(response_tokens)

        # Append assistant entry first, then trim — so the trim window always
        # contains complete turns and never cuts off half a turn.
        # Bug #10 fix (part 1): trim moved to here, after assistant append.
        conversation_history.append({
            "role": "assistant",
            "content": full_response,
        })

        # Bug #10 fix (part 2): one turn = 2 messages (user + assistant), so
        # the correct slice for MAX_HISTORY_TURNS complete turns is *2.
        # The old [-MAX_HISTORY_TURNS:] kept only 5 turns, not 10.
        conversation_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]

        await token_queue.put(END_OF_RESPONSE)