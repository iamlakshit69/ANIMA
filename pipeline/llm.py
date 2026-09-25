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

        # Do NOT drain text_queue. A recognized utterance in text_queue may be
        # a new user utterance captured during barge-in.
        print(f"[llm] processing user text: {user_text}")

        # Append user entry before API call
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
            if conversation_history and conversation_history[-1].get("role") == "user":
                conversation_history.pop()
            await token_queue.put(END_OF_RESPONSE)
            continue

        response_tokens = []
        first_token = True
        interrupted = False

        try:
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if not token:
                    continue

                if first_token:
                    ev.llm_first_token_at = time.monotonic()
                    if ev.stt_done_at > 0:
                        print(f"[llm] first token in {ev.llm_first_token_at - ev.stt_done_at:.2f}s")
                    first_token = False

                response_tokens.append(token)

                if not interrupt_event.is_set():
                    await token_queue.put(token)
                else:
                    interrupted = True
                    print("[llm] generation interrupted — aborting token stream")
                    break
        finally:
            # Explicitly close the stream response to terminate server-side Ollama generation
            await stream.close()

        if interrupted:
            # Interrupted: do NOT commit partial assistant text to conversation history.
            # Pop the pending user message so conversation history retains only completed turns.
            if conversation_history and conversation_history[-1].get("role") == "user":
                conversation_history.pop()
            print("[llm] discarded interrupted turn from history")
            await token_queue.put(END_OF_RESPONSE)
            continue

        full_response = "".join(response_tokens)

        # Commit completed turn and trim
        conversation_history.append({
            "role": "assistant",
            "content": full_response,
        })
        conversation_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]

        await token_queue.put(END_OF_RESPONSE)