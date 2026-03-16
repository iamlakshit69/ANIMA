import asyncio
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
from core.sentinel import END_OF_RESPONSE

MAX_HISTORY_TURNS = 10  # keep last 5 exchanges (10 messages) to cap input tokens


async def llm_stream():
    client = AsyncGroq(api_key=GROQ_API_KEY)
    conversation_history = []

    print("[llm] ready...")

    while True:
        user_text = await text_queue.get()

        if interrupt_event.is_set():
            # Drain any stale messages that piled up during interruption.
            # Without this, old half-heard utterances get sent to LLM
            # after the user has already moved on.
            while not text_queue.empty():
                try:
                    text_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            continue

        print(f"[llm] sending: {user_text}")

        conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        # Cap history to avoid growing input token count every turn.
        # After 10+ turns the extra tokens add 50-200ms of latency per request.
        conversation_history = conversation_history[-MAX_HISTORY_TURNS:]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + conversation_history

        stream = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE,
            stream=True,
        )

        # Use a list and join at the end — O(n) vs O(n²) for string +=
        response_tokens = []

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue

            # Always accumulate regardless of interrupt so conversation
            # history stays accurate even if the user barged in mid-response.
            response_tokens.append(token)

            # Only push to TTS pipeline if not interrupted
            if not interrupt_event.is_set():
                await token_queue.put(token)

        full_response = "".join(response_tokens)

        conversation_history.append({
            "role": "assistant",
            "content": full_response,
        })

        await token_queue.put(END_OF_RESPONSE)