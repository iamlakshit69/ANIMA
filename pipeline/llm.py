import asyncio
from groq import Groq

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


async def llm_stream():
    client = Groq(api_key=GROQ_API_KEY)
    conversation_history = []

    print("[llm] ready...")

    while True:
        user_text = await text_queue.get()

        if interrupt_event.is_set():
            continue

        print(f"[llm] sending: {user_text}")

        conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + conversation_history

        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE,
            stream=True,
        )

        full_response = ""

        for chunk in stream:
            if interrupt_event.is_set():
                break

            token = chunk.choices[0].delta.content
            if token:
                await token_queue.put(token)
                full_response += token

        conversation_history.append({
            "role": "assistant",
            "content": full_response,
        })

        await token_queue.put(END_OF_RESPONSE)