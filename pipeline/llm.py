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
import core.queues as queues
from core.state import TurnLatency
from core.turn import turn_controller
from core.sentinel import END_OF_RESPONSE
from core.retry import retry_with_backoff, FALLBACK_APOLOGY
from core.logger import get_logger

logger = get_logger("anima.llm")

# 10 complete back-and-forth turns kept in context.
# One turn = 1 user message + 1 assistant message = 2 entries in the list.
# Slice [-(MAX_HISTORY_TURNS * 2):] correctly preserves 10 full turns.
MAX_HISTORY_TURNS = 10


async def _create_chat_stream(client: AsyncGroq, messages: list[dict]):
    """Initiate streaming chat completion with Groq."""
    return await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=GROQ_MAX_TOKENS,
        temperature=GROQ_TEMPERATURE,
        stream=True,
    )


async def llm_stream():
    """Continuously receive transcripts tagged with generation IDs, query Groq LLM,
    and stream tokens tagged with generation ID and TurnLatency.
    """
    client = AsyncGroq(api_key=GROQ_API_KEY)
    conversation_history: list[dict] = []

    logger.info("[llm] ready (model: %s, max_tokens=%d)...", GROQ_MODEL, GROQ_MAX_TOKENS)

    while True:
        item = await queues.text_queue.get()

        # Unpack (gen_id, user_text, latency)
        if isinstance(item, tuple):
            if len(item) == 3:
                gen_id, user_text, latency = item
            else:
                gen_id, user_text = item
                latency = TurnLatency()
        else:
            gen_id = turn_controller.current
            user_text = item
            latency = TurnLatency()

        # Staleness check: drop superseded turn without any busy-waiting
        if gen_id != turn_controller.current:
            logger.info("[llm] dropping stale user text for gen %d (current=%d)", gen_id, turn_controller.current)
            continue

        logger.info("[llm] gen=%d sending: '%s'", gen_id, user_text)

        # Append user entry before API call
        conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + conversation_history

        try:
            stream = await retry_with_backoff(
                _create_chat_stream,
                client,
                messages,
                max_retries=3,
                initial_delay=0.5,
                stage_name="llm",
            )
        except Exception as e:
            logger.error("[llm] Groq LLM error for gen %d after retries: %s", gen_id, e)
            conversation_history.pop()  # prevent corrupted history with consecutive user roles
            if gen_id == turn_controller.current:
                await queues.token_queue.put((gen_id, FALLBACK_APOLOGY, latency))
                await queues.token_queue.put((gen_id, END_OF_RESPONSE, latency))
            continue

        response_tokens: list[str] = []
        first_token = True
        interrupted = False
        was_truncated = False

        async for chunk in stream:
            # Check finish_reason for length truncation (Finding 4.7)
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.finish_reason == "length":
                    was_truncated = True

                token = choice.delta.content or ""
            else:
                token = ""

            if not token:
                continue

            if first_token:
                latency.llm_first_token_at = time.monotonic()
                if latency.stt_done_at > 0:
                    ttft = latency.llm_first_token_at - latency.stt_done_at
                    logger.info("[llm] gen=%d first token (TTFT) in %.2fs", gen_id, ttft)
                first_token = False

            # Check staleness: if superseded mid-stream, break immediately
            if gen_id != turn_controller.current:
                logger.info("[llm] gen=%d superseded by %d during streaming — stopping token stream", gen_id, turn_controller.current)
                interrupted = True
                break

            response_tokens.append(token)
            await queues.token_queue.put((gen_id, token, latency))

        if was_truncated:
            logger.warning("[llm] gen=%d response reached max_tokens=%d cutoff", gen_id, GROQ_MAX_TOKENS)

        full_response = "".join(response_tokens)

        # Only retain non-empty response in history
        if full_response:
            conversation_history.append({
                "role": "assistant",
                "content": full_response,
            })
            conversation_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]

        # Always signal end of response for this generation
        if not interrupted or gen_id == turn_controller.current:
            await queues.token_queue.put((gen_id, END_OF_RESPONSE, latency))