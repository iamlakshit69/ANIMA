import asyncio
import logging
import numpy as np

logger = logging.getLogger("anima.retry")

# Fallback apology phrase sent when LLM or STT fails
FALLBACK_APOLOGY = "I'm sorry, I'm having trouble processing that right now. Please try again."


async def retry_with_backoff(
    coro_func,
    *args,
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    stage_name: str = "api",
    **kwargs,
):
    """Execute an async function with bounded exponential backoff for transient errors."""
    delay = initial_delay
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            # Check for non-retryable errors (invalid API key, terms required, 400/404)
            err_msg = str(e).lower()
            status_code = getattr(e, "status_code", None)
            is_non_retryable = (
                status_code in (400, 401, 403, 404, 422)
                or "400" in err_msg
                or "404" in err_msg
                or "terms" in err_msg
                or "model_not_found" in err_msg
                or "invalid_request_error" in err_msg
                or "invalid api key" in err_msg
                or "authentication" in err_msg
                or "unauthorized" in err_msg
            )
            if is_non_retryable:
                logger.error("[%s] non-retryable client error: %s", stage_name, e)
                raise

            if attempt == max_retries:
                logger.error(
                    "[%s] all %d attempts failed. Final error: %s",
                    stage_name, max_retries, e
                )
                raise

            logger.warning(
                "[%s] attempt %d/%d failed (%s). Retrying in %.2fs...",
                stage_name, attempt, max_retries, e, delay
            )
            await asyncio.sleep(delay)
            delay *= backoff_factor

    raise last_exception


def generate_error_tone(sample_rate: int = 16000, duration: float = 0.25) -> np.ndarray:
    """Generate a clean 2-tone soft alert chime (440Hz -> 330Hz) for audible error cues."""
    t1 = np.linspace(0, duration / 2, int(sample_rate * duration / 2), False)
    t2 = np.linspace(0, duration / 2, int(sample_rate * duration / 2), False)
    tone1 = 0.2 * np.sin(2 * np.pi * 440 * t1)
    tone2 = 0.2 * np.sin(2 * np.pi * 330 * t2)
    # Apply fade in and out to prevent click
    full = np.concatenate([tone1, tone2]).astype(np.float32)
    envelope = np.ones_like(full)
    fade_len = int(sample_rate * 0.02)
    if len(full) > 2 * fade_len:
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    return full * envelope
