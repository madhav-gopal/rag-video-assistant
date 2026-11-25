"""
llm_utils.py – Shared LLM factory + rate-limit helpers
-------------------------------------------------------
Mistral free-tier limits:  ~1 req/s  |  ~500 k tokens/min
All callers import get_llm() and invoke_with_retry() from here.
"""

import os
import time
import random
import logging

import httpx
from langchain_mistralai import ChatMistralAI

logger = logging.getLogger(__name__)

# ── Tunable constants ────────────────────────────────────────────────────────
INTER_CALL_DELAY   = 1.5   # seconds to wait between sequential API calls
MAX_RETRY_ATTEMPTS = 6     # max retries on 429
RETRY_BASE_DELAY   = 5.0   # first wait after a 429 (seconds)
RETRY_MAX_DELAY    = 60.0  # cap for exponential back-off
# ─────────────────────────────────────────────────────────────────────────────


def get_llm() -> ChatMistralAI:
    """Return a ChatMistralAI instance. max_retries=0 so WE control retries."""
    model_name = os.getenv("MISTRAL_MODEL", "open-mistral-7b")
    return ChatMistralAI(
        model=model_name,
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
        max_retries=0,   # disable built-in tenacity retries – we do it ourselves
    )


def invoke_with_retry(chain, input_data, label: str = "LLM call"):
    """
    Invoke a LangChain chain/runnable with exponential back-off on 429.

    Args:
        chain:      Any LangChain Runnable (chain.invoke is called).
        input_data: The argument passed to chain.invoke().
        label:      Human-readable name shown in log messages.

    Returns:
        The string output from the chain.

    Raises:
        httpx.HTTPStatusError: if all retries are exhausted.
    """
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            result = chain.invoke(input_data)
            return result
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                # Add jitter so parallel callers don't all retry at the same time
                jitter   = random.uniform(0, delay * 0.3)
                wait_for = min(delay + jitter, RETRY_MAX_DELAY)
                logger.warning(
                    "[%s] Rate-limited (429). Attempt %d/%d – waiting %.1fs …",
                    label, attempt, MAX_RETRY_ATTEMPTS, wait_for,
                )
                time.sleep(wait_for)
                delay *= 2          # exponential back-off
            else:
                raise               # non-429 errors bubble up immediately
    # Final attempt after all sleeps
    return chain.invoke(input_data)


def invoke_chunks_with_delay(chain, chunks: list, label: str = "chunk") -> list:
    """
    Invoke a chain on a list of text chunks one-by-one,
    sleeping INTER_CALL_DELAY seconds between calls to stay under rate limits.
    Each call is itself wrapped with retry logic.
    """
    results = []
    for i, chunk in enumerate(chunks):
        result = invoke_with_retry(chain, {"text": chunk}, label=f"{label}[{i+1}/{len(chunks)}]")
        results.append(result)
        if i < len(chunks) - 1:          # no sleep after the last chunk
            time.sleep(INTER_CALL_DELAY)
    return results
