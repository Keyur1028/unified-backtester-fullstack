"""Production boundary for remote model calls.

The gateway applies a finite request timeout, bounded transient retries, and
response-size checks. It never logs prompts, API keys, or model response bodies.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

DEFAULT_TIMEOUT_MS = 30_000
MAX_RESPONSE_CHARS = 128_000


def _positive_int(name: str, default: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if 1 <= value <= maximum else default


def _is_retryable(exc: Exception) -> bool:
    """Retry only likely transient network, rate-limit, or server errors."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status is not None:
        try:
            return int(status) == 429 or 500 <= int(status) <= 599
        except (TypeError, ValueError):
            return False
    # SDK transport errors usually expose a request/connection class name.
    name = type(exc).__name__.casefold()
    return any(token in name for token in ("timeout", "connection", "temporar", "unavailable"))


def invoke_json_generation(
    api_key: str,
    model: str,
    prompt: str,
    *,
    client_factory: Callable[..., Any] | None = None,
) -> str:
    """Call Gemini with a hard timeout and at most one configurable retry.

    Retries default to one and are bounded to two attempts. A timeout may occur
    after a provider has accepted a request, so retrying can still incur a charge.
    """
    if not api_key.strip():
        raise ValueError("A non-empty model API key is required.")
    if not str(prompt).strip():
        raise ValueError("Generation prompt must not be empty.")
    timeout_ms = _positive_int("ALPHA_MODEL_TIMEOUT_MS", DEFAULT_TIMEOUT_MS, 120_000)
    retries = _positive_int("ALPHA_MODEL_RETRIES", 1, 1)
    if client_factory is None:
        from google import genai

        try:
            from google.genai import types
            client_factory = lambda **kwargs: genai.Client(
                api_key=kwargs["api_key"],
                http_options=types.HttpOptions(timeout=timeout_ms),
            )
        except ImportError:
            # Compatibility path for older SDKs. The official google-genai
            # dependency supports HttpOptions; production should keep it current.
            client_factory = lambda **kwargs: genai.Client(api_key=kwargs["api_key"])

    client = client_factory(api_key=api_key)
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0.35},
            )
            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Model returned an empty response.")
            if len(text) > MAX_RESPONSE_CHARS:
                raise ValueError("Model response exceeded the configured size limit.")
            return text
        except Exception as exc:
            if attempt >= retries or not _is_retryable(exc):
                raise
            time.sleep(0.25 * (2 ** attempt))
    raise RuntimeError("Model generation exhausted retries without a response.")
