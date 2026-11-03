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
MAX_PROMPT_CHARS = 80_000
DEFAULT_MAX_OUTPUT_TOKENS = 4096

# One SDK client per (key, timeout): building one per call re-opens its connection pool.
_CLIENTS: dict[tuple[str, int], Any] = {}


def _positive_int(name: str, default: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if 1 <= value <= maximum else default


def _status(exc: Exception) -> int | None:
    # google-genai exposes the HTTP status as ``code``; other SDKs use ``status_code``.
    for value in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _is_retryable(exc: Exception) -> bool:
    """Retry only transient rate-limit, server, or connection failures.

    Never a timeout (the provider may already have billed it) and never a daily quota
    (retrying cannot succeed until it resets).
    """
    text = str(exc).casefold()
    if "perday" in text or "per day" in text or "insufficient_quota" in text:
        return False
    status = _status(exc)
    if status is not None:
        return status == 429 or 500 <= status <= 599
    name = type(exc).__name__.casefold()
    return "timeout" not in name and any(token in name for token in ("connection", "temporar", "unavailable"))


def plain_reason(exc: Exception) -> str | None:
    """A provider failure in words the user can act on; None when no specific advice applies."""
    status, text = _status(exc), str(exc).casefold()
    if status in (401, 403) or any(token in text for token in ("api_key_invalid", "permission_denied", "unauthenticated")):
        return "Gemini refused the API key. Check GEMINI_API_KEY in backend/.env is complete and still valid."
    if status == 404 or ("not found" in text and "model" in text):
        return "That Gemini model is not available to this key. Set GEMINI_MODEL to one your key can use."
    if status == 429 or "resource_exhausted" in text:
        return "Gemini's quota for this key is used up. Wait for it to reset, or use another key."
    if status == 503:
        return "Gemini is overloaded right now. Try again in a minute."
    return None


def _default_client(api_key: str, timeout_ms: int) -> Any:
    from google import genai

    try:
        from google.genai import types

        options: dict[str, Any] = {"http_options": types.HttpOptions(timeout=timeout_ms)}
    except ImportError:
        # Older SDKs have no HttpOptions; the official dependency does.
        options = {}
    # Keyed on the class too, so a replaced SDK (as in tests) never reuses an old client.
    key = (genai.Client, api_key, timeout_ms)
    if key not in _CLIENTS:
        _CLIENTS[key] = genai.Client(api_key=api_key, **options)
    return _CLIENTS[key]


def invoke_json_generation(
    api_key: str,
    model: str,
    prompt: str,
    *,
    client_factory: Callable[..., Any] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> str:
    """Call Gemini with a hard timeout and at most one configurable retry.

    Retries default to zero; opt in with ALPHA_MODEL_RETRIES=1. ``response_schema``
    constrains the reply to a JSON schema when given.
    """
    if not api_key.strip():
        raise ValueError("A non-empty model API key is required.")
    if not str(prompt).strip():
        raise ValueError("Generation prompt must not be empty.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("Generation prompt exceeded the configured character limit.")
    timeout_ms = _positive_int("ALPHA_MODEL_TIMEOUT_MS", DEFAULT_TIMEOUT_MS, 120_000)
    retries = 1 if os.getenv("ALPHA_MODEL_RETRIES", "0").strip() == "1" else 0
    max_output_tokens = _positive_int("ALPHA_MODEL_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS, 8192)
    if max_output_tokens < 128:
        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
    config: dict[str, Any] = {
        "response_mime_type": "application/json",
        "temperature": 0.35,
        "max_output_tokens": max_output_tokens,
    }
    if response_schema is not None:
        config["response_json_schema"] = response_schema

    client = client_factory(api_key=api_key) if client_factory else _default_client(api_key, timeout_ms)
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
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
