"""Strict contract for model-generated Idea → Expression payloads."""
from __future__ import annotations

import json
from typing import Any

MAX_CANDIDATES = 5
MAX_RESPONSE_CHARS = 128_000


class ModelResponseContractError(ValueError):
    """Raised when a model response violates the documented JSON contract."""


def parse_idea_candidates(response_text: str) -> list[dict[str, Any]]:
    """Parse provider output as a bounded JSON object containing candidate objects.

    This layer checks shape only. Candidate fields, expression syntax, registry
    membership, and engine runtime compatibility are validated downstream.
    """
    if not isinstance(response_text, str) or not response_text.strip():
        raise ModelResponseContractError("Model returned an empty response.")
    if len(response_text) > MAX_RESPONSE_CHARS:
        raise ModelResponseContractError("Model response exceeded the configured size limit.")
    try:
        payload = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ModelResponseContractError("Model response is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ModelResponseContractError("Model response must be a JSON object.")
    candidates = payload.get("ideas")
    if not isinstance(candidates, list):
        raise ModelResponseContractError("Model response must contain an ideas array.")
    if len(candidates) > MAX_CANDIDATES:
        raise ModelResponseContractError(f"Model returned more than {MAX_CANDIDATES} candidates.")
    if any(not isinstance(candidate, dict) for candidate in candidates):
        raise ModelResponseContractError("Every item in ideas must be a JSON object.")
    return candidates
