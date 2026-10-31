"""Small, dependency-free persistent cache for idea-to-expression responses.

Only a digest of the normalized prompt is persisted; raw user prompts are not
written to disk. The catalog/model fingerprint is part of the key so changes
to supported fields, operators, or model selection naturally invalidate hits.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

CACHE_VERSION = 1
# Increment whenever prompt instructions, output schema, or retrieval scoring change.
PROMPT_VERSION = 3
DEFAULT_MAX_ENTRIES = 200
MAX_CACHE_FILE_BYTES = 5 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024
MAX_IDEAS = 5
_TEXT_LIMITS = {
    "id": 48,
    "name": 100,
    "hypothesis": 300,
    "expression": 1000,
    "rationale": 250,
    "expected_behavior": 200,
    "validation": 80,
}
_TOP_LEVEL_TEXT_LIMITS = {"model": 100, "note": 600}
_PROVENANCE_KEYS = {"validator": 100, "catalog_fingerprint": 128, "scope": 100}


def normalize_idea(idea: str) -> str:
    """Normalize harmless whitespace/case differences for exact-match caching."""
    return " ".join(str(idea).casefold().split())


def make_cache_key(
    idea: str,
    fields: list[str],
    operators: list[str],
    model: str,
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "idea": normalize_idea(idea),
        "fields": sorted(fields),
        "operators": sorted(operators),
        "model": model,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compact_text_fields(source: Any, limits: dict[str, int]) -> dict[str, str]:
    if not isinstance(source, dict):
        return {}
    return {
        key: value.strip()[:limit]
        for key, limit in limits.items()
        if isinstance((value := source.get(key)), str) and value.strip()
    }


def _compact_response(response: Any) -> dict[str, Any] | None:
    """Allowlist and bound a cached API response before it reaches disk or UI."""
    if not isinstance(response, dict):
        return None
    if not isinstance(response.get("ideas"), list):
        # Preserve small generic entries for compatibility with the cache helper,
        # but never let an arbitrary payload inflate the persistent cache.
        try:
            encoded = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return None
        if len(encoded.encode("utf-8")) > MAX_RESPONSE_BYTES:
            return None
        return json.loads(encoded)

    compact: dict[str, Any] = {"ideas": []}
    for item in response["ideas"][:MAX_IDEAS]:
        if not isinstance(item, dict):
            continue
        candidate: dict[str, Any] = {}
        expression = item.get("expression")
        if isinstance(expression, str) and expression.strip():
            expression = expression.strip()
            # Expressions are never truncated: a clipped expression changes meaning.
            if len(expression) > _TEXT_LIMITS["expression"]:
                continue
        candidate.update(_compact_text_fields(item, _TEXT_LIMITS))
        if isinstance(item.get("risks"), list):
            risks = [
                risk.strip()[:160]
                for risk in item["risks"][:3]
                if isinstance(risk, str) and risk.strip()
            ]
            if risks:
                candidate["risks"] = risks
        provenance = _compact_text_fields(item.get("provenance"), _PROVENANCE_KEYS)
        if provenance:
            candidate["provenance"] = provenance
        if candidate:
            compact["ideas"].append(candidate)

    rejected = response.get("rejected")
    if isinstance(rejected, list):
        compact["rejected"] = [
            {
                key: value.strip()[:limit]
                for key, limit in {"expression": 1000, "reason": 250}.items()
                if isinstance((value := row.get(key)), str) and value.strip()
            }
            for row in rejected[:MAX_IDEAS]
            if isinstance(row, dict)
        ]
    compact.update(_compact_text_fields(response, _TOP_LEVEL_TEXT_LIMITS))
    provenance = _compact_text_fields(response.get("provenance"), _PROVENANCE_KEYS)
    if provenance:
        compact["provenance"] = provenance
    retrieval = response.get("retrieval")
    if isinstance(retrieval, dict):
        allowed = {
            "strategy": 80,
            "matches_used": None,
            "similarity": None,
            "raw_prompt_persisted": None,
        }
        compact["retrieval"] = {}
        for key, limit in allowed.items():
            value = retrieval.get(key)
            if limit is None and isinstance(value, (int, float, bool)):
                compact["retrieval"][key] = value
            elif isinstance(value, str) and limit is not None:
                compact["retrieval"][key] = value[:limit]
        if not compact["retrieval"]:
            compact.pop("retrieval")
    cache = response.get("cache")
    if isinstance(cache, dict):
        hit, strategy = cache.get("hit"), cache.get("strategy")
        if isinstance(hit, bool) and isinstance(strategy, str):
            compact["cache"] = {"hit": hit, "strategy": strategy[:80]}

    try:
        encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    if len(encoded.encode("utf-8")) > MAX_RESPONSE_BYTES:
        return None
    return compact


def read_cached_response(path: str | Path, key: str) -> dict[str, Any] | None:
    """Return a defensive, compact copy; treat corruption and oversized files as misses."""
    try:
        source = Path(path)
        if source.stat().st_size > MAX_CACHE_FILE_BYTES:
            return None
        with source.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            return None
        entries = data.get("entries")
        if not isinstance(entries, dict):
            return None
        entry = entries.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("response"), dict):
            return None
        return _compact_response(entry["response"])
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def write_cached_response(
    path: str | Path,
    key: str,
    response: dict[str, Any],
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> None:
    """Atomically persist bounded responses, retaining only recent entries."""
    compact_response = _compact_response(response)
    if compact_response is None:
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    try:
        if destination.stat().st_size <= MAX_CACHE_FILE_BYTES:
            with destination.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict) and existing.get("version") == CACHE_VERSION:
                candidate = existing.get("entries", {})
                if isinstance(candidate, dict):
                    for old_key, old_entry in candidate.items():
                        if not isinstance(old_key, str) or not isinstance(old_entry, dict):
                            continue
                        old_response = _compact_response(old_entry.get("response"))
                        if old_response is not None:
                            entries[old_key] = {"response": old_response}
    except (OSError, ValueError, TypeError):
        pass

    entries.pop(key, None)
    entries[key] = {"response": compact_response}
    limit = max(1, int(max_entries))
    entries = dict(list(entries.items())[-limit:])
    data = {"version": CACHE_VERSION, "entries": entries}

    fd, temporary = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, destination)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass
