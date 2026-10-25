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
DEFAULT_MAX_ENTRIES = 200


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
        "idea": normalize_idea(idea),
        "fields": sorted(fields),
        "operators": sorted(operators),
        "model": model,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_cached_response(path: str | Path, key: str) -> dict[str, Any] | None:
    """Return a defensive copy of a valid cache entry; treat corruption as a miss."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            return None
        entry = data.get("entries", {}).get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("response"), dict):
            return None
        return json.loads(json.dumps(entry["response"]))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def write_cached_response(
    path: str | Path,
    key: str,
    response: dict[str, Any],
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> None:
    """Atomically persist a response, retaining only the most recent entries."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    try:
        with destination.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, dict) and existing.get("version") == CACHE_VERSION:
            candidate = existing.get("entries", {})
            if isinstance(candidate, dict):
                entries = candidate
    except (OSError, ValueError, TypeError):
        pass

    entries.pop(key, None)
    entries[key] = {"response": response}
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
