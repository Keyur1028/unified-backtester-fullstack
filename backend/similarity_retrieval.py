"""Small local retrieval index for similar prior alpha ideas.

This is deterministic lexical retrieval (token-weighted cosine similarity), not
an embedding model or a semantic vector database. Raw prompts are not persisted;
the index stores normalized tokens plus validated candidate metadata. Entries
are scoped to the active model and engine catalog fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

INDEX_VERSION = 1
DEFAULT_MAX_ENTRIES = 300
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{1,40}", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "with", "would", "want", "strategy", "idea", "capture", "using",
}


def idea_tokens(idea: str) -> list[str]:
    """Return normalized lexical features without retaining the original prompt."""
    return [
        token.casefold()
        for token in _TOKEN_RE.findall(str(idea))
        if token.casefold() not in _STOPWORDS
    ]


def cosine_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    """Cosine similarity over token-frequency vectors."""
    from collections import Counter

    a, b = Counter(left), Counter(right)
    if not a or not b:
        return 0.0
    dot = sum(count * b.get(token, 0) for token, count in a.items())
    norm_a = math.sqrt(sum(count * count for count in a.values()))
    norm_b = math.sqrt(sum(count * count for count in b.values()))
    return dot / (norm_a * norm_b)


def retrieve_similar(
    path: str | Path,
    idea: str,
    model: str,
    catalog_fingerprint: str,
    limit: int = 3,
    min_similarity: float = 0.25,
) -> list[dict[str, Any]]:
    """Return highest-scoring compatible prior responses, without raw prompts."""
    tokens = idea_tokens(idea)
    if not tokens:
        return []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or data.get("version") != INDEX_VERSION:
            return []
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            return []
    except (OSError, ValueError, TypeError):
        return []

    results = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("model") != model or entry.get("catalog_fingerprint") != catalog_fingerprint:
            continue
        stored_tokens = entry.get("tokens")
        response = entry.get("response")
        if not isinstance(stored_tokens, list) or not isinstance(response, dict):
            continue
        score = cosine_similarity(tokens, [str(x) for x in stored_tokens])
        if score >= min_similarity:
            results.append({
                "similarity": round(score, 4),
                "ideas": response.get("ideas", [])[:3] if isinstance(response.get("ideas"), list) else [],
            })
    results.sort(key=lambda row: row["similarity"], reverse=True)
    return results[:max(0, int(limit))]


def store_idea_response(
    path: str | Path,
    idea: str,
    model: str,
    catalog_fingerprint: str,
    response: dict[str, Any],
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> None:
    """Persist validated response metadata and tokens, never the raw prompt."""
    tokens = idea_tokens(idea)
    if not tokens:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    try:
        with destination.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and data.get("version") == INDEX_VERSION:
            existing = data.get("entries", [])
            if isinstance(existing, list):
                entries = [x for x in existing if isinstance(x, dict)]
    except (OSError, ValueError, TypeError):
        pass

    # Replace duplicate token signatures within the same catalog/model scope.
    signature = hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()
    entries = [
        row for row in entries
        if not (row.get("token_signature") == signature
                and row.get("model") == model
                and row.get("catalog_fingerprint") == catalog_fingerprint)
    ]
    entries.append({
        "token_signature": signature,
        "tokens": tokens,
        "model": model,
        "catalog_fingerprint": catalog_fingerprint,
        "response": response,
    })
    entries = entries[-max(1, int(max_entries)):]
    fd, temporary = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"version": INDEX_VERSION, "entries": entries}, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, destination)
    finally:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        except OSError:
            pass
