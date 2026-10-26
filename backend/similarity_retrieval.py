"""Small local retrieval index for similar prior alpha ideas.

This is deterministic lexical retrieval (token-weighted cosine similarity), not
an embedding model or a semantic vector database. Raw prompts are not persisted;
the index stores normalized tokens plus bounded candidate metadata. Entries
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
MAX_TOKENS_PER_IDEA = 256
MAX_STORED_CANDIDATES = 3
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{1,40}", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "with", "would", "want", "strategy", "idea", "capture", "using",
}
_CANDIDATE_TEXT_LIMITS = {
    "id": 48,
    "name": 100,
    "hypothesis": 300,
    "expression": 1000,
    "rationale": 250,
    "expected_behavior": 200,
    "validation": 80,
}


def idea_tokens(idea: str) -> list[str]:
    """Return bounded normalized lexical features, not the original prompt."""
    return [
        token.casefold()
        for token in _TOKEN_RE.findall(str(idea))
        if token.casefold() not in _STOPWORDS
    ][:MAX_TOKENS_PER_IDEA]


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


def _compact_response(response: Any) -> dict[str, Any]:
    """Keep only bounded candidate fields needed for retrieval and reuse."""
    raw_ideas = response.get("ideas", []) if isinstance(response, dict) else []
    if not isinstance(raw_ideas, list):
        return {"ideas": []}

    ideas = []
    for item in raw_ideas[:MAX_STORED_CANDIDATES]:
        if not isinstance(item, dict):
            continue
        candidate: dict[str, Any] = {}
        expression = item.get("expression")
        if isinstance(expression, str) and expression.strip():
            # Never truncate expressions: a clipped expression could change meaning.
            if len(expression.strip()) > _CANDIDATE_TEXT_LIMITS["expression"]:
                continue
        for field, limit in _CANDIDATE_TEXT_LIMITS.items():
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                candidate[field] = value.strip()[:limit]
        if isinstance(item.get("risks"), list):
            risks = [
                risk.strip()[:160]
                for risk in item["risks"][:3]
                if isinstance(risk, str) and risk.strip()
            ]
            if risks:
                candidate["risks"] = risks
        provenance = item.get("provenance")
        if isinstance(provenance, dict):
            compact_provenance = {
                key: str(provenance[key])[:120]
                for key in ("validator", "catalog_fingerprint", "scope")
                if isinstance(provenance.get(key), str)
            }
            if compact_provenance:
                candidate["provenance"] = compact_provenance
        if candidate:
            ideas.append(candidate)
    return {"ideas": ideas}


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
        score = cosine_similarity(tokens, [str(x) for x in stored_tokens[:MAX_TOKENS_PER_IDEA]])
        if score >= min_similarity:
            ideas = response.get("ideas", [])
            results.append({
                "similarity": round(score, 4),
                "matched_tokens": [str(token).casefold() for token in stored_tokens[:MAX_TOKENS_PER_IDEA]],
                "ideas": ideas[:MAX_STORED_CANDIDATES] if isinstance(ideas, list) else [],
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
    """Persist bounded candidate metadata and tokens, never the raw prompt."""
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
        "response": _compact_response(response),
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
