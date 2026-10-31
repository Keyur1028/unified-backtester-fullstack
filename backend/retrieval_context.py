"""Bounded, compact examples for alpha-generation prompts."""
from __future__ import annotations

import json
from typing import Any

MAX_MATCHES = 2
MAX_CANDIDATES_PER_MATCH = 2
MAX_CONTEXT_CHARS = 6000
_FIELD_LIMITS = {
    "name": 100,
    "hypothesis": 300,
    "expression": 1000,
    "rationale": 250,
    "expected_behavior": 200,
}


def format_retrieval_context(matches: Any) -> str:
    """Serialize a small allowlist of retrieved fields under a strict size cap."""
    if not isinstance(matches, list):
        return "[]"

    compact = []
    for match in matches[:MAX_MATCHES]:
        if not isinstance(match, dict):
            continue
        raw_ideas = match.get("ideas")
        if not isinstance(raw_ideas, list):
            continue
        ideas = []
        for item in raw_ideas[:MAX_CANDIDATES_PER_MATCH]:
            if not isinstance(item, dict):
                continue
            candidate = {}
            for field, limit in _FIELD_LIMITS.items():
                value = item.get(field)
                if isinstance(value, str) and value.strip():
                    candidate[field] = value.strip()[:limit]
            if isinstance(item.get("risks"), list):
                risks = [
                    str(risk).strip()[:120]
                    for risk in item["risks"][:2]
                    if str(risk).strip()
                ]
                if risks:
                    candidate["risks"] = risks
            if candidate:
                ideas.append(candidate)
        if ideas:
            compact.append({
                "similarity": match.get("similarity"),
                "ideas": ideas,
            })

    encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    while len(encoded) > MAX_CONTEXT_CHARS and compact:
        last_match = compact[-1]
        if len(last_match["ideas"]) > 1:
            last_match["ideas"].pop()
        else:
            compact.pop()
        encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    return encoded
