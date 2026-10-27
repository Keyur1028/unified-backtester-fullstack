"""Lightweight retrieval over the active engine's field/operator capability catalog."""
from __future__ import annotations

import math
import re
from typing import Any

from vector_retrieval import retrieve_catalog_vectors


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{1,40}", re.IGNORECASE)
_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
              "in", "into", "is", "it", "of", "on", "or", "that", "the", "this", "to",
              "with", "idea", "strategy", "signal", "alpha"}


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(str(text).replace("_", " "))
            if token.casefold() not in _STOPWORDS}


def _score(query_tokens: set[str], name: str, description: str = "") -> float:
    item_tokens = _tokens(name + " " + description)
    if not query_tokens or not item_tokens:
        return 0.0
    overlap = len(query_tokens & item_tokens)
    exact_name = 1.0 if name.casefold() in query_tokens else 0.0
    return overlap / math.sqrt(len(query_tokens) * len(item_tokens)) + exact_name


def retrieve_catalog_context(
    query: str,
    interface: dict[str, Any],
    fields: set[str],
    operators: set[str],
    limit_per_kind: int = 8,
    vector_db_path: str | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Rank live registry entries with local vectors, falling back to lexical matching."""
    field_meta = interface.get("fields", {}) if isinstance(interface, dict) else {}
    field_kinds: dict[str, str] = {}
    if isinstance(field_meta, dict):
        for key, value in field_meta.items():
            if isinstance(value, list):
                for item in value:
                    field_kinds[str(item)] = str(key)
    operator_meta = interface.get("operators", {}) if isinstance(interface, dict) else {}
    semantics: dict[str, str] = {}
    if isinstance(operator_meta, dict):
        for key in ("verified_semantics", "unverified_semantics"):
            value = operator_meta.get(key, {})
            if isinstance(value, dict):
                for name, description in value.items():
                    semantics[str(name)] = str(description)

    if vector_db_path:
        catalog_entries = [
            {"kind": "field", "name": name, "description": field_kinds.get(name, "registered field")}
            for name in sorted(fields)
        ]
        catalog_entries.extend(
            {"kind": "operator", "name": name, "description": semantics.get(name, "")}
            for name in sorted(operators)
        )
        try:
            vector_rows = retrieve_catalog_vectors(
                vector_db_path, query, catalog_entries, limit=len(catalog_entries)
            )
        except Exception:
            vector_rows = []
        if vector_rows:
            field_rows = [
                {"name": row["name"], "kind": row["description"] or "registered",
                 "score": row["score"]}
                for row in vector_rows if row["kind"] == "field"
            ][:max(0, int(limit_per_kind))]
            operator_rows = [
                {"name": row["name"],
                 "semantics": row["description"] or "No verified semantic description in the engine contract.",
                 "score": row["score"]}
                for row in vector_rows if row["kind"] == "operator"
            ][:max(0, int(limit_per_kind))]
            return {"fields": field_rows, "operators": operator_rows}

    query_tokens = _tokens(query)
    field_rows = []
    for name in sorted(fields):
        score = _score(query_tokens, name)
        if score > 0:
            field_rows.append({"name": name, "kind": field_kinds.get(name, "registered"),
                               "score": f"{score:.4f}"})
    operator_rows = []
    for name in sorted(operators):
        description = semantics.get(name, "")
        score = _score(query_tokens, name, description)
        if score > 0:
            operator_rows.append({
                "name": name,
                "semantics": description or "No verified semantic description in the engine contract.",
                "score": f"{score:.4f}",
            })
    field_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    operator_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return {
        "fields": field_rows[:max(0, int(limit_per_kind))],
        "operators": operator_rows[:max(0, int(limit_per_kind))],
    }
