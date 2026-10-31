"""SQLite-backed local vector retrieval with an optional local sentence-transformer.

No model is downloaded. If ALPHA_EMBEDDING_MODEL points to an existing local
SentenceTransformer directory and sentence-transformers is installed, semantic
embeddings are used. Otherwise a deterministic hashed lexical vector is used.
Only vectors, tokenized safety metadata, and bounded candidate payloads persist.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from similarity_retrieval import _compact_response, idea_tokens

VECTOR_DIMENSIONS = 384
MAX_ENTRIES = 300
_MAX_DB_BYTES = 25 * 1024 * 1024
_LOCAL_MODEL_CACHE: dict[str, Any] = {}


def _local_embedding_model():
    model_path = os.getenv("ALPHA_EMBEDDING_MODEL", "").strip()
    if not model_path or not Path(model_path).is_dir():
        return None, "hashed-token-v1"
    if model_path in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[model_path], "sentence-transformer:" + str(Path(model_path).resolve())
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path, device="cpu")
        _LOCAL_MODEL_CACHE[model_path] = model
        return model, "sentence-transformer:" + str(Path(model_path).resolve())
    except Exception:
        # Missing package, incompatible model, or corrupt local cache: no network
        # fallback is attempted; the deterministic local vectorizer remains usable.
        return None, "hashed-token-v1"


def embed_text(text: str) -> tuple[list[float], str]:
    """Embed text locally and return (normalized vector, embedding backend id)."""
    model, backend_id = _local_embedding_model()
    if model is not None:
        try:
            vector = model.encode(str(text), normalize_embeddings=True).tolist()
            return [float(value) for value in vector], backend_id
        except Exception:
            pass

    tokens = idea_tokens(str(text).replace("_", " "))
    vector = [0.0] * VECTOR_DIMENSIONS
    features = list(tokens)
    features.extend("bi:" + tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1))
    for token in features:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % VECTOR_DIMENSIONS
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector, "hashed-token-v1"


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    score = sum(a * b for a, b in zip(left, right))
    return max(-1.0, min(1.0, score))


def _connect(path: str | Path) -> sqlite3.Connection:
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists() and database.stat().st_size > _MAX_DB_BYTES:
        raise ValueError("Local vector database exceeds the configured size limit.")
    connection = sqlite3.connect(str(database), timeout=5)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS idea_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature TEXT NOT NULL,
            model TEXT NOT NULL,
            catalog_fingerprint TEXT NOT NULL,
            embedding_backend TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            safety_tokens_json TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(signature, model, catalog_fingerprint, embedding_backend)
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS catalog_vectors (
            catalog_content_fingerprint TEXT NOT NULL,
            embedding_backend TEXT NOT NULL,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            PRIMARY KEY(catalog_content_fingerprint, embedding_backend, kind, name)
        )"""
    )
    return connection


@contextmanager
def _database(path: str | Path):
    connection = _connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def retrieve_similar_vectors(
    path: str | Path,
    idea: str,
    model: str,
    catalog_fingerprint: str,
    limit: int = 3,
    min_similarity: float = 0.25,
    safety_text: str | None = None,
) -> list[dict[str, Any]]:
    """Search stored vectors within the exact model/catalog/backend scope."""
    query_vector, backend_id = embed_text(idea)
    query_safety_tokens = idea_tokens(safety_text if safety_text is not None else idea)
    if not any(query_vector):
        return []
    try:
        with _database(path) as connection:
            rows = connection.execute(
                """SELECT embedding_json, safety_tokens_json, response_json
                   FROM idea_vectors
                   WHERE model = ? AND catalog_fingerprint = ? AND embedding_backend = ?
                   ORDER BY id DESC LIMIT ?""",
                (model, catalog_fingerprint, backend_id, MAX_ENTRIES),
            ).fetchall()
    except (OSError, sqlite3.Error, ValueError):
        return []

    results = []
    for vector_json, safety_json, response_json in rows:
        try:
            stored_vector = json.loads(vector_json)
            safety_tokens = json.loads(safety_json)
            response = json.loads(response_json)
            if not isinstance(stored_vector, list) or not isinstance(safety_tokens, list):
                continue
            if not isinstance(response, dict) or not isinstance(response.get("ideas"), list):
                continue
            score = _cosine(query_vector, [float(value) for value in stored_vector])
            if score >= min_similarity:
                results.append({
                    "similarity": round(score, 4),
                    "matched_tokens": [str(token) for token in safety_tokens],
                    "ideas": response["ideas"],
                    "retrieval_strategy": "local-vector",
                })
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    results.sort(key=lambda row: row["similarity"], reverse=True)
    return results[:max(0, int(limit))]


def store_idea_vector_response(
    path: str | Path,
    idea: str,
    model: str,
    catalog_fingerprint: str,
    response: dict[str, Any],
    safety_text: str | None = None,
    max_entries: int = MAX_ENTRIES,
) -> None:
    """Persist a vector and bounded candidate metadata, never the raw prompt."""
    vector, backend_id = embed_text(idea)
    if not any(vector):
        return
    safety_tokens = idea_tokens(safety_text if safety_text is not None else idea)
    signature = hashlib.sha256(
        (backend_id + "|" + " ".join(idea_tokens(idea)) + "|safety|" + " ".join(safety_tokens)).encode("utf-8")
    ).hexdigest()
    compact = _compact_response(response)
    with _database(path) as connection:
        connection.execute(
            """INSERT OR REPLACE INTO idea_vectors
               (signature, model, catalog_fingerprint, embedding_backend,
                embedding_json, safety_tokens_json, response_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (signature, model, catalog_fingerprint, backend_id,
             json.dumps(vector, separators=(",", ":")),
             json.dumps(safety_tokens, separators=(",", ":")),
             json.dumps(compact, ensure_ascii=False, separators=(",", ":"))),
        )
        connection.execute(
            "DELETE FROM idea_vectors WHERE id NOT IN (SELECT id FROM idea_vectors ORDER BY id DESC LIMIT ?)",
            (max(1, int(max_entries)),),
        )

def _catalog_content_fingerprint(entries: list[dict[str, str]]) -> str:
    canonical = json.dumps(
        sorted(entries, key=lambda row: (row["kind"], row["name"], row.get("description", ""))),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def retrieve_catalog_vectors(
    path: str | Path,
    query: str,
    catalog_entries: list[dict[str, str]],
    limit: int = 20,
) -> list[dict[str, str]]:
    """Vector-search the active field/operator catalog, indexing it on first use."""
    entries = [
        {"kind": str(row.get("kind", "")), "name": str(row.get("name", "")),
         "description": str(row.get("description", ""))}
        for row in catalog_entries
        if isinstance(row, dict) and row.get("kind") and row.get("name")
    ]
    if not entries:
        return []
    query_vector, backend_id = embed_text(query)
    if not any(query_vector):
        return []
    fingerprint = _catalog_content_fingerprint(entries)
    try:
        with _database(path) as connection:
            existing = connection.execute(
                """SELECT COUNT(*) FROM catalog_vectors
                   WHERE catalog_content_fingerprint = ? AND embedding_backend = ?""",
                (fingerprint, backend_id),
            ).fetchone()[0]
            if not existing:
                connection.execute(
                    "DELETE FROM catalog_vectors WHERE catalog_content_fingerprint != ?", (fingerprint,)
                )
                for entry in entries:
                    entry_vector, entry_backend = embed_text(
                        entry["kind"] + ": " + entry["name"] + " " + entry["description"]
                    )
                    if entry_backend != backend_id:
                        continue
                    connection.execute(
                        """INSERT OR REPLACE INTO catalog_vectors
                           (catalog_content_fingerprint, embedding_backend, kind, name,
                            description, embedding_json)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (fingerprint, backend_id, entry["kind"], entry["name"],
                         entry["description"], json.dumps(entry_vector, separators=(",", ":"))),
                    )
            rows = connection.execute(
                """SELECT kind, name, description, embedding_json FROM catalog_vectors
                   WHERE catalog_content_fingerprint = ? AND embedding_backend = ?""",
                (fingerprint, backend_id),
            ).fetchall()
    except (OSError, sqlite3.Error, ValueError):
        return []

    results = []
    for kind, name, description, vector_json in rows:
        try:
            stored_vector = json.loads(vector_json)
            score = _cosine(query_vector, [float(value) for value in stored_vector])
            results.append({
                "kind": str(kind), "name": str(name), "description": str(description),
                "score": f"{score:.4f}",
            })
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    results.sort(key=lambda row: float(row["score"]), reverse=True)
    return results[:max(0, int(limit))]

