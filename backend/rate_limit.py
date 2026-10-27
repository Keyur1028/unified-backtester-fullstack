"""SQLite-backed fixed-window rate limiter for expensive API endpoints.

The client key is hashed before persistence. This is suitable for a single
application instance or shared SQLite volume; horizontally scaled deployments
should use a shared atomic store (for example Redis) instead.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if minimum <= value <= maximum else default


def check_rate_limit(
    client_key: str,
    *,
    database_path: str | Path,
    limit: int | None = None,
    window_seconds: int | None = None,
    now: int | None = None,
) -> tuple[bool, int]:
    """Return (allowed, retry_after_seconds) using an atomic fixed time window."""
    limit = limit if limit is not None else _bounded_env_int("ALPHA_GENERATION_RATE_LIMIT", 30, 1, 10000)
    window_seconds = window_seconds if window_seconds is not None else _bounded_env_int(
        "ALPHA_GENERATION_RATE_WINDOW_SECONDS", 60, 1, 86400
    )
    if limit < 1 or window_seconds < 1:
        raise ValueError("Rate limit and window must be positive integers.")
    current = int(time.time()) if now is None else int(now)
    window_start = current - (current % window_seconds)
    retry_after = max(1, window_start + window_seconds - current)
    key_hash = hashlib.sha256(str(client_key).encode("utf-8")).hexdigest()
    database = Path(database_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(database), timeout=5, isolation_level=None) as connection:
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS api_rate_limits (
                key_hash TEXT NOT NULL,
                window_start INTEGER NOT NULL,
                request_count INTEGER NOT NULL,
                PRIMARY KEY (key_hash, window_start)
            )"""
        )
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT request_count FROM api_rate_limits WHERE key_hash=? AND window_start=?",
                (key_hash, window_start),
            ).fetchone()
            count = int(row[0]) if row else 0
            if count >= limit:
                connection.execute("COMMIT")
                return False, retry_after
            connection.execute(
                """INSERT INTO api_rate_limits(key_hash, window_start, request_count)
                   VALUES (?, ?, 1)
                   ON CONFLICT(key_hash, window_start)
                   DO UPDATE SET request_count=request_count+1""",
                (key_hash, window_start),
            )
            # Bound storage by removing windows older than two days.
            connection.execute(
                "DELETE FROM api_rate_limits WHERE window_start < ?",
                (window_start - 2 * 86400,),
            )
            connection.execute("COMMIT")
            return True, 0
        except Exception:
            connection.execute("ROLLBACK")
            raise
