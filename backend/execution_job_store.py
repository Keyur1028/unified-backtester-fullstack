"""SQLite persistence primitives for execution jobs.

This is a storage boundary only: it does not enqueue or execute work and does not
change the synchronous HTTP API. Use a shared Postgres-backed implementation before
running multiple backend instances against independent local files.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from execution_job_contract import ExecutionJobStatus, RunConfiguration


_TERMINAL = {
    ExecutionJobStatus.SUCCEEDED,
    ExecutionJobStatus.FAILED,
    ExecutionJobStatus.CANCELLED,
}
_ALLOWED_TRANSITIONS = {
    ExecutionJobStatus.QUEUED: {
        ExecutionJobStatus.RUNNING,
        ExecutionJobStatus.CANCELLED,
    },
    ExecutionJobStatus.RUNNING: _TERMINAL,
    ExecutionJobStatus.SUCCEEDED: set(),
    ExecutionJobStatus.FAILED: set(),
    ExecutionJobStatus.CANCELLED: set(),
}


class ExecutionJobNotFoundError(LookupError):
    """Raised when a requested job does not exist."""


class InvalidJobTransitionError(ValueError):
    """Raised when a job lifecycle transition is not allowed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_object(value: Any, field_name: str) -> str:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a JSON object.")
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only JSON-safe values.") from exc


class SQLiteExecutionJobStore:
    """Small durable job repository using Python's standard-library SQLite.

    The database path is explicit so tests and callers can choose their storage.
    The store is not wired into Flask or the current synchronous backtest endpoint.
    """

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_execution_jobs_created_at "
                "ON execution_jobs(created_at DESC)"
            )

    def create_job(
        self, configuration: RunConfiguration, job_id: str | None = None
    ) -> dict[str, Any]:
        """Persist a queued job and return its detached public representation."""
        if not isinstance(configuration, RunConfiguration):
            raise TypeError("configuration must be a RunConfiguration.")
        identifier = job_id or str(uuid.uuid4())
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO execution_jobs (
                    job_id, status, config_json, config_fingerprint,
                    result_json, error_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    identifier, ExecutionJobStatus.QUEUED.value,
                    configuration.canonical_json, configuration.fingerprint,
                    timestamp, timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM execution_jobs WHERE job_id = ?", (identifier,)
            ).fetchone()
        return self._decode(row)

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return a detached job record or raise if the identifier is unknown."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM execution_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise ExecutionJobNotFoundError(f"Execution job '{job_id}' was not found.")
        return self._decode(row)

    def list_queued_jobs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return oldest queued jobs first for worker polling."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM execution_jobs WHERE status = ? "
                "ORDER BY created_at ASC, rowid ASC LIMIT ?",
                (ExecutionJobStatus.QUEUED.value, limit),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the newest job records, with a bounded page size."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM execution_jobs ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def transition(
        self,
        job_id: str,
        new_status: ExecutionJobStatus,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically apply an allowed state transition and optional terminal payload."""
        if not isinstance(new_status, ExecutionJobStatus):
            try:
                new_status = ExecutionJobStatus(new_status)
            except (TypeError, ValueError) as exc:
                raise ValueError("Unknown execution job status.") from exc

        if new_status == ExecutionJobStatus.SUCCEEDED:
            result_json = _json_object(result, "result")
            error_json = None
        elif new_status == ExecutionJobStatus.FAILED:
            error_json = _json_object(error, "error")
            result_json = None
        else:
            if result is not None or error is not None:
                raise ValueError("Only succeeded or failed jobs may store a result or error.")
            result_json = error_json = None

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM execution_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise ExecutionJobNotFoundError(
                    f"Execution job '{job_id}' was not found."
                )
            current = ExecutionJobStatus(row["status"])
            if new_status not in _ALLOWED_TRANSITIONS[current]:
                connection.rollback()
                raise InvalidJobTransitionError(
                    f"Cannot transition execution job from '{current.value}' "
                    f"to '{new_status.value}'."
                )
            connection.execute(
                """
                UPDATE execution_jobs
                SET status = ?, result_json = ?, error_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (new_status.value, result_json, error_json, _now(), job_id),
            )
            updated = connection.execute(
                "SELECT * FROM execution_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            connection.commit()
        return self._decode(updated)

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "config": json.loads(row["config_json"]),
            "config_fingerprint": row["config_fingerprint"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
