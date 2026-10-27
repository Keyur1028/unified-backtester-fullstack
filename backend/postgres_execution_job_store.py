"""PostgreSQL persistence for execution jobs, using psycopg 3.

The schema and public record shape intentionally match SQLiteExecutionJobStore.
This is persistence only: execution remains synchronous and no queue is introduced.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from execution_job_contract import ExecutionJobStatus, RunConfiguration
from execution_job_store import (
    ExecutionJobNotFoundError,
    InvalidJobTransitionError,
    _ALLOWED_TRANSITIONS,
    _json_object,
    _now,
)


class PostgresExecutionJobStore:
    """Durable execution-job repository backed by PostgreSQL."""

    def __init__(self, database_url: str):
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("A PostgreSQL connection URL is required.")
        self.database_url = database_url.strip()
        self._initialize()

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL execution-job storage requires psycopg 3; "
                "install backend/requirements.txt."
            ) from exc
        return psycopg.connect(self.database_url, row_factory=dict_row)

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
                ) VALUES (%s, %s, %s, %s, NULL, NULL, %s, %s)
                """,
                (
                    identifier, ExecutionJobStatus.QUEUED.value,
                    configuration.canonical_json, configuration.fingerprint,
                    timestamp, timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM execution_jobs WHERE job_id = %s", (identifier,)
            ).fetchone()
        return self._decode(row)

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return a detached job record or raise if the identifier is unknown."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM execution_jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
        if row is None:
            raise ExecutionJobNotFoundError(
                f"Execution job '{job_id}' was not found."
            )
        return self._decode(row)

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the newest job records, with a bounded page size."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM execution_jobs "
                "ORDER BY created_at DESC, job_id DESC LIMIT %s",
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
        """Atomically apply an allowed transition and optional terminal payload."""
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
                raise ValueError(
                    "Only succeeded or failed jobs may store a result or error."
                )
            result_json = error_json = None

        with self._connect() as connection:
            # Row locking prevents two writers from validating the same old status.
            row = connection.execute(
                "SELECT status FROM execution_jobs WHERE job_id = %s FOR UPDATE",
                (job_id,),
            ).fetchone()
            if row is None:
                raise ExecutionJobNotFoundError(
                    f"Execution job '{job_id}' was not found."
                )
            current = ExecutionJobStatus(row["status"])
            if new_status not in _ALLOWED_TRANSITIONS[current]:
                raise InvalidJobTransitionError(
                    f"Cannot transition execution job from '{current.value}' "
                    f"to '{new_status.value}'."
                )
            connection.execute(
                """
                UPDATE execution_jobs
                SET status = %s, result_json = %s, error_json = %s, updated_at = %s
                WHERE job_id = %s
                """,
                (new_status.value, result_json, error_json, _now(), job_id),
            )
            updated = connection.execute(
                "SELECT * FROM execution_jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
        return self._decode(updated)

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
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
