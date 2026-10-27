"""Select the execution-job persistence backend from environment configuration."""
from __future__ import annotations

from execution_job_store import SQLiteExecutionJobStore


def create_execution_job_store(
    database_url: str | None = None,
    sqlite_path: str = "storage/execution_jobs.sqlite3",
):
    """Use PostgreSQL only when explicitly configured; otherwise use SQLite.

    A configured PostgreSQL URL never silently falls back to local storage if the
    database is unavailable, because that would split job history across backends.
    """
    url = (database_url or "").strip()
    if url:
        if not url.lower().startswith(("postgres://", "postgresql://")):
            raise ValueError(
                "EXECUTION_JOB_DATABASE_URL must be a PostgreSQL connection URL."
            )
        from postgres_execution_job_store import PostgresExecutionJobStore

        return PostgresExecutionJobStore(url)
    return SQLiteExecutionJobStore(sqlite_path)
