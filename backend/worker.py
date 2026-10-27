"""Poll persisted execution jobs and execute them outside HTTP requests.

Run from the backend directory with the same database environment as the API:
    python worker.py

A worker refreshes updated_at while a job is running. Jobs whose heartbeat is
older than EXECUTION_JOB_STALE_MINUTES are marked failed, never silently rerun.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone

from app import JOB_STORE, app, execute_execution_job


def _iso_cutoff(stale_minutes: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)).isoformat()


def _execute_with_heartbeat(job_id: str, heartbeat_seconds: float) -> None:
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(heartbeat_seconds):
            try:
                if not JOB_STORE.heartbeat_job(job_id):
                    app.logger.warning(
                        "Execution job %s is no longer running; stopping heartbeat", job_id
                    )
                    return
            except Exception:
                app.logger.exception("Heartbeat failed for execution job %s", job_id)

    thread = threading.Thread(
        target=heartbeat, name=f"job-heartbeat-{job_id}", daemon=True
    )
    thread.start()
    try:
        execute_execution_job(job_id)
    finally:
        stop.set()
        thread.join(timeout=max(1.0, heartbeat_seconds + 0.5))


def run_worker() -> None:
    poll_seconds = max(0.2, float(os.getenv("EXECUTION_JOB_POLL_SECONDS", "1.0")))
    batch_size = min(100, max(1, int(os.getenv("EXECUTION_JOB_BATCH_SIZE", "10"))))
    heartbeat_seconds = max(1.0, float(os.getenv("EXECUTION_JOB_HEARTBEAT_SECONDS", "10")))
    stale_minutes = max(2.0, float(os.getenv("EXECUTION_JOB_STALE_MINUTES", "30")))
    app.logger.info(
        "Execution worker started (poll=%ss, batch=%s, heartbeat=%ss, stale_after=%sm)",
        poll_seconds, batch_size, heartbeat_seconds, stale_minutes,
    )
    while True:
        try:
            failed_count = JOB_STORE.fail_stale_running_jobs(_iso_cutoff(stale_minutes))
            if failed_count:
                app.logger.warning(
                    "Marked %s abandoned execution job(s) failed after heartbeat timeout",
                    failed_count,
                )
            queued = JOB_STORE.list_queued_jobs(limit=batch_size)
            if not queued:
                time.sleep(poll_seconds)
                continue
            for job in queued:
                try:
                    _execute_with_heartbeat(job["job_id"], heartbeat_seconds)
                except Exception:
                    # A concurrent worker may already have claimed the job; log and
                    # continue so one bad record does not stop this worker process.
                    app.logger.exception("Could not process execution job %s", job.get("job_id"))
        except Exception:
            app.logger.exception("Execution worker polling failed")
            time.sleep(poll_seconds)


if __name__ == "__main__":
    run_worker()
