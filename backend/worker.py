"""Poll persisted execution jobs and execute them outside HTTP requests.

Run from the backend directory with the same database environment as the API:
    python worker.py
"""
from __future__ import annotations

import os
import time

from app import JOB_STORE, app, execute_execution_job


def run_worker() -> None:
    poll_seconds = max(0.2, float(os.getenv("EXECUTION_JOB_POLL_SECONDS", "1.0")))
    batch_size = min(100, max(1, int(os.getenv("EXECUTION_JOB_BATCH_SIZE", "10"))))
    app.logger.info(
        "Execution worker started (poll=%ss, batch=%s)", poll_seconds, batch_size
    )
    while True:
        try:
            queued = JOB_STORE.list_queued_jobs(limit=batch_size)
            if not queued:
                time.sleep(poll_seconds)
                continue
            for job in queued:
                try:
                    execute_execution_job(job["job_id"])
                except Exception:
                    # A concurrent worker may already have claimed the job; log and
                    # continue so one bad record does not stop this worker process.
                    app.logger.exception("Could not process execution job %s", job.get("job_id"))
        except Exception:
            app.logger.exception("Execution worker polling failed")
            time.sleep(poll_seconds)


if __name__ == "__main__":
    run_worker()
