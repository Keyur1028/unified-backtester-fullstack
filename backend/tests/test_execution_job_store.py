"""Offline tests for persisted execution-job lifecycle transitions."""
import os
import sys
import tempfile
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from execution_job_contract import ExecutionJobStatus, RunConfiguration  # noqa: E402
from execution_job_store import (  # noqa: E402
    ExecutionJobNotFoundError,
    InvalidJobTransitionError,
    SQLiteExecutionJobStore,
)


class SQLiteExecutionJobStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "jobs.sqlite3")
        self.store = SQLiteExecutionJobStore(self.database_path)
        self.configuration = RunConfiguration.from_payload({
            "expression": "rank(close)",
            "universe": "NIFTY50",
        })

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_and_read_job_persists_snapshot_and_fingerprint(self):
        created = self.store.create_job(self.configuration, job_id="job-1")
        self.assertEqual(created["job_id"], "job-1")
        self.assertEqual(created["status"], "queued")
        self.assertEqual(created["config"], self.configuration.to_dict())
        self.assertEqual(created["config_fingerprint"], self.configuration.fingerprint)
        self.assertIsNone(created["result"])
        self.assertIsNone(created["error"])

        # A new repository instance proves the record is stored on disk.
        reopened = SQLiteExecutionJobStore(self.database_path)
        self.assertEqual(reopened.get_job("job-1"), created)

    def test_successful_lifecycle_persists_result(self):
        self.store.create_job(self.configuration, job_id="job-2")
        running = self.store.transition("job-2", ExecutionJobStatus.RUNNING)
        self.assertEqual(running["status"], "running")
        succeeded = self.store.transition(
            "job-2", ExecutionJobStatus.SUCCEEDED,
            result={"summary": {"sharpe": 1.2}},
        )
        self.assertEqual(succeeded["status"], "succeeded")
        self.assertEqual(succeeded["result"], {"summary": {"sharpe": 1.2}})
        self.assertIsNone(succeeded["error"])

    def test_failure_lifecycle_persists_error(self):
        self.store.create_job(self.configuration, job_id="job-3")
        self.store.transition("job-3", "running")
        failed = self.store.transition(
            "job-3", "failed", error={"type": "EngineError", "message": "failed"}
        )
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"]["type"], "EngineError")

    def test_queued_job_can_be_cancelled(self):
        self.store.create_job(self.configuration, job_id="job-4")
        cancelled = self.store.transition("job-4", "cancelled")
        self.assertEqual(cancelled["status"], "cancelled")

    def test_terminal_state_cannot_be_changed(self):
        self.store.create_job(self.configuration, job_id="job-5")
        self.store.transition("job-5", "cancelled")
        with self.assertRaises(InvalidJobTransitionError):
            self.store.transition("job-5", "running")

    def test_running_job_cannot_transition_to_cancelled(self):
        self.store.create_job(self.configuration, job_id="job-running-cancel")
        self.store.transition("job-running-cancel", "running")
        with self.assertRaises(InvalidJobTransitionError):
            self.store.transition("job-running-cancel", "cancelled")
        self.assertEqual(self.store.get_job("job-running-cancel")["status"], "running")

    def test_queued_job_cannot_skip_running_to_succeed(self):
        self.store.create_job(self.configuration, job_id="job-6")
        with self.assertRaises(InvalidJobTransitionError):
            self.store.transition(
                "job-6", "succeeded", result={"summary": {}}
            )

    def test_terminal_payloads_must_be_json_objects(self):
        self.store.create_job(self.configuration, job_id="job-7")
        self.store.transition("job-7", "running")
        with self.assertRaisesRegex(TypeError, "result must be a JSON object"):
            self.store.transition("job-7", "succeeded", result=["not", "object"])
        with self.assertRaisesRegex(ValueError, "JSON-safe"):
            self.store.transition("job-7", "failed", error={"value": float("nan")})

    def test_list_jobs_returns_newest_records_with_bounded_limit(self):
        for index in range(3):
            self.store.create_job(
                RunConfiguration.from_payload({"expression": f"rank(close) + {index}"}),
                job_id=f"job-{index}",
            )
        jobs = self.store.list_jobs(limit=2)
        self.assertEqual(len(jobs), 2)
        self.assertEqual({job["job_id"] for job in jobs}, {"job-1", "job-2"})

    def test_list_jobs_rejects_invalid_limits(self):
        for limit in (0, 101, True, "2"):
            with self.subTest(limit=limit):
                with self.assertRaises((TypeError, ValueError)):
                    self.store.list_jobs(limit=limit)

    def test_unknown_job_raises_not_found(self):
        with self.assertRaises(ExecutionJobNotFoundError):
            self.store.get_job("missing")

    def test_returned_configuration_is_detached(self):
        created = self.store.create_job(self.configuration, job_id="job-8")
        created["config"]["universe"] = "TOP500"
        self.assertEqual(self.store.get_job("job-8")["config"]["universe"], "NIFTY50")

    def test_heartbeat_refreshes_only_running_jobs(self):
        self.store.create_job(self.configuration, job_id="heartbeat-1")
        self.assertFalse(self.store.heartbeat_job("heartbeat-1"))
        self.store.transition("heartbeat-1", "running")
        self.assertTrue(self.store.heartbeat_job("heartbeat-1"))
        self.assertFalse(self.store.heartbeat_job("missing-job"))

    def test_stale_running_jobs_are_failed_but_queued_jobs_are_untouched(self):
        self.store.create_job(self.configuration, job_id="stale-running")
        self.store.create_job(self.configuration, job_id="still-queued")
        self.store.transition("stale-running", "running")
        failed_count = self.store.fail_stale_running_jobs("9999-01-01T00:00:00+00:00")
        self.assertEqual(failed_count, 1)
        failed = self.store.get_job("stale-running")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"]["type"], "WorkerLeaseExpired")
        self.assertIsNone(failed["result"])
        self.assertEqual(self.store.get_job("still-queued")["status"], "queued")

    def test_heartbeat_prevents_stale_job_recovery(self):
        self.store.create_job(self.configuration, job_id="alive-running")
        self.store.transition("alive-running", "running")
        self.store.heartbeat_job("alive-running")
        failed_count = self.store.fail_stale_running_jobs("2000-01-01T00:00:00+00:00")
        self.assertEqual(failed_count, 0)
        self.assertEqual(self.store.get_job("alive-running")["status"], "running")

    def test_stale_cutoff_must_be_non_empty(self):
        for cutoff in ("", None):
            with self.subTest(cutoff=cutoff):
                with self.assertRaises(ValueError):
                    self.store.fail_stale_running_jobs(cutoff)


if __name__ == "__main__":
    unittest.main()
