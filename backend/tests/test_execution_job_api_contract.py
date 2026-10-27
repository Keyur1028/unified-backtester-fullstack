"""Offline Flask contract tests for queued execution-job endpoints."""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app as backend_app  # noqa: E402
from execution_job_store import SQLiteExecutionJobStore  # noqa: E402


class ExecutionJobApiContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SQLiteExecutionJobStore(
            os.path.join(self.temp_dir.name, "execution-jobs.sqlite3")
        )
        self.store_patch = patch.object(backend_app, "JOB_STORE", self.store)
        self.store_patch.start()
        self.backtest_patch = patch.object(backend_app, "run_backtest")
        self.mock_backtest = self.backtest_patch.start()
        backend_app.app.config.update(TESTING=True)
        self.client = backend_app.app.test_client()

    def tearDown(self):
        self.backtest_patch.stop()
        self.store_patch.stop()
        self.temp_dir.cleanup()

    def test_requires_json_content_type(self):
        response = self.client.post(
            "/api/execution-jobs", data='{"expression":"rank(close)"}',
            content_type="text/plain",
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.get_json()["error"]["type"], "ValidationError")

    def test_rejects_non_object_json_payload(self):
        response = self.client.post("/api/execution-jobs", json=["rank(close)"])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["type"], "ValidationError")
        self.mock_backtest.assert_not_called()

    def test_submission_returns_queued_job_without_running_backtest(self):
        payload = {
            "expression": "rank(close)",
            "universe": "NIFTY50",
            "start": "2024-01-01",
            "end": "2024-03-01",
        }
        response = self.client.post("/api/execution-jobs", json=payload)
        self.assertEqual(response.status_code, 202)
        job = response.get_json()
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["config"], payload)
        self.mock_backtest.assert_not_called()

    def test_worker_execution_persists_successful_result_and_is_retrievable(self):
        self.mock_backtest.return_value = {
            "status": "success",
            "experiment_id": "experiment-test-1",
            "result": {
                "net_returns": {}, "gross_returns": {}, "ic_series": {},
                "drawdown_series": {}, "turnover_series": {}, "cum_net_returns": {},
                "cum_gross_returns": {}, "exposures": {}, "quantile_summary": {},
                "universe_metadata": {}, "coverage": {},
            },
        }
        response = self.client.post(
            "/api/execution-jobs",
            json={"expression": "rank(close)", "universe": "NIFTY50", "start": "2022-01-01", "end": "2023-01-01"},
        )
        job = response.get_json()
        completed = backend_app.execute_execution_job(job["job_id"])
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["result"]["status"], "success")
        self.mock_backtest.assert_called_once_with(job["config"])
        fetched = self.client.get(f"/api/execution-jobs/{job['job_id']}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.get_json()["status"], "succeeded")
        listed = self.client.get("/api/execution-jobs?limit=10")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.get_json()["count"], 1)

    def test_runtime_backtest_failure_is_persisted_as_failed_job(self):
        self.mock_backtest.return_value = {
            "status": "error",
            "error": {"type": "ValidationError", "message": "Expression is invalid."},
        }
        response = self.client.post(
            "/api/execution-jobs", json={"expression": "rank(close)", "universe": "NIFTY50", "start": "2022-01-01", "end": "2023-01-01"}
        )
        job = response.get_json()
        completed = backend_app.execute_execution_job(job["job_id"])
        self.assertEqual(completed["status"], "failed")
        self.assertEqual(completed["error"]["type"], "ValidationError")
        self.assertIsNone(completed["result"])
        self.assertEqual(self.store.get_job(job["job_id"])["status"], "failed")

    def test_unknown_job_returns_not_found(self):
        response = self.client.get("/api/execution-jobs/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"]["type"], "NotFound")

    def test_list_rejects_out_of_range_limit(self):
        for limit in ("0", "101", "not-a-number"):
            with self.subTest(limit=limit):
                response = self.client.get(f"/api/execution-jobs?limit={limit}")
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["error"]["type"], "ValidationError")

