"""Offline API tests for persisted background execution-job endpoints."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app as api_app  # noqa: E402
from execution_job_contract import RunConfiguration  # noqa: E402
from execution_job_store import SQLiteExecutionJobStore  # noqa: E402


class ExecutionJobApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SQLiteExecutionJobStore(
            Path(self.temp_dir.name) / "execution-jobs.sqlite3"
        )
        self.store_patch = patch.object(api_app, "JOB_STORE", self.store)
        self.store_patch.start()
        api_app.app.config.update(TESTING=True)
        self.client = api_app.app.test_client()

    def tearDown(self):
        self.store_patch.stop()
        self.temp_dir.cleanup()

    def test_successful_execution_persists_result_and_can_be_fetched(self):
        envelope = {
            "status": "success",
            "experiment_id": "exp-test",
            "result": {
                "expression": "rank(close)",
                "net_returns": {},
                "gross_returns": {},
                "ic_series": {},
                "drawdown_series": {},
                "turnover_series": {},
                "cum_net_returns": {},
                "cum_gross_returns": {},
            },
        }
        with patch.object(api_app, "run_backtest", return_value=envelope):
            with patch.object(api_app.engine.TRACKER, "save"):
                response = self.client.post(
                    "/api/execution-jobs",
                    json={"expression": "rank(close)", "universe": "NIFTY50"},
                )
                self.assertEqual(response.status_code, 202)
                queued = response.get_json()
                self.assertEqual(queued["status"], "queued")
                completed = api_app.execute_execution_job(queued["job_id"])
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["config"]["expression"], "rank(close)")
        self.assertEqual(completed["result"]["experiment_id"], "exp-test")
        self.assertIsNone(completed["error"])

        fetched = self.client.get(f"/api/execution-jobs/{queued['job_id']}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.get_json(), completed)

    def test_engine_error_is_persisted_as_failed_job(self):
        envelope = {
            "status": "error",
            "experiment_id": "exp-invalid",
            "error": {"type": "ValidationError", "message": "Unsupported expression"},
            "config": {"expression": "unsupported(close)"},
        }
        with patch.object(api_app, "run_backtest", return_value=envelope):
            with patch.object(api_app.engine.TRACKER, "save"):
                response = self.client.post(
                    "/api/execution-jobs", json={"expression": "unsupported(close)"}
                )
                self.assertEqual(response.status_code, 202)
                queued = response.get_json()
                completed = api_app.execute_execution_job(queued["job_id"])
        self.assertEqual(completed["status"], "failed")
        self.assertEqual(completed["error"]["type"], "ValidationError")
        self.assertIsNone(completed["result"])

    def test_invalid_request_is_rejected_before_job_creation(self):
        response = self.client.post("/api/execution-jobs", json=["not", "an", "object"])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["type"], "ValidationError")

    def test_non_finite_configuration_is_rejected_before_job_creation(self):
        response = self.client.post(
            "/api/execution-jobs",
            data='{"expression":"rank(close)","value":NaN}',
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["type"], "ValidationError")
        self.assertEqual(self.store.list_jobs(), [])

    def test_list_jobs_returns_recent_job_records(self):
        first = self.store.create_job(
            RunConfiguration.from_payload({"expression": "rank(close)"}),
            job_id="job-first",
        )
        response = self.client.get("/api/execution-jobs")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["limit"], 20)
        self.assertEqual(body["jobs"][0]["job_id"], first["job_id"])

    def test_list_jobs_rejects_invalid_limit(self):
        for limit in ("0", "101", "abc"):
            with self.subTest(limit=limit):
                response = self.client.get(f"/api/execution-jobs?limit={limit}")
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["error"]["type"], "ValidationError")

    def test_queued_job_can_be_cancelled_through_api(self):
        queued = self.store.create_job(
            RunConfiguration.from_payload({"expression": "rank(close)"}),
            job_id="cancel-queued",
        )
        response = self.client.post(f"/api/execution-jobs/{queued['job_id']}/cancel")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "cancelled")
        self.assertEqual(self.store.get_job(queued["job_id"])["status"], "cancelled")

    def test_running_job_cannot_be_cancelled_through_api(self):
        queued = self.store.create_job(
            RunConfiguration.from_payload({"expression": "rank(close)"}),
            job_id="cancel-running",
        )
        self.store.transition(queued["job_id"], "running")
        response = self.client.post(f"/api/execution-jobs/{queued['job_id']}/cancel")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["type"], "InvalidJobState")
        self.assertEqual(self.store.get_job(queued["job_id"])["status"], "running")

    def test_cancelling_unknown_job_returns_not_found(self):
        response = self.client.post("/api/execution-jobs/missing/cancel")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"]["type"], "NotFound")

    def test_missing_job_returns_not_found(self):
        response = self.client.get("/api/execution-jobs/not-a-job")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"]["type"], "NotFound")


if __name__ == "__main__":
    unittest.main()
