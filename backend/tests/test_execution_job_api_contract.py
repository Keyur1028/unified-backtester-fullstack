"""Offline Flask contract tests for execution-job endpoints."""
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
        response = self.client.post(
            "/api/execution-jobs", json=["rank(close)"]
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["type"], "ValidationError")
        self.mock_backtest.assert_not_called()

    def test_successful_run_persists_terminal_result_and_is_retrievable(self):
        self.mock_backtest.return_value = {
            "status": "success",
            "experiment_id": "experiment-test-1",
            "result": {},
        }
        payload = {
            "expression": "rank(close)",
            "universe": "NIFTY50",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
        }

        response = self.client.post("/api/execution-jobs", json=payload)

        self.assertEqual(response.status_code, 200)
        job = response.get_json()
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["config"], payload)
        self.assertEqual(job["result"]["status"], "success")
        self.assertIsNone(job["error"])
        self.mock_backtest.assert_called_once_with(payload)

        fetched = self.client.get(f"/api/execution-jobs/{job['job_id']}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.get_json()["job_id"], job["job_id"])
        self.assertEqual(fetched.get_json()["status"], "succeeded")

        listed = self.client.get("/api/execution-jobs?limit=10")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.get_json()["count"], 1)
        self.assertEqual(listed.get_json()["jobs"][0]["job_id"], job["job_id"])

    def test_backtest_validation_failure_is_persisted_as_failed_job(self):
        self.mock_backtest.return_value = {
            "status": "error",
            "experiment_id": "experiment-test-2",
            "error": {
                "type": "ValidationError",
                "message": "Expression is invalid.",
            },
            "config": {"expression": "invalid("},
        }

        response = self.client.post(
            "/api/execution-jobs", json={"expression": "invalid("}
        )

        self.assertEqual(response.status_code, 200)
        job = response.get_json()
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"]["type"], "ValidationError")
        self.assertIsNone(job["result"])
        persisted = self.store.get_job(job["job_id"])
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["error"]["message"], "Expression is invalid.")

    def test_unknown_job_returns_not_found(self):
        response = self.client.get("/api/execution-jobs/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"]["type"], "NotFound")

    def test_list_rejects_out_of_range_limit(self):
        for limit in ("0", "101", "not-a-number"):
            with self.subTest(limit=limit):
                response = self.client.get(
                    f"/api/execution-jobs?limit={limit}"
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(
                    response.get_json()["error"]["type"], "ValidationError"
                )


if __name__ == "__main__":
    unittest.main()
