"""Regression tests for execution-job owner isolation on authenticated routes."""
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
from execution_job_store import SQLiteExecutionJobStore  # noqa: E402


class ExecutionJobOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SQLiteExecutionJobStore(Path(self.temp_dir.name) / "jobs.sqlite3")
        self.store_patch = patch.object(api_app, "JOB_STORE", self.store)
        self.store_patch.start()
        self.env_patch = patch.dict(os.environ, {
            "SUPABASE_AUTH_REQUIRED": "true",
            "SUPABASE_URL": "https://example.supabase.co",
        })
        self.env_patch.start()
        self.verify_patch = patch(
            "authentication.verify_supabase_access_token",
            side_effect=lambda token: {"sub": "user-a" if token == "token-a" else "user-b"},
        )
        self.verify_patch.start()
        api_app.app.config.update(TESTING=True)
        self.client = api_app.app.test_client()

    def tearDown(self):
        self.verify_patch.stop()
        self.env_patch.stop()
        self.store_patch.stop()
        self.temp_dir.cleanup()

    def _create_job(self, token):
        return self.client.post(
            "/api/execution-jobs",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "expression": "rank(close)",
                "universe": "NIFTY50",
                "start": "2022-01-01",
                "end": "2023-01-01",
            },
        )

    def test_users_only_list_and_fetch_their_own_jobs(self):
        first = self._create_job("token-a")
        second = self._create_job("token-b")
        self.assertEqual(first.status_code, 202, first.get_json())
        self.assertEqual(second.status_code, 202, second.get_json())
        first_id = first.get_json()["job_id"]
        second_id = second.get_json()["job_id"]

        user_a_headers = {"Authorization": "Bearer token-a"}
        user_b_headers = {"Authorization": "Bearer token-b"}
        self.assertEqual(
            self.client.get("/api/execution-jobs", headers=user_a_headers).get_json()["count"], 1
        )
        self.assertEqual(
            self.client.get("/api/execution-jobs", headers=user_b_headers).get_json()["count"], 1
        )
        self.assertEqual(
            self.client.get(f"/api/execution-jobs/{first_id}", headers=user_b_headers).status_code, 404
        )
        self.assertEqual(
            self.client.get(f"/api/execution-jobs/{second_id}", headers=user_a_headers).status_code, 404
        )

    def test_other_user_cannot_cancel_a_job(self):
        created = self._create_job("token-a")
        self.assertEqual(created.status_code, 202, created.get_json())
        job_id = created.get_json()["job_id"]
        response = self.client.post(
            f"/api/execution-jobs/{job_id}/cancel",
            headers={"Authorization": "Bearer token-b"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.store.get_job(job_id)["status"], "queued")


if __name__ == "__main__":
    unittest.main()
