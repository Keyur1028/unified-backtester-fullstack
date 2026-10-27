"""Opt-in PostgreSQL integration tests for the shared job-store contract.

Set TEST_POSTGRESQL_URL to a disposable PostgreSQL database to run these tests.
They create uniquely named rows in the execution_jobs table and never truncate it.
"""
import os
import sys
import unittest
import uuid

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from execution_job_contract import ExecutionJobStatus, RunConfiguration  # noqa: E402
from execution_job_store import ExecutionJobNotFoundError, InvalidJobTransitionError  # noqa: E402
from postgres_execution_job_store import PostgresExecutionJobStore  # noqa: E402


@unittest.skipUnless(
    os.getenv("TEST_POSTGRESQL_URL"),
    "Set TEST_POSTGRESQL_URL to run PostgreSQL integration tests.",
)
class PostgresExecutionJobStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = PostgresExecutionJobStore(os.environ["TEST_POSTGRESQL_URL"])

    def setUp(self):
        self.job_id = "test-" + uuid.uuid4().hex
        self.configuration = RunConfiguration.from_payload({
            "expression": "rank(close)",
            "universe": "NIFTY50",
        })
        self.store.create_job(self.configuration, job_id=self.job_id)

    def test_job_lifecycle_round_trips_configuration_and_result(self):
        created = self.store.get_job(self.job_id)
        self.assertEqual(created["status"], "queued")
        self.assertEqual(created["config"], self.configuration.to_dict())
        self.store.transition(self.job_id, ExecutionJobStatus.RUNNING)
        done = self.store.transition(
            self.job_id, ExecutionJobStatus.SUCCEEDED,
            result={"summary": {"sharpe": 1.2}},
        )
        self.assertEqual(done["result"], {"summary": {"sharpe": 1.2}})
        self.assertEqual(self.store.get_job(self.job_id), done)

    def test_terminal_transition_is_rejected(self):
        self.store.transition(self.job_id, ExecutionJobStatus.CANCELLED)
        with self.assertRaises(InvalidJobTransitionError):
            self.store.transition(self.job_id, ExecutionJobStatus.RUNNING)

    def test_missing_job_raises_not_found(self):
        with self.assertRaises(ExecutionJobNotFoundError):
            self.store.get_job("missing-" + uuid.uuid4().hex)


if __name__ == "__main__":
    unittest.main()
