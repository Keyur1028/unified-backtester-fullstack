"""Offline tests for execution-job storage backend selection."""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import execution_job_store_factory as factory  # noqa: E402
from execution_job_store import SQLiteExecutionJobStore  # noqa: E402


class ExecutionJobStoreFactoryTests(unittest.TestCase):
    def test_sqlite_is_default_and_uses_configured_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "jobs.sqlite3")
            store = factory.create_execution_job_store("", path)
            self.assertIsInstance(store, SQLiteExecutionJobStore)
            self.assertEqual(store.database_path, path)

    def test_rejects_non_postgres_database_url(self):
        with self.assertRaisesRegex(ValueError, "PostgreSQL connection URL"):
            factory.create_execution_job_store("mysql://example/db", ":memory:")

    def test_postgres_url_selects_postgres_store_without_sqlite_fallback(self):
        with patch(
            "postgres_execution_job_store.PostgresExecutionJobStore"
        ) as postgres_store:
            postgres_store.return_value = object()
            selected = factory.create_execution_job_store(
                "postgresql://user:pass@localhost/jobs", ":memory:"
            )
        self.assertIs(selected, postgres_store.return_value)
        postgres_store.assert_called_once_with(
            "postgresql://user:pass@localhost/jobs"
        )


if __name__ == "__main__":
    unittest.main()
