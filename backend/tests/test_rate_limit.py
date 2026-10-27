"""Offline tests for the SQLite-backed generation rate limiter."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from rate_limit import check_rate_limit


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "limits.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def test_allows_up_to_limit_then_returns_retry_after(self):
        self.assertEqual(check_rate_limit("client-a", database_path=self.db, limit=2, window_seconds=60, now=121), (True, 0))
        self.assertEqual(check_rate_limit("client-a", database_path=self.db, limit=2, window_seconds=60, now=122), (True, 0))
        allowed, retry = check_rate_limit("client-a", database_path=self.db, limit=2, window_seconds=60, now=123)
        self.assertFalse(allowed)
        self.assertEqual(retry, 57)

    def test_new_window_and_other_client_have_independent_budgets(self):
        check_rate_limit("client-a", database_path=self.db, limit=1, window_seconds=60, now=10)
        self.assertEqual(check_rate_limit("client-b", database_path=self.db, limit=1, window_seconds=60, now=10), (True, 0))
        self.assertEqual(check_rate_limit("client-a", database_path=self.db, limit=1, window_seconds=60, now=60), (True, 0))

    def test_rejects_nonpositive_configuration(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            check_rate_limit("client-a", database_path=self.db, limit=0, window_seconds=60, now=10)
        with self.assertRaisesRegex(ValueError, "positive"):
            check_rate_limit("client-a", database_path=self.db, limit=1, window_seconds=0, now=10)


if __name__ == "__main__":
    unittest.main()
