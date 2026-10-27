"""Flask-level contract tests for the expensive generation endpoint limits."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app as backend_app  # noqa: E402


class GenerationRateLimitApiTests(unittest.TestCase):
    def test_per_client_limit_returns_429_and_retry_after(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(backend_app, "STORE", Path(directory)), patch.dict(
                os.environ,
                {
                    "ALPHA_GENERATION_RATE_LIMIT": "1",
                    "ALPHA_GENERATION_RATE_WINDOW_SECONDS": "60",
                },
            ):
                client = backend_app.app.test_client()
                first = client.post("/api/generate-alpha", data="not-json", content_type="application/json")
                second = client.post("/api/generate-alpha", data="not-json", content_type="application/json")

        self.assertEqual(first.status_code, 400)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.json["error"]["type"], "RateLimitExceeded")
        self.assertGreaterEqual(int(second.headers["Retry-After"]), 1)
        self.assertEqual(
            second.json["error"]["retry_after_seconds"],
            int(second.headers["Retry-After"]),
        )


if __name__ == "__main__":
    unittest.main()
