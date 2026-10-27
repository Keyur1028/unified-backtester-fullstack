"""Offline tests for the deterministic alpha runtime preflight."""
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import alpha_engine  # noqa: E402
from candidate_runtime_validation import validate_candidate_runtime  # noqa: E402


class CandidateRuntimeValidationTests(unittest.TestCase):
    def test_valid_expression_runs_on_synthetic_ohlcv(self):
        passed, reason = validate_candidate_runtime(
            alpha_engine, "rank(ts_mean(close, 20))",
            {"open", "high", "low", "close", "volume", "momentum_20"},
        )
        self.assertTrue(passed, reason)

    def test_runtime_invalid_operator_is_rejected(self):
        passed, reason = validate_candidate_runtime(
            alpha_engine, "rank(made_up_operator(close))",
            {"open", "high", "low", "close", "volume"},
        )
        self.assertFalse(passed)
        self.assertTrue(reason)

    def test_expression_with_no_finite_values_is_rejected(self):
        passed, reason = validate_candidate_runtime(
            alpha_engine, "inverse(close - close)",
            {"open", "high", "low", "close", "volume"},
        )
        self.assertFalse(passed)
        self.assertIn("finite values", reason)


if __name__ == "__main__":
    unittest.main()
