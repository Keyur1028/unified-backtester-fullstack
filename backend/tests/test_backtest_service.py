"""Unit tests for the backtest service boundary."""
import os
import sys
import unittest
from unittest.mock import Mock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import backtest_service  # noqa: E402


class BacktestServiceTests(unittest.TestCase):
    def test_delegates_payload_to_authoritative_engine(self):
        payload = {"expression": "rank(close)", "universe": "NIFTY50"}
        expected = {"status": "error", "error": {"type": "ValidationError"}}
        engine = Mock()
        engine.run_backtest_api.return_value = expected

        with patch.object(backtest_service, "engine", engine):
            result = backtest_service.run_backtest(payload)

        engine.run_backtest_api.assert_called_once_with(payload)
        self.assertIs(result, expected)

    def test_does_not_rewrite_success_result(self):
        payload = {"expression": "rank(close)"}
        expected = {
            "status": "success",
            "experiment_id": "exp-test",
            "result": {"annual_return_net": 0.12, "expression": "rank(close)"},
        }
        engine = Mock()
        engine.run_backtest_api.return_value = expected

        with patch.object(backtest_service, "engine", engine):
            result = backtest_service.run_backtest(payload)

        self.assertEqual(result, expected)
        self.assertEqual(result["result"]["annual_return_net"], 0.12)


if __name__ == "__main__":
    unittest.main()
