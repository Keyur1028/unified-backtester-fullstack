"""Offline end-to-end contract test for the notebook-derived research engine.

This exercises the real engine calculation path with deterministic synthetic
market data. It deliberately injects data and universe metadata so CI never
calls Yahoo Finance or depends on local universe CSV files.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import alpha_engine  # noqa: E402


class NotebookEngineIntegrationTests(unittest.TestCase):
    def setUp(self):
        alpha_engine.clear_expression_cache()
        self.dates = pd.bdate_range("2020-01-01", periods=520)
        self.tickers = [f"SYNTH{i:02d}" for i in range(12)]
        rng = np.random.default_rng(20261027)
        daily_returns = rng.normal(
            loc=np.linspace(-0.0002, 0.0005, len(self.tickers)),
            scale=0.01,
            size=(len(self.dates), len(self.tickers)),
        )
        close_values = 100 * np.exp(np.cumsum(daily_returns, axis=0))
        close = pd.DataFrame(close_values, index=self.dates, columns=self.tickers)
        open_ = close * (1 + rng.normal(0, 0.001, size=close.shape))
        high = pd.DataFrame(
            np.maximum(open_.to_numpy(), close.to_numpy()) * 1.005,
            index=self.dates, columns=self.tickers,
        )
        low = pd.DataFrame(
            np.minimum(open_.to_numpy(), close.to_numpy()) * 0.995,
            index=self.dates, columns=self.tickers,
        )
        volume = pd.DataFrame(
            rng.lognormal(mean=12, sigma=0.25, size=close.shape),
            index=self.dates, columns=self.tickers,
        )
        self.ohlcv = {
            "open": open_, "high": high, "low": low,
            "close": close, "volume": volume,
        }
        self.fields = dict(self.ohlcv)
        self.mask = pd.DataFrame(True, index=self.dates, columns=self.tickers)
        self.universe_meta = {
            "point_in_time": False,
            "survivorship_bias_risk": True,
            "notes": "Deterministic synthetic test universe.",
        }
        self.benchmark = close.mean(axis=1)

    def tearDown(self):
        alpha_engine.clear_expression_cache()

    def test_public_engine_boundary_runs_real_calculation_offline(self):
        response = alpha_engine.run_backtest_api(
            {
                "expression": "rank(close)",
                "universe": "NIFTY50",
                "start": "2020-07-01",
                "end": "2021-12-31",
                "holding_period": 3,
                "execution_lag_bars": 1,
                "execution_price": "close",
            },
            _ohlcv=self.ohlcv,
            _fields=self.fields,
            _universe_mask_and_meta=(self.mask, self.universe_meta),
            _benchmark_price=self.benchmark,
        )

        self.assertEqual(response["status"], "success", response.get("error"))
        self.assertIsNone(response["error"])
        self.assertEqual(response["config"]["expression"], "rank(close)")
        result = response["result"]
        self.assertEqual(result["expression"], "rank(close)")
        self.assertTrue(result["net_returns"])
        self.assertTrue(result["gross_returns"])
        self.assertTrue(result["cum_net_returns"])
        self.assertIn("sharpe_net", result)
        self.assertIn("annual_return_net", result)
        self.assertIn("coverage", result)
        self.assertIsInstance(response["experiment_id"], str)

    def test_public_engine_boundary_reports_runtime_errors_in_envelope(self):
        response = alpha_engine.run_backtest_api(
            {
                "expression": "rank(close)",
                "universe": "NIFTY50",
                "start": "2020-07-01",
                "end": "2021-12-31",
                "holding_period": 3,
            },
            _ohlcv=self.ohlcv,
            _fields=self.fields,
            _universe_mask_and_meta=(self.mask, self.universe_meta),
            _benchmark_price=self.benchmark,
        )
        self.assertEqual(response["status"], "success", response.get("error"))

        # A valid request with a missing field must fail through the same public
        # engine boundary rather than escaping as an unstructured exception.
        invalid = alpha_engine.run_backtest_api(
            {
                "expression": "rank(field_that_does_not_exist)",
                "universe": "NIFTY50",
                "start": "2020-07-01",
                "end": "2021-12-31",
            },
            _ohlcv=self.ohlcv,
            _fields=self.fields,
            _universe_mask_and_meta=(self.mask, self.universe_meta),
            _benchmark_price=self.benchmark,
        )
        self.assertEqual(invalid["status"], "error")
        self.assertIsNotNone(invalid["error"])
        self.assertIsNone(invalid["result"])


if __name__ == "__main__":
    unittest.main()
