"""Cross-boundary offline integration: generated idea -> real engine -> persisted job.

The model response is mocked, but candidate validation, request validation,
the notebook-derived calculation engine, result adaptation, and job persistence
remain real. No live Gemini or market-data requests are made.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app as api_app  # noqa: E402
from execution_job_store import SQLiteExecutionJobStore  # noqa: E402


class IdeaToBacktestFlowTests(unittest.TestCase):
    def setUp(self):
        api_app.app.config.update(TESTING=True)
        self.client = api_app.app.test_client()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store_patch = patch.object(api_app, "STORE", self.root)
        self.store_patch.start()
        self.job_store_patch = patch.object(
            api_app, "JOB_STORE", SQLiteExecutionJobStore(self.root / "jobs.sqlite3")
        )
        self.job_store_patch.start()
        self.env_patch = patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "offline-test-key", "GEMINI_MODEL": "test-model"},
        )
        self.env_patch.start()
        self.runtime_patch = patch.object(
            api_app, "validate_candidate_runtime", return_value=(True, None)
        )
        self.runtime_patch.start()

        dates = pd.bdate_range("2020-01-01", periods=520)
        tickers = [f"SYNTH{i:02d}" for i in range(12)]
        rng = np.random.default_rng(20261027)
        returns = rng.normal(0.0002, 0.01, size=(len(dates), len(tickers)))
        close = pd.DataFrame(
            100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers
        )
        open_ = close * (1 + rng.normal(0, 0.001, size=close.shape))
        self.ohlcv = {
            "open": open_,
            "high": pd.DataFrame(
                np.maximum(open_.to_numpy(), close.to_numpy()) * 1.005,
                index=dates, columns=tickers,
            ),
            "low": pd.DataFrame(
                np.minimum(open_.to_numpy(), close.to_numpy()) * 0.995,
                index=dates, columns=tickers,
            ),
            "close": close,
            "volume": pd.DataFrame(
                rng.lognormal(12, 0.25, size=close.shape), index=dates, columns=tickers
            ),
        }
        self.mask = pd.DataFrame(True, index=dates, columns=tickers)
        self.universe_meta = {
            "point_in_time": False,
            "survivorship_bias_risk": True,
            "notes": "Synthetic integration-test universe.",
        }

    def tearDown(self):
        for active_patch in (
            self.runtime_patch, self.env_patch, self.job_store_patch, self.store_patch
        ):
            active_patch.stop()
        self.temp_dir.cleanup()

    def test_generated_expression_executes_and_persists_real_backtest_result(self):
        generated = json.dumps({
            "ideas": [{
                "name": "Cross-sectional price rank",
                "hypothesis": "Stocks with stronger price levels rank higher.",
                "expression": "rank(close)",
                "rationale": "Rank the current close across the available universe.",
                "expected_behavior": "Higher price ranks receive higher signal scores.",
                "risks": ["Price level is not a scale-neutral fundamental signal."],
            }]
        })

        with patch.object(api_app, "generate_candidates", return_value=generated) as model:
            with patch.object(api_app.engine.TRACKER, "save"):
                idea_response = self.client.post(
                    "/api/generate-alpha",
                    json={"idea": "Rank stocks by their cross-sectional price strength."},
                )
                self.assertEqual(idea_response.status_code, 200, idea_response.get_json())
                expression = idea_response.get_json()["ideas"][0]["expression"]
                self.assertEqual(expression, "rank(close)")

                def run_with_synthetic_market_data(payload):
                    return api_app.engine.run_backtest_api(
                        payload,
                        _ohlcv=self.ohlcv,
                        _fields=dict(self.ohlcv),
                        _universe_mask_and_meta=(self.mask, self.universe_meta),
                        _benchmark_price=self.ohlcv["close"].mean(axis=1),
                    )

                with patch.object(api_app, "run_backtest", side_effect=run_with_synthetic_market_data):
                    queued_response = self.client.post(
                        "/api/execution-jobs",
                        json={
                            "expression": expression,
                            "universe": "NIFTY50",
                            "start": "2020-07-01",
                            "end": "2021-06-30",
                            "holding_period": 3,
                            "execution_lag_bars": 1,
                            "execution_price": "close",
                        },
                    )
                    self.assertEqual(queued_response.status_code, 202, queued_response.get_json())
                    queued = queued_response.get_json()
                    completed = api_app.execute_execution_job(queued["job_id"])

        model.assert_called_once()
        self.assertEqual(completed["status"], "succeeded", completed.get("error"))
        self.assertEqual(completed["config"]["expression"], expression)
        self.assertEqual(completed["result"]["status"], "success")
        self.assertEqual(completed["result"]["config"]["expression"], expression)
        self.assertTrue(completed["result"]["series"]["dates"])
        self.assertIn("sharpe_net", completed["result"]["summary"])

        fetched = self.client.get(f"/api/execution-jobs/{queued['job_id']}")
        self.assertEqual(fetched.status_code, 200, fetched.get_json())
        self.assertEqual(fetched.get_json(), completed)


if __name__ == "__main__":
    unittest.main()
