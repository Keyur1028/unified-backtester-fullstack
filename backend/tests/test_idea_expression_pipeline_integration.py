"""Offline end-to-end tests for idea -> candidate -> validation -> cache flow."""
import json
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


class IdeaExpressionPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        api_app.app.config.update(TESTING=True)
        self.client = api_app.app.test_client()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_patch = patch.object(api_app, "STORE", Path(self.temp_dir.name))
        self.store_patch.start()
        self.interface_patch = patch.object(
            api_app.engine,
            "describe_interface",
            return_value={
                "fields": {
                    "native_ohlcv": ["close", "volume"],
                    "engineered": ["momentum_20"],
                    "external": "",
                },
                "universes": {"NIFTY50": "static"},
                "portfolio_types": ["LONG_ONLY", "LONG_SHORT"],
                "weighting_schemes": ["EQUAL_WEIGHT", "SIGNAL_WEIGHTED"],
            },
        )
        self.interface_patch.start()
        self.operators_patch = patch.object(
            api_app.engine, "OPERATORS", {"rank": object(), "ts_mean": object()}
        )
        self.operators_patch.start()
        self.group_ops_patch = patch.object(api_app.engine, "GROUP_OPERATORS", {})
        self.group_ops_patch.start()
        self.runtime_patch = patch.object(
            api_app, "validate_candidate_runtime", return_value=(True, None)
        )
        self.runtime_patch.start()
        self.env_patch = patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"},
        )
        self.env_patch.start()

    def tearDown(self):
        for active_patch in (
            self.env_patch,
            self.runtime_patch,
            self.group_ops_patch,
            self.operators_patch,
            self.interface_patch,
            self.store_patch,
        ):
            active_patch.stop()
        self.temp_dir.cleanup()

    def test_model_candidate_is_validated_persisted_and_reused_without_second_call(self):
        model_response = json.dumps({
            "ideas": [{
                "name": "Persistent momentum",
                "hypothesis": "Cross-sectional price strength persists.",
                "expression": "rank(ts_mean(close, 20))",
                "rationale": "Ranks stocks by their trailing mean price.",
                "expected_behavior": "Higher scores indicate higher recent prices.",
                "risks": ["Price level is not a point-in-time fundamental signal."],
                "required_fields": ["close"],
                "entry_rules": ["Enter the top-ranked available names."],
                "exit_rules": ["Rebalance after the configured holding period."],
                "execution_rules": ["Use the configured daily-bar execution timing."],
                "clarifying_questions": [],
                "assumptions": ["Daily OHLCV data is available."],
            }]
        })

        with patch.object(
            api_app, "generate_candidates", return_value=model_response
        ) as generate:
            first = self.client.post(
                "/api/generate-alpha",
                json={"idea": "Find stocks with persistent price momentum over one month."},
            )

            self.assertEqual(first.status_code, 200, first.get_json())
            first_body = first.get_json()
            self.assertEqual(first_body["ideas"][0]["expression"], "rank(ts_mean(close, 20))")
            handoff = first_body["ideas"][0]["handoff"]
            self.assertEqual(handoff["assistant_status"], "ready_for_notebook2")
            self.assertEqual(handoff["alpha_expression"], "rank(ts_mean(close, 20))")
            self.assertEqual(handoff["required_fields"], ["close"])
            self.assertFalse(handoff["backtest_executed"])
            self.assertTrue(handoff["notebook2_must_revalidate"])
            self.assertEqual(first_body["cache"], {"hit": False, "strategy": "exact-match"})
            self.assertEqual(
                first_body["ideas"][0]["runtime_validation"],
                "synthetic-data-probe-pass",
            )
            self.assertTrue((Path(self.temp_dir.name) / "idea_generation_cache.json").exists())
            self.assertTrue((Path(self.temp_dir.name) / "idea_retrieval_index.json").exists())

            second = self.client.post(
                "/api/generate-alpha",
                json={"idea": "Find stocks with persistent price momentum over one month."},
            )

        self.assertEqual(second.status_code, 200, second.get_json())
        self.assertEqual(second.get_json()["cache"], {"hit": True, "strategy": "exact-match"})
        generate.assert_called_once()

    def test_intraday_context_is_rejected_before_model_call(self):
        with patch.object(api_app, "generate_candidates") as generate:
            response = self.client.post(
                "/api/generate-alpha",
                json={
                    "idea": "Trade the intraday breakout when volume expands and momentum rises.",
                    "cluster": "D0",
                    "universe": "NIFTY50",
                },
            )
        self.assertEqual(response.status_code, 422, response.get_json())
        self.assertEqual(response.get_json()["assistant_status"], "unsupported")
        self.assertFalse(response.get_json()["backtest_executed"])
        generate.assert_not_called()

    def test_invalid_strategy_context_is_rejected_before_model_call(self):
        with patch.object(api_app, "generate_candidates") as generate:
            response = self.client.post(
                "/api/generate-alpha",
                json={
                    "idea": "Find stocks with persistent price momentum over one month.",
                    "cluster": "D1",
                    "universe": "NIFTY50",
                    "strategy_context": {"holding_period_bars": True},
                },
            )
        self.assertEqual(response.status_code, 400, response.get_json())
        self.assertEqual(response.get_json()["error"]["type"], "ValidationError")
        generate.assert_not_called()

    def test_invalid_model_expression_is_rejected_and_not_cached(self):
        model_response = json.dumps({
            "ideas": [{
                "name": "Invalid",
                "expression": "unknown_operator(close)",
                "hypothesis": "Not supported",
            }]
        })
        with patch.object(api_app, "generate_candidates", return_value=model_response):
            response = self.client.post(
                "/api/generate-alpha",
                json={"idea": "Find a persistent momentum signal with price data."},
            )

        self.assertEqual(response.status_code, 422, response.get_json())
        self.assertEqual(
            response.get_json()["error"]["type"], "GenerationValidationError"
        )
        cache_path = Path(self.temp_dir.name) / "idea_generation_cache.json"
        self.assertFalse(cache_path.exists())


if __name__ == "__main__":
    unittest.main()
