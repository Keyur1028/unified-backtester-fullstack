"""Offline contract tests for the Flask API.

These tests mock the engine boundary and never call Yahoo Finance or Gemini.
Run from the repository root with:
    python -m unittest discover -s backend/tests -v
"""
import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app as api_app  # noqa: E402
from idea_cache import make_cache_key, write_cached_response  # noqa: E402


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_app.app.config.update(TESTING=True)
        cls.client = api_app.app.test_client()

    def test_health_returns_service_status(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["engine"], "alpha_backtesting_engine_production_hardened")
        self.assertIn("yfinance_available", payload)
        self.assertIn("tracker_count", payload)

    def test_meta_is_derived_from_engine_interface(self):
        interface = {
            "fields": {
                "native_ohlcv": ["close"],
                "engineered": ["momentum_5"],
                "external": "optional external scores",
            },
            "operators": {
                "elementwise_and_timeseries": ["rank"],
                "group_aware": ["group_neutralize"],
            },
            "universes": {"NIFTY50": {"description": "test universe"}},
            "portfolio_types": ["LONG_SHORT"],
            "weighting_schemes": ["EQUAL_WEIGHT"],
            "execution_settings": {"execution_lag_bars": 1},
            "expression_syntax": "registered functions and fields",
        }
        with patch.object(api_app.engine, "describe_interface", return_value=interface):
            response = self.client.get("/api/meta")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["universes"], ["NIFTY50"])
        self.assertEqual(payload["fields"]["native"], ["close"])
        self.assertEqual(payload["fields"]["engineered"], ["momentum_5"])
        self.assertEqual(payload["operators"], ["rank", "group_neutralize"])
        self.assertEqual(payload["portfolio_types"], ["LONG_SHORT"])

    def test_examples_returns_expression_records(self):
        response = self.client.get("/api/examples")
        self.assertEqual(response.status_code, 200)
        examples = response.get_json()
        self.assertIsInstance(examples, list)
        self.assertTrue(examples)
        self.assertTrue(all({"name", "expression", "category"} <= set(row) for row in examples))

    def test_experiments_returns_newest_records_first(self):
        rows = [
            {"experiment_id": "old", "status": "success", "expression": "rank(close)", "at": "t1"},
            {"experiment_id": "new", "status": "success", "expression": "rank(volume)", "at": "t2"},
        ]
        with patch.object(api_app.engine.TRACKER, "to_dict", return_value=rows):
            response = self.client.get("/api/experiments")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual([row["experiment_id"] for row in payload], ["new", "old"])

    def test_backtest_rejects_non_json_content_type(self):
        response = self.client.post("/api/backtest", data="{}")
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.get_json()["error"]["type"], "ValidationError")

    def test_backtest_rejects_non_object_json(self):
        response = self.client.post("/api/backtest", json=["not", "an", "object"])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["type"], "ValidationError")

    def test_backtest_surfaces_engine_validation_errors_without_running_data(self):
        envelope = {
            "status": "error",
            "experiment_id": "test-experiment",
            "error": {"type": "ValidationError", "message": "Unsupported expression"},
            "config": {"expression": "not_supported(close)"},
        }
        with patch.object(api_app.engine, "run_backtest_api", return_value=envelope):
            with patch.object(api_app.engine.TRACKER, "save"):
                response = self.client.post(
                    "/api/backtest",
                    json={"expression": "not_supported(close)"},
                )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["message"], "Unsupported expression")

    def test_generate_alpha_reports_missing_gemini_configuration(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            response = self.client.post(
                "/api/generate-alpha",
                json={"idea": "Capture persistent momentum with a volatility penalty."},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"]["type"], "ConfigurationError")


    def test_generate_alpha_high_similarity_hit_skips_gemini_key(self):
        interface = {"fields": {"native_ohlcv": ["close"], "engineered": [], "external": ""}}
        idea = "Find persistent momentum using price trend and recent trading volume"
        prior_ideas = [{
            "id": "idea-1",
            "expression": "rank(ts_mean(close, 20))",
            "validation": "syntax-and-registry-pass",
            "provenance": {"scope": "static-syntax-and-registry-only"},
        }]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(api_app, "STORE", Path(directory)):
                with patch.object(api_app.engine, "describe_interface", return_value=interface):
                    with patch.object(api_app.engine, "OPERATORS", {"rank": object()}):
                        with patch.object(api_app.engine, "GROUP_OPERATORS", {}):
                            with patch.object(api_app, "retrieve_similar", return_value=[
                                {"similarity": 0.96, "ideas": prior_ideas}
                            ]):
                                with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_MODEL": "test-model"}):
                                    response = self.client.post("/api/generate-alpha", json={"idea": idea})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["ideas"][0]["expression"], "rank(ts_mean(close, 20))")
        self.assertEqual(payload["cache"], {"hit": True, "strategy": "similarity"})
        self.assertEqual(payload["retrieval"]["similarity"], 0.96)
        self.assertIn("lexical", payload["note"])

    def test_generate_alpha_cache_hit_does_not_require_gemini_key(self):
        interface = {"fields": {"native_ohlcv": ["close"], "engineered": [], "external": ""}}
        idea = "Capture persistent momentum with a volatility penalty."
        model = "test-model"
        response_body = {"ideas": [{"expression": "rank(close)", "id": "idea-1"}], "rejected": [], "model": model}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(api_app, "STORE", Path(directory)):
                with patch.object(api_app.engine, "describe_interface", return_value=interface):
                    with patch.object(api_app.engine, "OPERATORS", {"rank": object()}):
                        with patch.object(api_app.engine, "GROUP_OPERATORS", {}):
                            key = make_cache_key(idea, ["close"], ["rank"], model)
                            write_cached_response(Path(directory) / "idea_generation_cache.json", key, response_body)
                            with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_MODEL": model}):
                                response = self.client.post("/api/generate-alpha", json={"idea": idea})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["ideas"][0]["expression"], "rank(close)")
        self.assertEqual(payload["cache"], {"hit": True, "strategy": "exact-match"})

if __name__ == "__main__":
    unittest.main()
