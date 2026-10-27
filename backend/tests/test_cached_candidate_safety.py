"""Tests that cached alpha candidates are revalidated before reuse."""
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
from app import valid_cached_ideas  # noqa: E402
from idea_cache import make_cache_key, write_cached_response  # noqa: E402
from similarity_retrieval import idea_tokens  # noqa: E402


class CachedCandidateSafetyTests(unittest.TestCase):
    def setUp(self):
        self.fields = {"close", "volume"}
        self.operators = {"rank", "ts_mean"}

    def test_valid_cached_expression_is_preserved_without_mutating_input(self):
        ideas = [{"id": "idea-1", "expression": " rank(ts_mean(close, 20)) "}]
        result = valid_cached_ideas(ideas, self.fields, self.operators)
        self.assertEqual(result[0]["expression"], "rank(ts_mean(close, 20))")
        self.assertEqual(ideas[0]["expression"], " rank(ts_mean(close, 20)) ")

    def test_unknown_operator_rejects_entire_candidate_set(self):
        ideas = [
            {"id": "idea-1", "expression": "rank(close)"},
            {"id": "idea-2", "expression": "made_up_operator(volume)"},
        ]
        self.assertEqual(valid_cached_ideas(ideas, self.fields, self.operators), [])

    def test_unknown_field_rejects_candidate_set(self):
        self.assertEqual(
            valid_cached_ideas([{"expression": "rank(unavailable_field)"}], self.fields, self.operators),
            [],
        )

    def test_malformed_candidate_rejects_cached_set(self):
        self.assertEqual(valid_cached_ideas([{"expression": "rank("}], self.fields, self.operators), [])

    def test_non_object_candidate_rejects_cached_set(self):
        self.assertEqual(valid_cached_ideas([{"expression": "rank(close)"}, None], self.fields, self.operators), [])

    def test_empty_cache_is_not_reusable(self):
        self.assertEqual(valid_cached_ideas([], self.fields, self.operators), [])


    def test_invalid_exact_cache_falls_through_instead_of_returning_cached_idea(self):
        idea = "Capture persistent momentum with a volatility penalty."
        model = "test-model"
        invalid_response = {
            "ideas": [{"id": "idea-1", "expression": "made_up_operator(close)"}],
            "rejected": [],
            "model": model,
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(api_app, "STORE", Path(directory)):
                with patch.object(api_app.engine, "describe_interface", return_value={
                    "fields": {"native_ohlcv": ["close"], "engineered": [], "external": ""},
                    "universes": {"NIFTY50": "static"},
                    "portfolio_types": ["LONG_ONLY", "LONG_SHORT"],
                    "weighting_schemes": ["EQUAL_WEIGHT", "SIGNAL_WEIGHTED"]
                }):
                    with patch.object(api_app.engine, "OPERATORS", {"rank": object(), "ts_mean": object()}):
                        with patch.object(api_app.engine, "GROUP_OPERATORS", {}):
                            key = make_cache_key(idea, ["close"], ["rank", "ts_mean"], model)
                            write_cached_response(Path(directory) / "idea_generation_cache.json", key, invalid_response)
                            with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_MODEL": model}):
                                response = api_app.app.test_client().post(
                                    "/api/generate-alpha", json={"idea": idea}
                                )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"]["type"], "ConfigurationError")

    def test_invalid_similar_cache_falls_through_instead_of_reusing_candidate(self):
        idea = "Find persistent bullish momentum using price trend and recent trading volume"
        model = "test-model"
        invalid_ideas = [{"id": "idea-1", "expression": "made_up_operator(close)"}]
        tokens = idea_tokens(idea)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(api_app, "STORE", Path(directory)):
                with patch.object(api_app.engine, "describe_interface", return_value={
                    "fields": {"native_ohlcv": ["close"], "engineered": [], "external": ""},
                    "universes": {"NIFTY50": "static"},
                    "portfolio_types": ["LONG_ONLY", "LONG_SHORT"],
                    "weighting_schemes": ["EQUAL_WEIGHT", "SIGNAL_WEIGHTED"]
                }):
                    with patch.object(api_app.engine, "OPERATORS", {"rank": object(), "ts_mean": object()}):
                        with patch.object(api_app.engine, "GROUP_OPERATORS", {}):
                            with patch.object(api_app, "retrieve_similar", return_value=[{
                                "similarity": 0.99,
                                "matched_tokens": tokens,
                                "ideas": invalid_ideas,
                            }]):
                                with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_MODEL": model}):
                                    response = api_app.app.test_client().post(
                                        "/api/generate-alpha", json={"idea": idea}
                                    )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"]["type"], "ConfigurationError")


if __name__ == "__main__":
    unittest.main()
