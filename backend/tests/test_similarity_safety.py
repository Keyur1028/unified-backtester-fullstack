"""Offline tests for conservative near-duplicate alpha reuse safeguards."""
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
from idea_similarity_safety import has_directional_conflict  # noqa: E402
from similarity_retrieval import idea_tokens  # noqa: E402


class SimilaritySafetyTests(unittest.TestCase):
    def test_opposing_bullish_bearish_terms_conflict(self):
        self.assertTrue(has_directional_conflict(
            idea_tokens("Build a bullish momentum strategy using price trend"),
            idea_tokens("Build a bearish momentum strategy using price trend"),
        ))

    def test_same_direction_does_not_conflict(self):
        self.assertFalse(has_directional_conflict(
            idea_tokens("Find bullish momentum using price trend"),
            idea_tokens("Find bullish momentum using price trend and volume"),
        ))

    def test_unrelated_terms_do_not_conflict(self):
        self.assertFalse(has_directional_conflict(
            idea_tokens("Rank volatility and trading volume"),
            idea_tokens("Rank price momentum and earnings growth"),
        ))

    def test_conflicting_near_duplicate_falls_through_to_gemini(self):
        interface = {"fields": {"native_ohlcv": ["close"], "engineered": [], "external": ""}}
        idea = "Find bullish momentum using price trend and recent trading volume"
        prior_ideas = [{
            "id": "idea-1",
            "expression": "rank(ts_mean(close, 20))",
            "validation": "syntax-and-registry-pass",
        }]
        opposing_tokens = idea_tokens(
            "Find bearish momentum using price trend and recent trading volume"
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(api_app, "STORE", Path(directory)):
                with patch.object(api_app.engine, "describe_interface", return_value=interface):
                    with patch.object(api_app.engine, "OPERATORS", {"rank": object(), "ts_mean": object()}):
                        with patch.object(api_app.engine, "GROUP_OPERATORS", {}):
                            with patch.object(api_app, "retrieve_similar", return_value=[{
                                "similarity": 0.97,
                                "matched_tokens": opposing_tokens,
                                "ideas": prior_ideas,
                            }]):
                                with patch.dict(os.environ, {
                                    "GEMINI_API_KEY": "", "GEMINI_MODEL": "test-model"
                                }):
                                    response = api_app.app.test_client().post(
                                        "/api/generate-alpha", json={"idea": idea}
                                    )
        # No API key means a 503 only when the high-similarity candidate was
        # correctly rejected for reuse and the request falls through to Gemini.
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"]["type"], "ConfigurationError")

    def test_same_direction_near_duplicate_can_still_be_reused(self):
        interface = {"fields": {"native_ohlcv": ["close"], "engineered": [], "external": ""}}
        idea = "Find bullish momentum using price trend and recent trading volume"
        prior_ideas = [{
            "id": "idea-1",
            "expression": "rank(ts_mean(close, 20))",
            "validation": "syntax-and-registry-pass",
        }]
        matching_tokens = idea_tokens(
            "Find bullish momentum using price trend and recent trading volume"
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(api_app, "STORE", Path(directory)):
                with patch.object(api_app.engine, "describe_interface", return_value=interface):
                    with patch.object(api_app.engine, "OPERATORS", {"rank": object(), "ts_mean": object()}):
                        with patch.object(api_app.engine, "GROUP_OPERATORS", {}):
                            with patch.object(api_app, "retrieve_similar", return_value=[{
                                "similarity": 0.97,
                                "matched_tokens": matching_tokens,
                                "ideas": prior_ideas,
                            }]):
                                with patch.dict(os.environ, {
                                    "GEMINI_API_KEY": "", "GEMINI_MODEL": "test-model"
                                }):
                                    response = api_app.app.test_client().post(
                                        "/api/generate-alpha", json={"idea": idea}
                                    )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["cache"], {
            "hit": True, "strategy": "similarity"
        })


if __name__ == "__main__":
    unittest.main()
