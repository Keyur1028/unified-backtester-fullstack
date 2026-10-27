"""Tests for bounded model-generated alpha candidate payloads."""
import json
import os
import sys
import types
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app as api_app  # noqa: E402
from candidate_payload import compact_generated_candidate  # noqa: E402


class CandidatePayloadTests(unittest.TestCase):
    def test_bounds_candidate_text_and_risks_and_drops_unknown_fields(self):
        candidate, error = compact_generated_candidate({
            "expression": "rank(close)",
            "name": "n" * 150,
            "hypothesis": "h" * 400,
            "rationale": "r" * 300,
            "expected_behavior": "e" * 250,
            "risks": ["x" * 200, "second", "third", "fourth"],
            "provider_debug": "do not return",
        })
        self.assertIsNone(error)
        self.assertEqual(candidate["expression"], "rank(close)")
        self.assertEqual(len(candidate["name"]), 100)
        self.assertEqual(len(candidate["hypothesis"]), 300)
        self.assertEqual(len(candidate["rationale"]), 250)
        self.assertEqual(len(candidate["expected_behavior"]), 200)
        self.assertEqual(len(candidate["risks"]), 3)
        self.assertEqual(len(candidate["risks"][0]), 160)
        self.assertNotIn("provider_debug", candidate)

    def test_oversized_expression_is_rejected_not_truncated(self):
        candidate, error = compact_generated_candidate({"expression": "x" * 1001})
        self.assertIsNone(candidate)
        self.assertIn("1000-character limit", error)

    def test_missing_or_non_string_expression_is_rejected(self):
        for item in ({}, {"expression": 123}, None):
            with self.subTest(item=item):
                candidate, error = compact_generated_candidate(item)
                self.assertIsNone(candidate)
                self.assertTrue(error)

    def test_endpoint_returns_bounded_allowlisted_model_fields(self):
        model_candidate = {
            "name": "n" * 150,
            "hypothesis": "h" * 400,
            "expression": "rank(close)",
            "rationale": "r" * 300,
            "expected_behavior": "e" * 250,
            "risks": ["risk" * 100] * 5,
            "internal_debug": "should not escape",
        }
        response_module = types.ModuleType("google.genai")
        class FakeClient:
            def __init__(self, api_key):
                self.models = self
            def generate_content(self, **kwargs):
                return types.SimpleNamespace(text=json.dumps({"ideas": [model_candidate]}))
        response_module.Client = FakeClient
        google_module = types.ModuleType("google")
        google_module.genai = response_module
        interface = {"fields": {"native_ohlcv": ["close"], "engineered": [], "external": ""}}

        with patch.object(api_app.engine, "describe_interface", return_value=interface):
            with patch.object(api_app.engine, "OPERATORS", {"rank": object()}):
                with patch.object(api_app.engine, "GROUP_OPERATORS", {}):
                    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"}):
                        with patch.dict(sys.modules, {"google": google_module, "google.genai": response_module}):
                            response = api_app.app.test_client().post(
                                "/api/generate-alpha",
                                json={"idea": "Capture a persistent price trend with a cross-sectional rank."},
                            )

        self.assertEqual(response.status_code, 200, response.get_json())
        candidate = response.get_json()["ideas"][0]
        self.assertEqual(len(candidate["name"]), 100)
        self.assertEqual(len(candidate["hypothesis"]), 300)
        self.assertEqual(len(candidate["rationale"]), 250)
        self.assertEqual(len(candidate["expected_behavior"]), 200)
        self.assertEqual(len(candidate["risks"]), 3)
        self.assertNotIn("internal_debug", candidate)
        self.assertEqual(candidate["expression"], "rank(close)")


if __name__ == "__main__":
    unittest.main()
