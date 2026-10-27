"""Unit tests for the isolated alpha generation service."""
import json
import sys
import types
import unittest
from unittest.mock import patch

BACKEND_DIR = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from alpha_generation_service import build_generation_prompt, generate_candidates  # noqa: E402


class AlphaGenerationServiceTests(unittest.TestCase):
    def test_prompt_includes_idea_and_current_catalog(self):
        prompt = build_generation_prompt(
            "Capture persistent price momentum.",
            ["close", "volume"],
            ["rank", "ts_mean"],
        )
        self.assertIn("Capture persistent price momentum.", prompt)
        self.assertIn('"close"', prompt)
        self.assertIn('"ts_mean"', prompt)
        self.assertIn("Return ONLY valid JSON", prompt)
        self.assertIn("do not invent data", prompt)

    def test_prompt_adds_retrieval_matches_as_context_not_authority(self):
        prompt = build_generation_prompt(
            "Capture persistent price momentum.",
            ["close"],
            ["rank"],
            [{"similarity": 0.8, "ideas": [{"expression": "rank(close)"}]}],
        )
        self.assertIn("Retrieved prior candidates", prompt)
        self.assertIn("examples only, not proof of validity", prompt)
        self.assertIn("still obey the active field/operator registry", prompt)
        self.assertIn("\nRetrieved prior candidates", prompt)
        self.assertNotIn("\\\\nRetrieved prior candidates", prompt)

    def test_generate_candidates_uses_json_mode_and_returns_response_text(self):
        calls = []
        class FakeClient:
            def __init__(self, api_key):
                calls.append({"api_key": api_key})
                self.models = self
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(text=json.dumps({"ideas": []}))

        genai_module = types.ModuleType("google.genai")
        genai_module.Client = FakeClient
        google_module = types.ModuleType("google")
        google_module.genai = genai_module
        with patch.dict(sys.modules, {"google": google_module, "google.genai": genai_module}):
            result = generate_candidates("test-key", "test-model", "test prompt")

        self.assertEqual(json.loads(result), {"ideas": []})
        self.assertEqual(calls[0]["api_key"], "test-key")
        self.assertEqual(calls[1]["model"], "test-model")
        self.assertEqual(calls[1]["contents"], "test prompt")
        self.assertEqual(calls[1]["config"]["response_mime_type"], "application/json")
        self.assertEqual(calls[1]["config"]["temperature"], 0.35)


if __name__ == "__main__":
    unittest.main()
