"""Offline tests for local-first retrieval summarization."""
import os
import sys
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import local_summarizer  # noqa: E402


class LocalSummarizerTests(unittest.TestCase):
    def test_deterministic_fallback_normalizes_whitespace_without_network(self):
        with patch.dict(os.environ, {"LOCAL_SUMMARIZER_MODEL": ""}):
            result = local_summarizer.summarize_for_retrieval("  rank(close)\\n\\n momentum  ")
        self.assertEqual(result["mode"], "deterministic-normalization")
        self.assertEqual(result["summary"], "rank(close) momentum")

    def test_configured_local_model_is_used_when_available(self):
        class FakeTokenizer:
            def __call__(self, *args, **kwargs):
                return {"input_ids": "fake"}
            def decode(self, output, skip_special_tokens=True):
                return "momentum using close and volume"

        class FakeModel:
            def eval(self):
                return None
            def generate(self, **kwargs):
                return ["summary-output"]

        with patch.dict(os.environ, {"LOCAL_SUMMARIZER_MODEL": "local-test-model"}):
            with patch.object(local_summarizer, "_load_local_model", return_value=(FakeTokenizer(), FakeModel())):
                result = local_summarizer.summarize_for_retrieval("Long idea with many details.")
        self.assertEqual(result["mode"], "local-pretrained")
        self.assertIn("momentum", result["summary"])


if __name__ == "__main__":
    unittest.main()
