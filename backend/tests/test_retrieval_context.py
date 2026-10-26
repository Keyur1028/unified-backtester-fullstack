"""Offline tests for bounded retrieval context passed to the LLM."""
import json
import unittest

from retrieval_context import (
    MAX_CONTEXT_CHARS,
    format_retrieval_context,
)


class RetrievalContextTests(unittest.TestCase):
    def test_limits_matches_and_candidates(self):
        matches = [
            {
                "similarity": 0.8,
                "ideas": [{"expression": f"rank(close) + {i}"} for i in range(4)],
            }
            for _ in range(4)
        ]
        result = json.loads(format_retrieval_context(matches))
        self.assertLessEqual(len(result), 2)
        self.assertTrue(all(len(row["ideas"]) <= 2 for row in result))

    def test_only_allowlisted_fields_are_serialized(self):
        result = format_retrieval_context([{
            "similarity": 0.9,
            "raw_prompt": "should not be included",
            "ideas": [{
                "name": "Momentum",
                "expression": "rank(close)",
                "hypothesis": "Price trend persists",
                "secret_metadata": "should not be included",
            }],
        }])
        self.assertNotIn("raw_prompt", result)
        self.assertNotIn("secret_metadata", result)
        self.assertNotIn("should not be included", result)
        self.assertEqual(json.loads(result)[0]["ideas"][0]["expression"], "rank(close)")

    def test_long_fields_are_truncated_and_total_context_is_bounded(self):
        huge = "x" * 20000
        result = format_retrieval_context([{
            "similarity": 0.9,
            "ideas": [{
                "name": huge,
                "hypothesis": huge,
                "expression": huge,
                "rationale": huge,
                "expected_behavior": huge,
                "risks": [huge, huge, huge],
            }],
        }])
        self.assertLessEqual(len(result), MAX_CONTEXT_CHARS)
        parsed = json.loads(result)
        if parsed:
            candidate = parsed[0]["ideas"][0]
            self.assertLessEqual(len(candidate["name"]), 100)
            self.assertLessEqual(len(candidate["hypothesis"]), 300)
            self.assertLessEqual(len(candidate["risks"]), 2)

    def test_invalid_or_empty_input_is_empty_json_array(self):
        self.assertEqual(format_retrieval_context(None), "[]")
        self.assertEqual(format_retrieval_context([None, {"ideas": "bad"}]), "[]")


if __name__ == "__main__":
    unittest.main()
