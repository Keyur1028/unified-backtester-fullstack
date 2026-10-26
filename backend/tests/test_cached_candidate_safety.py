"""Tests that cached alpha candidates are revalidated before reuse."""
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app import valid_cached_ideas  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
