"""Tests for strict provider response parsing."""
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from model_response_contract import ModelResponseContractError, parse_idea_candidates


class ModelResponseContractTests(unittest.TestCase):
    def test_accepts_candidate_object_array(self):
        result = parse_idea_candidates('{"ideas":[{"expression":"rank(close)"}]}')
        self.assertEqual(result, [{"expression": "rank(close)"}])

    def test_accepts_empty_array_for_downstream_validation(self):
        self.assertEqual(parse_idea_candidates('{"ideas":[]}'), [])

    def test_rejects_invalid_json_and_wrong_top_level_shape(self):
        for response in ("not json", "[]", '"string"', '{"results":[]}'):
            with self.subTest(response=response):
                with self.assertRaises(ModelResponseContractError):
                    parse_idea_candidates(response)

    def test_rejects_non_object_items_and_too_many_candidates(self):
        for response in ('{"ideas":[null]}', '{"ideas":' + "[" + ",".join('{"x":1}' for _ in range(6)) + "]}"):
            with self.subTest(response=response):
                with self.assertRaises(ModelResponseContractError):
                    parse_idea_candidates(response)

    def test_rejects_empty_and_oversized_response(self):
        with self.assertRaisesRegex(ModelResponseContractError, "empty"):
            parse_idea_candidates(" ")
        with self.assertRaisesRegex(ModelResponseContractError, "size limit"):
            parse_idea_candidates("x" * 128001)


if __name__ == "__main__":
    unittest.main()
