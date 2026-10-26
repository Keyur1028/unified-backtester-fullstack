"""Offline tests for the local alpha idea retrieval index."""
import json
import tempfile
import unittest
from pathlib import Path

from similarity_retrieval import (
    MAX_INDEX_FILE_BYTES,
    MAX_TOKENS_PER_IDEA,
    cosine_similarity,
    idea_tokens,
    retrieve_similar,
    store_idea_response,
)


class SimilarityRetrievalTests(unittest.TestCase):
    def test_tokens_normalize_and_drop_common_words(self):
        self.assertEqual(idea_tokens("Capture MOMENTUM with high volume"), ["momentum", "high", "volume"])

    def test_token_count_is_bounded_for_long_ideas(self):
        idea = " ".join(f"token{index}" for index in range(MAX_TOKENS_PER_IDEA + 50))
        self.assertEqual(len(idea_tokens(idea)), MAX_TOKENS_PER_IDEA)

    def test_cosine_similarity_is_bounded_and_order_independent(self):
        self.assertAlmostEqual(cosine_similarity(["momentum", "volume"], ["volume", "momentum"]), 1.0)
        self.assertGreater(cosine_similarity(["momentum", "volume"], ["momentum", "price"]), 0.0)
        self.assertEqual(cosine_similarity([], ["momentum"]), 0.0)

    def test_retrieves_compatible_prior_ideas_by_similarity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            response = {"ideas": [{"expression": "rank(ts_mean(close, 20))", "hypothesis": "trend"}]}
            store_idea_response(path, "Find persistent momentum using price trend and volume", "model-a", "catalog-a", response)
            rows = retrieve_similar(path, "Find persistent momentum using price trend", "model-a", "catalog-a", min_similarity=0.2)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ideas"][0]["expression"], "rank(ts_mean(close, 20))")
            self.assertGreater(rows[0]["similarity"], 0.2)

    def test_retrieval_is_scoped_to_model_and_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            store_idea_response(path, "momentum and volume trend", "model-a", "catalog-a", {"ideas": [{"expression": "rank(close)"}]})
            self.assertEqual(retrieve_similar(path, "momentum and volume trend", "model-b", "catalog-a"), [])
            self.assertEqual(retrieve_similar(path, "momentum and volume trend", "model-a", "catalog-b"), [])

    def test_raw_prompt_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            prompt = "secret project codename momentum thesis"
            store_idea_response(path, prompt, "model-a", "catalog-a", {"ideas": []})
            disk = path.read_text(encoding="utf-8")
            self.assertNotIn(prompt, disk)
            self.assertIn('"tokens"', disk)

    def test_persisted_candidate_payload_is_allowlisted_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            response = {
                "ideas": [
                    {
                        "id": "idea-1",
                        "name": "Momentum",
                        "expression": "rank(close)",
                        "hypothesis": "H" * 500,
                        "rationale": "R" * 500,
                        "expected_behavior": "E" * 500,
                        "risks": ["x" * 300, "risk two", "risk three", "risk four"],
                        "validation": "V" * 200,
                        "provenance": {
                            "validator": "validator-a",
                            "catalog_fingerprint": "catalog-a",
                            "scope": "static",
                            "unexpected": "do not persist",
                        },
                        "unexpected_model_payload": "X" * 10000,
                    },
                    {"expression": "z" * 1001, "name": "expression should not be truncated"},
                ],
                "rejected": [{"expression": "unneeded"}],
                "note": "unneeded top-level content",
            }
            store_idea_response(path, "momentum trend and volume", "model-a", "catalog-a", response)
            persisted = json.loads(path.read_text(encoding="utf-8"))
            stored_response = persisted["entries"][0]["response"]
            self.assertEqual(set(stored_response), {"ideas"})
            self.assertEqual(len(stored_response["ideas"]), 1)
            candidate = stored_response["ideas"][0]
            self.assertEqual(candidate["expression"], "rank(close)")
            self.assertEqual(len(candidate["hypothesis"]), 300)
            self.assertEqual(len(candidate["rationale"]), 250)
            self.assertEqual(len(candidate["expected_behavior"]), 200)
            self.assertEqual(len(candidate["risks"]), 3)
            self.assertEqual(len(candidate["risks"][0]), 160)
            self.assertEqual(set(candidate["provenance"]), {"validator", "catalog_fingerprint", "scope"})
            self.assertNotIn("unexpected_model_payload", candidate)
            self.assertNotIn("rejected", stored_response)

            matches = retrieve_similar(path, "momentum trend and volume", "model-a", "catalog-a")
            self.assertEqual(matches[0]["ideas"][0]["expression"], "rank(close)")

    def test_legacy_oversized_entry_is_compacted_when_read_and_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(json.dumps({
                "version": 1,
                "entries": [{
                    "token_signature": "legacy",
                    "tokens": ["momentum"] * (MAX_TOKENS_PER_IDEA + 20),
                    "model": "model-a",
                    "catalog_fingerprint": "catalog-a",
                    "response": {
                        "ideas": [{
                            "expression": "rank(close)",
                            "hypothesis": "H" * 1000,
                            "unexpected_payload": "X" * 10000,
                        }],
                        "unexpected_top_level": "Y" * 10000,
                    },
                }],
            }), encoding="utf-8")

            matches = retrieve_similar(path, "momentum", "model-a", "catalog-a")
            self.assertEqual(len(matches), 1)
            self.assertEqual(len(matches[0]["matched_tokens"]), MAX_TOKENS_PER_IDEA)
            self.assertEqual(len(matches[0]["ideas"][0]["hypothesis"]), 300)
            self.assertNotIn("unexpected_payload", matches[0]["ideas"][0])

            store_idea_response(path, "new momentum and volume idea", "model-a", "catalog-a", {"ideas": []})
            persisted = json.loads(path.read_text(encoding="utf-8"))
            legacy = persisted["entries"][0]
            self.assertEqual(len(legacy["tokens"]), MAX_TOKENS_PER_IDEA)
            self.assertNotIn("unexpected_top_level", legacy["response"])
            self.assertLess(len(path.read_bytes()), MAX_INDEX_FILE_BYTES)

    def test_oversized_index_is_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(" " * (MAX_INDEX_FILE_BYTES + 1), encoding="utf-8")
            self.assertEqual(retrieve_similar(path, "momentum idea", "model-a", "catalog-a"), [])

    def test_corrupt_index_is_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text("{broken", encoding="utf-8")
            self.assertEqual(retrieve_similar(path, "momentum idea", "model-a", "catalog-a"), [])


if __name__ == "__main__":
    unittest.main()
