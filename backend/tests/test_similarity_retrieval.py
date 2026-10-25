"""Offline tests for the local alpha idea retrieval index."""
import json
import tempfile
import unittest
from pathlib import Path

from similarity_retrieval import (
    cosine_similarity,
    idea_tokens,
    retrieve_similar,
    store_idea_response,
)


class SimilarityRetrievalTests(unittest.TestCase):
    def test_tokens_normalize_and_drop_common_words(self):
        self.assertEqual(idea_tokens("Capture MOMENTUM with high volume"), ["momentum", "high", "volume"])

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

    def test_corrupt_index_is_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text("{broken", encoding="utf-8")
            self.assertEqual(retrieve_similar(path, "momentum idea", "model-a", "catalog-a"), [])


if __name__ == "__main__":
    unittest.main()
