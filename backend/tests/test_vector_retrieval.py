"""Offline tests for SQLite-backed vector retrieval."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from vector_retrieval import embed_text, retrieve_similar_vectors, store_idea_vector_response  # noqa: E402


class VectorRetrievalTests(unittest.TestCase):
    def test_hashed_embeddings_are_deterministic_and_normalized(self):
        with patch.dict(os.environ, {"ALPHA_EMBEDDING_MODEL": ""}):
            first, backend_a = embed_text("bullish momentum in close prices")
            second, backend_b = embed_text("bullish momentum in close prices")
        self.assertEqual(first, second)
        self.assertEqual(backend_a, "hashed-token-v1")
        self.assertEqual(backend_b, "hashed-token-v1")
        self.assertAlmostEqual(sum(value * value for value in first), 1.0)

    def test_persists_and_retrieves_vectors_with_catalog_and_model_scoping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.sqlite3"
            with patch.dict(os.environ, {"ALPHA_EMBEDDING_MODEL": ""}):
                store_idea_vector_response(
                    path, "bullish momentum using close and volume", "model-a", "catalog-a",
                    {"ideas": [{"expression": "rank(close)", "entry_rules": ["enter bullish names"]}]},
                    safety_text="bullish momentum using close and volume",
                )
                matches = retrieve_similar_vectors(
                    path, "bullish momentum using close and volume", "model-a", "catalog-a",
                    min_similarity=0.5,
                )
                wrong_model = retrieve_similar_vectors(
                    path, "bullish momentum using close and volume", "model-b", "catalog-a",
                )
                wrong_catalog = retrieve_similar_vectors(
                    path, "bullish momentum using close and volume", "model-a", "catalog-b",
                )
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["ideas"][0]["expression"], "rank(close)")
            self.assertEqual(matches[0]["matched_tokens"][:2], ["bullish", "momentum"])
            self.assertEqual(wrong_model, [])
            self.assertEqual(wrong_catalog, [])

            with sqlite3.connect(path) as connection:
                stored = connection.execute("SELECT response_json, safety_tokens_json FROM idea_vectors").fetchone()
            self.assertNotIn("bullish momentum using close and volume", stored[0])
            self.assertIn("bullish", json.loads(stored[1]))

    def test_similar_text_scores_above_unrelated_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.sqlite3"
            with patch.dict(os.environ, {"ALPHA_EMBEDDING_MODEL": ""}):
                store_idea_vector_response(
                    path, "bullish momentum using price strength", "model-a", "catalog-a",
                    {"ideas": [{"expression": "rank(close)"}]},
                )
                related = retrieve_similar_vectors(
                    path, "bullish momentum price strength", "model-a", "catalog-a", min_similarity=-1,
                )
                unrelated = retrieve_similar_vectors(
                    path, "seasonal volatility reversal", "model-a", "catalog-a", min_similarity=-1,
                )
        self.assertGreater(related[0]["similarity"], unrelated[0]["similarity"])


if __name__ == "__main__":
    unittest.main()
