"""Unit tests for the persistent exact-match idea cache."""
import json
import tempfile
import unittest
from pathlib import Path

from idea_cache import make_cache_key, normalize_idea, read_cached_response, write_cached_response


class IdeaCacheTests(unittest.TestCase):
    def test_normalization_ignores_case_and_repeated_whitespace(self):
        self.assertEqual(normalize_idea("  Capture   MOMENTUM\nwith volume "), "capture momentum with volume")

    def test_cache_key_changes_when_catalog_or_model_changes(self):
        base = make_cache_key("momentum idea", ["close", "volume"], ["rank"], "model-a")
        self.assertEqual(base, make_cache_key(" MOMENTUM  IDEA ", ["volume", "close"], ["rank"], "model-a"))
        self.assertNotEqual(base, make_cache_key("momentum idea", ["close"], ["rank"], "model-a"))
        self.assertNotEqual(base, make_cache_key("momentum idea", ["close", "volume"], ["rank"], "model-b"))

    def test_response_persists_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            response = {"ideas": [{"expression": "rank(close)"}], "model": "test-model"}
            key = make_cache_key("momentum idea", ["close"], ["rank"], "test-model")
            write_cached_response(path, key, response)
            self.assertEqual(read_cached_response(path, key), response)
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("momentum idea", path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["version"], 1)

    def test_corrupt_cache_is_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text("{not-json", encoding="utf-8")
            self.assertIsNone(read_cached_response(path, "key"))

    def test_cache_retains_only_most_recent_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            write_cached_response(path, "one", {"value": 1}, max_entries=2)
            write_cached_response(path, "two", {"value": 2}, max_entries=2)
            write_cached_response(path, "three", {"value": 3}, max_entries=2)
            self.assertIsNone(read_cached_response(path, "one"))
            self.assertEqual(read_cached_response(path, "two"), {"value": 2})
            self.assertEqual(read_cached_response(path, "three"), {"value": 3})


if __name__ == "__main__":
    unittest.main()
