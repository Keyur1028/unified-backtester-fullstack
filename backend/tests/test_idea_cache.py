"""Unit tests for the persistent exact-match idea cache."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from idea_cache import (
    MAX_CACHE_FILE_BYTES,
    MAX_RESPONSE_BYTES,
    make_cache_key,
    normalize_idea,
    read_cached_response,
    write_cached_response,
)


class IdeaCacheTests(unittest.TestCase):
    def test_normalization_ignores_case_and_repeated_whitespace(self):
        self.assertEqual(normalize_idea("  Capture   MOMENTUM\nwith volume "), "capture momentum with volume")

    def test_cache_key_changes_when_catalog_or_model_changes(self):
        base = make_cache_key("momentum idea", ["close", "volume"], ["rank"], "model-a")
        self.assertEqual(base, make_cache_key(" MOMENTUM  IDEA ", ["volume", "close"], ["rank"], "model-a"))
        self.assertNotEqual(base, make_cache_key("momentum idea", ["close"], ["rank"], "model-a"))
        self.assertNotEqual(base, make_cache_key("momentum idea", ["close", "volume"], ["rank"], "model-b"))

    def test_cache_key_changes_when_prompt_contract_version_changes(self):
        base = make_cache_key("momentum idea", ["close"], ["rank"], "model-a")
        with patch("idea_cache.PROMPT_VERSION", 3):
            changed = make_cache_key("momentum idea", ["close"], ["rank"], "model-a")
        self.assertNotEqual(base, changed)

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

    def test_cached_candidate_payload_is_allowlisted_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            key = "a" * 64
            response = {
                "ideas": [{
                    "id": "idea-1",
                    "name": "N" * 500,
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
                        "unexpected": "drop me",
                    },
                    "unexpected_model_payload": "X" * 10000,
                }, {"expression": "z" * 1001, "name": "do not truncate expression"}],
                "rejected": [{"expression": "q" * 2000, "reason": "r" * 500}],
                "model": "test-model",
                "note": "n" * 1000,
                "unexpected_top_level": "Y" * 10000,
            }
            write_cached_response(path, key, response)
            cached = read_cached_response(path, key)
            self.assertEqual(len(cached["ideas"]), 1)
            candidate = cached["ideas"][0]
            self.assertEqual(candidate["expression"], "rank(close)")
            self.assertEqual(len(candidate["name"]), 100)
            self.assertEqual(len(candidate["hypothesis"]), 300)
            self.assertEqual(len(candidate["rationale"]), 250)
            self.assertEqual(len(candidate["expected_behavior"]), 200)
            self.assertEqual(len(candidate["risks"]), 3)
            self.assertEqual(len(candidate["risks"][0]), 160)
            self.assertEqual(set(candidate["provenance"]), {"validator", "catalog_fingerprint", "scope"})
            self.assertNotIn("unexpected_model_payload", candidate)
            self.assertNotIn("unexpected_top_level", cached)
            self.assertEqual(len(cached["rejected"][0]["expression"]), 1000)
            self.assertEqual(len(cached["rejected"][0]["reason"]), 250)
            self.assertLess(path.stat().st_size, MAX_CACHE_FILE_BYTES)

    def test_oversized_expression_is_excluded_not_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            write_cached_response(path, "key", {"ideas": [{"expression": "x" * 1001}]})
            self.assertEqual(read_cached_response(path, "key")["ideas"], [])

    def test_oversized_response_is_not_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            write_cached_response(path, "key", {"value": "x" * MAX_RESPONSE_BYTES})
            self.assertFalse(path.exists())

    def test_oversized_cache_file_is_a_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(" " * (MAX_CACHE_FILE_BYTES + 1), encoding="utf-8")
            self.assertIsNone(read_cached_response(path, "key"))

    def test_legacy_entries_are_compacted_when_cache_is_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            legacy_key = "b" * 64
            path.write_text(json.dumps({
                "version": 1,
                "entries": {
                    legacy_key: {"response": {
                        "ideas": [{"expression": "rank(close)", "hypothesis": "H" * 1000, "unknown": "X" * 10000}],
                        "unknown_top_level": "Y" * 10000,
                    }}
                },
            }), encoding="utf-8")
            write_cached_response(path, "new-key", {"ideas": [{"expression": "rank(volume)"}]})
            legacy = read_cached_response(path, legacy_key)
            self.assertEqual(len(legacy["ideas"][0]["hypothesis"]), 300)
            self.assertNotIn("unknown", legacy["ideas"][0])
            self.assertNotIn("unknown_top_level", legacy)
            self.assertLess(path.stat().st_size, MAX_CACHE_FILE_BYTES)


if __name__ == "__main__":
    unittest.main()
