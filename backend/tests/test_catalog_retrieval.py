"""Tests for retrieval against the live engine capability catalog."""
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from catalog_retrieval import retrieve_catalog_context  # noqa: E402


class CatalogRetrievalTests(unittest.TestCase):
    def test_returns_only_registered_relevant_fields_and_operators(self):
        interface = {
            "fields": {"native_ohlcv": ["close", "volume"], "engineered": ["momentum_20"]},
            "operators": {"verified_semantics": {"ts_mean": "Rolling time-series mean"}},
        }
        result = retrieve_catalog_context(
            "persistent momentum trend using rolling mean of close and volume",
            interface, {"close", "volume", "momentum_20", "rsi_14"},
            {"rank", "ts_mean", "ts_std_dev"}, limit_per_kind=3,
        )
        self.assertLessEqual(len(result["fields"]), 3)
        self.assertLessEqual(len(result["operators"]), 3)
        self.assertIn("momentum_20", [row["name"] for row in result["fields"]])
        self.assertIn("ts_mean", [row["name"] for row in result["operators"]])
        self.assertNotIn("not_registered", [row["name"] for row in result["fields"]])


    def test_vector_catalog_is_persisted_and_used_on_repeat_query(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "catalog.sqlite3")
            interface = {
                "fields": {"native_ohlcv": ["close", "volume"], "engineered": ["momentum_20"]},
                "operators": {"verified_semantics": {"ts_mean": "Rolling time-series mean"}},
            }
            first = retrieve_catalog_context(
                "persistent momentum trend using rolling mean of close and volume",
                interface, {"close", "volume", "momentum_20", "rsi_14"},
                {"rank", "ts_mean", "ts_std_dev"}, limit_per_kind=3, vector_db_path=db_path,
            )
            second = retrieve_catalog_context(
                "persistent momentum trend using rolling mean of close and volume",
                interface, {"close", "volume", "momentum_20", "rsi_14"},
                {"rank", "ts_mean", "ts_std_dev"}, limit_per_kind=3, vector_db_path=db_path,
            )
            self.assertEqual(
                [row["name"] for row in first["fields"]],
                [row["name"] for row in second["fields"]],
            )
            self.assertIn("momentum_20", [row["name"] for row in first["fields"]])
            self.assertIn("ts_mean", [row["name"] for row in first["operators"]])

    def test_unverified_operator_semantics_are_labelled_from_registry(self):
        result = retrieve_catalog_context(
            "kth element", {"operators": {"unverified_semantics": {"kth_element": "Unverified semantics"}}},
            set(), {"kth_element"},
        )
        self.assertEqual(result["operators"][0]["semantics"], "Unverified semantics")


if __name__ == "__main__":
    unittest.main()
