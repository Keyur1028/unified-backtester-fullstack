"""Offline regression tests for the bundled NIFTY100 ticker vector."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import alpha_engine as engine  # noqa: E402


class Nifty100UniverseTests(unittest.TestCase):
    def test_bundled_vector_has_exactly_100_unique_symbols(self):
        tickers = engine.get_universe_tickers("NIFTY100")
        self.assertEqual(len(tickers), 100)
        self.assertEqual(len(set(tickers)), 100)
        self.assertEqual(tickers, engine._FALLBACK_NIFTY100)

    def test_nifty100_selection_does_not_depend_on_a_csv_file(self):
        original_dir = engine.CONFIG["universe_dir"]
        try:
            with tempfile.TemporaryDirectory() as empty_dir:
                engine.CONFIG["universe_dir"] = empty_dir
                tickers = engine.get_universe_tickers("NIFTY100")
                self.assertEqual(len(tickers), 100)
        finally:
            engine.CONFIG["universe_dir"] = original_dir

    def test_nifty100_builds_static_mask_from_its_own_vector(self):
        tickers = engine.get_universe_tickers("NIFTY100")
        dates = engine.pd.bdate_range("2025-01-01", periods=3)
        selected = tickers[:3]
        close = engine.pd.DataFrame(
            [[100.0, 101.0, 102.0]] * len(dates),
            index=dates,
            columns=selected,
        )
        mask, metadata = engine.get_universe("NIFTY100", close)
        self.assertEqual(metadata["name"], "NIFTY100")
        self.assertEqual(metadata["asset_count"], 100)
        self.assertEqual(list(mask.columns), selected)
        self.assertTrue(mask.to_numpy().all())
        self.assertFalse(metadata["point_in_time"])
        self.assertTrue(metadata["survivorship_bias_risk"])


if __name__ == "__main__":
    unittest.main()
