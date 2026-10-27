"""Tests for deterministic idea-assistant strategy-context validation."""
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from research_context import validate_research_context  # noqa: E402

INTERFACE = {
    "universes": {"NIFTY50": "static", "NIFTY100": "constituent file required"},
    "portfolio_types": ["LONG_ONLY", "LONG_SHORT"],
    "weighting_schemes": ["EQUAL_WEIGHT", "SIGNAL_WEIGHTED"],
}

class ResearchContextTests(unittest.TestCase):
    def test_normalizes_supported_context(self):
        context, errors, unsupported = validate_research_context(
            {"cluster": "d1", "universe": "nifty100"},
            {"holding_period_bars": 5, "execution_lag_bars": 1,
             "execution_price": "open", "portfolio": "LONG_ONLY"},
            INTERFACE,
        )
        self.assertEqual(errors, [])
        self.assertFalse(unsupported)
        self.assertEqual(context["cluster"], "D1")
        self.assertEqual(context["universe"], "NIFTY100")
        self.assertEqual(context["strategy_context"]["holding_period_bars"], 5)

    def test_daily_engine_rejects_intraday_before_model_call(self):
        context, errors, unsupported = validate_research_context(
            {"cluster": "D0", "universe": "NIFTY50"}, {}, INTERFACE
        )
        self.assertEqual(errors, [])
        self.assertTrue(unsupported)
        self.assertEqual(context["cluster"], "D0")

    def test_invalid_context_fields_fail_closed(self):
        _, errors, unsupported = validate_research_context(
            {"cluster": "D1", "universe": "NIFTY50"},
            {"holding_period_bars": True, "execution_lag_bars": -1,
             "execution_price": "mid", "portfolio": "NOT_REAL", "unexpected": "value"},
            INTERFACE,
        )
        self.assertFalse(unsupported)
        self.assertGreaterEqual(len(errors), 5)

    def test_unknown_universe_and_oversized_notes_are_rejected(self):
        _, errors, _ = validate_research_context(
            {"cluster": "D1", "universe": "CRYPTO"}, {"notes": "n" * 501}, INTERFACE
        )
        self.assertTrue(any("universe" in error for error in errors))
        self.assertTrue(any("500 characters" in error for error in errors))

if __name__ == "__main__":
    unittest.main()
