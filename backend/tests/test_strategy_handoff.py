"""Tests for strategy handoff readiness and authoritative expression inventory."""
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from strategy_handoff import build_strategy_handoff  # noqa: E402


class StrategyHandoffTests(unittest.TestCase):
    def test_incomplete_strategy_needs_clarification(self):
        handoff = build_strategy_handoff(
            {"expression": "rank(close)", "hypothesis": "Rank prices."},
            {"cluster": "D1", "universe": "NIFTY50", "strategy_context": {}},
            {"close"}, {"rank"},
        )
        self.assertEqual(handoff["assistant_status"], "needs_clarification")
        self.assertEqual(handoff["required_fields"], ["close"])
        self.assertEqual(handoff["expression_validation"]["operators_used"], ["rank"])
        self.assertFalse(handoff["backtest_executed"])

    def test_complete_daily_strategy_is_handed_off_for_independent_revalidation(self):
        candidate = {
            "expression": "rank(ts_mean(close, 20))",
            "hypothesis": "Price strength persists.",
            "entry_rules": ["Enter high-ranked names."],
            "exit_rules": ["Exit after the holding period."],
            "execution_rules": ["Use daily bars and configured lag."],
            "clarifying_questions": [],
        }
        handoff = build_strategy_handoff(
            candidate,
            {"cluster": "D1", "universe": "NIFTY100", "strategy_context": {"holding_period_bars": 5}},
            {"close"}, {"rank", "ts_mean"},
        )
        self.assertEqual(handoff["assistant_status"], "ready_for_notebook2")
        self.assertEqual(handoff["required_fields"], ["close"])
        self.assertEqual(handoff["expression_validation"]["operators_used"], ["rank", "ts_mean"])
        self.assertEqual(handoff["universe"], "NIFTY100")
        self.assertTrue(handoff["notebook2_must_revalidate"])


if __name__ == "__main__":
    unittest.main()
