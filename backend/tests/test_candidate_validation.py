"""Offline tests for deterministic alpha candidate validation and provenance."""
import unittest

from candidate_validation import VALIDATOR_VERSION, catalog_fingerprint, validate_expression


class CandidateValidationTests(unittest.TestCase):
    def setUp(self):
        self.fields = {"close", "volume", "momentum_5"}
        self.operators = {"rank", "ts_mean", "group_neutralize"}

    def test_accepts_registered_expression(self):
        self.assertEqual(validate_expression("rank(ts_mean(close, 5))", self.fields, self.operators), (True, None))

    def test_rejects_unknown_field(self):
        valid, reason = validate_expression("rank(secret_price)", self.fields, self.operators)
        self.assertFalse(valid)
        self.assertIn("Unsupported identifier: secret_price", reason)

    def test_rejects_unknown_function(self):
        valid, reason = validate_expression("invented_operator(close)", self.fields, self.operators)
        self.assertFalse(valid)
        self.assertIn("Unsupported operator", reason)

    def test_rejects_attribute_and_subscript_access(self):
        for expression in ("close.real", "close[0]"):
            with self.subTest(expression=expression):
                valid, reason = validate_expression(expression, self.fields, self.operators)
                self.assertFalse(valid)
                self.assertIn("Unsupported expression syntax", reason)

    def test_rejects_non_expression_syntax(self):
        valid, reason = validate_expression("rank(", self.fields, self.operators)
        self.assertFalse(valid)
        self.assertIn("Invalid expression syntax", reason)

    def test_catalog_fingerprint_is_order_independent_and_sensitive_to_changes(self):
        base = catalog_fingerprint(["close", "volume"], ["rank", "ts_mean"])
        self.assertEqual(base, catalog_fingerprint(["volume", "close"], ["ts_mean", "rank"]))
        self.assertNotEqual(base, catalog_fingerprint(["close"], ["rank", "ts_mean"]))
        self.assertEqual(len(base), 64)

    def test_validator_version_is_explicit(self):
        self.assertTrue(VALIDATOR_VERSION.startswith("alpha-expression-validator-v"))


if __name__ == "__main__":
    unittest.main()
