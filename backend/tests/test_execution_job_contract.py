"""Tests for the execution-job contract; no queue or provider is required."""
import os
import sys
import unittest
from dataclasses import FrozenInstanceError

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from execution_job_contract import ExecutionJobStatus, RunConfiguration  # noqa: E402


class ExecutionJobContractTests(unittest.TestCase):
    def test_status_values_are_stable_strings(self):
        self.assertEqual(
            {status.value for status in ExecutionJobStatus},
            {"queued", "running", "succeeded", "failed", "cancelled"},
        )
        self.assertEqual(ExecutionJobStatus.QUEUED, "queued")

    def test_configuration_snapshot_is_detached_from_input_and_output(self):
        payload = {"expression": "rank(close)", "options": {"universe": "NIFTY50"}}
        snapshot = RunConfiguration.from_payload(payload)
        payload["options"]["universe"] = "NIFTY100"
        exposed = snapshot.to_dict()
        exposed["options"]["universe"] = "TOP500"

        self.assertEqual(snapshot.to_dict()["options"]["universe"], "NIFTY50")
        with self.assertRaises(FrozenInstanceError):
            snapshot.canonical_json = "{}"

    def test_fingerprint_is_stable_across_object_key_order(self):
        first = RunConfiguration.from_payload({"expression": "rank(close)", "seed": 7})
        second = RunConfiguration.from_payload({"seed": 7, "expression": "rank(close)"})
        self.assertEqual(first.canonical_json, second.canonical_json)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(len(first.fingerprint), 64)

    def test_rejects_non_object_payload(self):
        with self.assertRaisesRegex(TypeError, "JSON object"):
            RunConfiguration.from_payload(["rank(close)"])

    def test_rejects_non_json_and_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "JSON-safe"):
            RunConfiguration.from_payload({"seed": object()})
        with self.assertRaisesRegex(ValueError, "JSON-safe"):
            RunConfiguration.from_payload({"metric": float("nan")})


if __name__ == "__main__":
    unittest.main()
