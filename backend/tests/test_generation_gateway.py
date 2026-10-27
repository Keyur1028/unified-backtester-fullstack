"""Offline contract tests for the bounded model gateway."""
import os
import sys
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from generation_gateway import invoke_json_generation


class ProviderError(Exception):
    def __init__(self, status_code):
        super().__init__(f"provider status {status_code}")
        self.status_code = status_code


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if not self.outcomes:
            raise AssertionError("unexpected provider call")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return type("Response", (), {"text": outcome})()


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


class GenerationGatewayTests(unittest.TestCase):
    def test_returns_nonempty_json_text(self):
        client = FakeClient(['{"ideas": []}'])
        result = invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client)
        self.assertEqual(result, '{"ideas": []}')
        self.assertEqual(client.models.calls, 1)

    def test_retries_transient_provider_error_once(self):
        client = FakeClient([ProviderError(503), '{"ideas": []}'])
        with patch("generation_gateway.time.sleep") as sleep:
            result = invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client)
        self.assertEqual(result, '{"ideas": []}')
        self.assertEqual(client.models.calls, 2)
        sleep.assert_called_once()

    def test_does_not_retry_nontransient_provider_error(self):
        client = FakeClient([ProviderError(400), '{"ideas": []}'])
        with self.assertRaises(ProviderError):
            invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client)
        self.assertEqual(client.models.calls, 1)

    def test_rejects_empty_or_oversized_responses(self):
        client = FakeClient([""])
        with self.assertRaisesRegex(ValueError, "empty response"):
            invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client)
        client = FakeClient(["x" * 128001])
        with self.assertRaisesRegex(ValueError, "size limit"):
            invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client)

    def test_rejects_missing_key_or_prompt_before_provider_call(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            invoke_json_generation("", "test-model", "prompt", client_factory=lambda **_: self.fail("called"))
        with self.assertRaisesRegex(ValueError, "prompt"):
            invoke_json_generation("secret", "test-model", " ", client_factory=lambda **_: self.fail("called"))


if __name__ == "__main__":
    unittest.main()
