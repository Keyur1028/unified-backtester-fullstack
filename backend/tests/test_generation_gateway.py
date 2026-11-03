"""Offline contract tests for the bounded model gateway."""
import os
import sys
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from generation_gateway import invoke_json_generation, plain_reason


class ProviderError(Exception):
    def __init__(self, status_code):
        super().__init__(f"provider status {status_code}")
        self.status_code = status_code


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.last_kwargs = None

    def generate_content(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
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
        self.assertEqual(client.models.last_kwargs["config"]["max_output_tokens"], 4096)

    def test_rejects_oversized_prompt_before_provider_call(self):
        client = FakeClient(['{"ideas": []}'])
        with self.assertRaisesRegex(ValueError, "prompt.*character limit"):
            invoke_json_generation("secret", "test-model", "x" * 80001, client_factory=lambda **_: client)
        self.assertEqual(client.models.calls, 0)

    def test_configurable_output_token_budget(self):
        client = FakeClient(['{"ideas": []}'])
        with patch.dict(os.environ, {"ALPHA_MODEL_MAX_OUTPUT_TOKENS": "1024"}):
            invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client)
        self.assertEqual(client.models.last_kwargs["config"]["max_output_tokens"], 1024)

    def test_retries_transient_provider_error_once(self):
        client = FakeClient([ProviderError(503), '{"ideas": []}'])
        with patch.dict(os.environ, {"ALPHA_MODEL_RETRIES": "1"}), patch("generation_gateway.time.sleep") as sleep:
            result = invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client)
        self.assertEqual(result, '{"ideas": []}')
        self.assertEqual(client.models.calls, 2)
        sleep.assert_called_once()

    def test_no_retry_by_default_to_avoid_double_billing(self):
        client = FakeClient([ProviderError(503), '{"ideas": []}'])
        with patch.dict(os.environ, {"ALPHA_MODEL_RETRIES": ""}), patch("generation_gateway.time.sleep") as sleep:
            with self.assertRaises(ProviderError):
                invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client)
        self.assertEqual(client.models.calls, 1)
        sleep.assert_not_called()

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

    def test_response_schema_is_sent_as_json_schema(self):
        client = FakeClient(['{"ideas": []}'])
        schema = {"type": "object"}
        invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client, response_schema=schema)
        self.assertEqual(client.models.last_kwargs["config"]["response_json_schema"], schema)

    def test_timeout_is_never_retried_because_it_may_be_billed(self):
        class ReadTimeout(Exception):
            pass

        client = FakeClient([ReadTimeout("slow"), '{"ideas": []}'])
        with patch.dict(os.environ, {"ALPHA_MODEL_RETRIES": "1"}), patch("generation_gateway.time.sleep"):
            with self.assertRaises(ReadTimeout):
                invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client)
        self.assertEqual(client.models.calls, 1)

    def test_daily_quota_is_not_retried(self):
        # The daily quota names its own window in the provider's words.
        client = FakeClient([Exception("429 RESOURCE_EXHAUSTED: quota exceeded, PerDay limit"), '{"ideas": []}'])
        with patch.dict(os.environ, {"ALPHA_MODEL_RETRIES": "1"}), patch("generation_gateway.time.sleep"):
            with self.assertRaises(Exception):
                invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client)
        self.assertEqual(client.models.calls, 1)

    def test_google_sdk_error_codes_are_classified(self):
        # google-genai exposes the HTTP status as ``code``, not ``status_code``.
        class SdkError(Exception):
            def __init__(self, code):
                super().__init__(f"{code} UNAVAILABLE")
                self.code = code

        client = FakeClient([SdkError(503), '{"ideas": []}'])
        with patch.dict(os.environ, {"ALPHA_MODEL_RETRIES": "1"}), patch("generation_gateway.time.sleep") as sleep:
            result = invoke_json_generation("secret", "test-model", "prompt", client_factory=lambda **_: client)
        self.assertEqual(result, '{"ideas": []}')
        sleep.assert_called_once()

    def test_plain_reason_names_the_fix_for_common_failures(self):
        self.assertIn("API key", plain_reason(ProviderError(403)))
        self.assertIn("overloaded", plain_reason(ProviderError(503)))
        self.assertIn("quota", plain_reason(ProviderError(429)))
        self.assertIsNone(plain_reason(ProviderError(418)))


if __name__ == "__main__":
    unittest.main()
