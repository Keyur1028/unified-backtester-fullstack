"""Offline tests for optional Supabase JWT enforcement at the Flask boundary."""
import os
import sys
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app as api_app  # noqa: E402


class SupabaseAuthMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.client = api_app.app.test_client()
        self.env = patch.dict(os.environ, {"SUPABASE_AUTH_REQUIRED": "true", "SUPABASE_URL": "https://example.supabase.co"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_protected_route_rejects_missing_bearer_token(self):
        response = self.client.get("/api/experiments")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json()["error"]["type"], "AuthenticationRequired")

    def test_protected_route_rejects_invalid_token(self):
        with patch("authentication.verify_supabase_access_token", side_effect=ValueError("bad token")):
            response = self.client.get("/api/experiments", headers={"Authorization": "Bearer invalid"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json()["error"]["type"], "InvalidAccessToken")

    def test_valid_token_attaches_verified_subject(self):
        with patch("authentication.verify_supabase_access_token", return_value={"sub": "user-123", "aud": "authenticated"}):
            response = self.client.get("/api/experiments", headers={"Authorization": "Bearer valid"})
        self.assertEqual(response.status_code, 200)

    def test_health_route_remains_public(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)

    def test_auth_can_be_disabled_for_local_development(self):
        with patch.dict(os.environ, {"SUPABASE_AUTH_REQUIRED": "false"}):
            response = self.client.get("/api/experiments")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
