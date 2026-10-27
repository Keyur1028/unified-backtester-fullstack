"""Fail-closed Supabase JWT verification for protected Flask API routes.

Local development remains unauthenticated unless SUPABASE_AUTH_REQUIRED=true.
Production deployments should set that flag and SUPABASE_URL; there is no
fallback to trusting client-supplied user IDs or unverified JWT claims.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import jwt
from flask import Flask, g, jsonify, request
from jwt import PyJWKClient


PROTECTED_PREFIXES = (
    "/api/generate-alpha",
    "/api/backtest",
    "/api/experiments",
    "/api/execution-jobs",
)


class AuthenticationConfigurationError(RuntimeError):
    """Raised when authentication is enabled without required configuration."""


@lru_cache(maxsize=8)
def _jwks_client(jwks_url: str) -> PyJWKClient:
    return PyJWKClient(jwks_url, cache_keys=True, lifespan=300)


def verify_supabase_access_token(token: str) -> dict[str, Any]:
    """Verify a Supabase access token's signature, issuer, audience and expiry."""
    project_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    if not project_url:
        raise AuthenticationConfigurationError(
            "SUPABASE_URL must be configured when authentication is required."
        )
    issuer = f"{project_url}/auth/v1"
    client = _jwks_client(f"{issuer}/.well-known/jwks.json")
    signing_key = client.get_signing_key_from_jwt(token)
    claims = jwt.decode(
        token,
        signing_key.key,
        algorithms=["ES256", "RS256"],
        audience="authenticated",
        issuer=issuer,
        options={"require": ["exp", "iat", "sub", "iss", "aud"]},
    )
    if not isinstance(claims.get("sub"), str) or not claims["sub"].strip():
        raise jwt.InvalidTokenError("Token subject is missing.")
    return claims


def _is_protected_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in PROTECTED_PREFIXES)


def register_auth_middleware(app: Flask) -> None:
    """Register auth checks while keeping health/meta/examples publicly readable."""
    @app.before_request
    def require_supabase_access_token():
        if not _is_protected_path(request.path):
            return None
        if os.getenv("SUPABASE_AUTH_REQUIRED", "false").strip().lower() not in {
            "1", "true", "yes", "on"
        }:
            g.user_id = None
            return None
        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return jsonify({
                "error": {
                    "type": "AuthenticationRequired",
                    "message": "A valid Supabase access token is required.",
                }
            }), 401
        try:
            claims = verify_supabase_access_token(token.strip())
        except AuthenticationConfigurationError:
            app.logger.exception("Supabase authentication is enabled but misconfigured")
            return jsonify({
                "error": {
                    "type": "AuthenticationUnavailable",
                    "message": "Authentication is temporarily unavailable.",
                }
            }), 503
        except Exception:
            return jsonify({
                "error": {
                    "type": "InvalidAccessToken",
                    "message": "The access token is invalid or expired.",
                }
            }), 401
        g.user_id = claims["sub"]
        g.auth_claims = claims
        return None
