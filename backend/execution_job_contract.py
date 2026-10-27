"""Stable, framework-independent contract for future backtest execution jobs.

This module defines job states and captures a JSON-safe configuration snapshot.
It does not enqueue work, persist jobs, or change the current synchronous API.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExecutionJobStatus(str, Enum):
    """Lifecycle states reserved for execution-job orchestration."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RunConfiguration:
    """Immutable, canonical snapshot of a backtest request payload.

    Store the canonical JSON string rather than a mutable dictionary. Each call
    to `to_dict` returns a new object, so callers cannot mutate the snapshot.
    The fingerprint is stable for equivalent JSON objects regardless of key order.
    """

    canonical_json: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RunConfiguration":
        if not isinstance(payload, dict):
            raise TypeError("Run configuration must be a JSON object.")
        try:
            canonical = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Run configuration must contain only JSON-safe values.") from exc
        return cls(canonical_json=canonical)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary; edits do not affect this snapshot."""
        return json.loads(self.canonical_json)

    @property
    def fingerprint(self) -> str:
        """SHA-256 identifier for this exact normalized configuration."""
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
