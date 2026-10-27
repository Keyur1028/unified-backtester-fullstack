"""Service boundary for executing a backtest through the research engine.

Keep HTTP concerns in the Flask layer and all research calculations in alpha_engine.
This small boundary is intentionally synchronous; background-job orchestration is a
separate change and must preserve the existing API contract.
"""
from __future__ import annotations

from typing import Any

import alpha_engine as engine


def run_backtest(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one backtest using the authoritative research engine.

    The request payload and engine result are passed through unchanged so this
    refactor does not duplicate validation, calculations, or result semantics.
    """
    return engine.run_backtest_api(payload)
