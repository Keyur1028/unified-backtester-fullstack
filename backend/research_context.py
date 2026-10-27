"""Deterministic validation for the idea-assistant strategy handoff context."""
from __future__ import annotations

from typing import Any

_CONTEXT_KEYS = {
    "holding_period_bars", "execution_lag_bars", "execution_price",
    "portfolio", "weighting", "notes",
}


def validate_research_context(
    payload: dict[str, Any],
    strategy_context: dict[str, Any],
    interface: dict[str, Any],
) -> tuple[dict[str, Any], list[str], bool]:
    """Normalize prototype handoff fields against the live engine contract.

    Returns (normalized_context, errors, unsupported). Unsupported is reserved
    for a well-formed D0 request, which the current daily-bar engine cannot run.
    """
    errors: list[str] = []
    cluster = payload.get("cluster", "D1")
    universe = payload.get("universe", "NIFTY50")
    if not isinstance(cluster, str):
        errors.append("cluster must be a string (D1 for daily bars).")
        cluster = ""
    cluster = cluster.strip().upper()
    if cluster not in {"D0", "D1"}:
        errors.append("cluster must be D0 or D1.")
    unsupported = cluster == "D0"

    if not isinstance(universe, str):
        errors.append("universe must be a string supported by the production engine.")
        universe = ""
    universe = universe.strip().upper()
    universes = interface.get("universes", {})
    if not isinstance(universes, dict) or universe not in universes:
        errors.append("universe must name a universe in the production engine capability contract.")

    context = dict(strategy_context)
    unknown = sorted(set(context) - _CONTEXT_KEYS)
    if unknown:
        errors.append("Unsupported strategy_context field(s): " + ", ".join(unknown))

    for key, minimum, maximum in (
        ("holding_period_bars", 1, 2520),
        ("execution_lag_bars", 0, 10),
    ):
        if key not in context:
            continue
        value = context[key]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            errors.append(f"strategy_context.{key} must be an integer in [{minimum}, {maximum}].")

    if "execution_price" in context and context["execution_price"] not in {"open", "close"}:
        errors.append("strategy_context.execution_price must be open or close.")

    portfolio_types = interface.get("portfolio_types", [])
    if "portfolio" in context and context["portfolio"] not in portfolio_types:
        errors.append("strategy_context.portfolio must be a supported engine portfolio type.")

    weighting_schemes = interface.get("weighting_schemes", [])
    if "weighting" in context and context["weighting"] not in weighting_schemes:
        errors.append("strategy_context.weighting must be a supported engine weighting scheme.")

    if "notes" in context:
        if not isinstance(context["notes"], str):
            errors.append("strategy_context.notes must be a string.")
        elif len(context["notes"]) > 500:
            errors.append("strategy_context.notes must be 500 characters or fewer.")

    normalized = {"cluster": cluster, "universe": universe, "strategy_context": context}
    return normalized, errors, unsupported and not errors
