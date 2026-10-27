"""Allowlist and bound model-generated alpha candidate payloads."""
from __future__ import annotations

from typing import Any

MAX_EXPRESSION_LENGTH = 1000
TEXT_LIMITS = {
    "name": 100,
    "hypothesis": 300,
    "rationale": 250,
    "expected_behavior": 200,
}
MAX_RISKS = 3
MAX_RISK_LENGTH = 160
MAX_RULES = 5
MAX_RULE_LENGTH = 240
LIST_TEXT_FIELDS = ("required_fields", "entry_rules", "exit_rules", "execution_rules",
                    "clarifying_questions", "assumptions")


def compact_generated_candidate(item: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Return a bounded candidate or a reason to reject it before validation."""
    if not isinstance(item, dict):
        return None, "Candidate must be a JSON object"
    expression = item.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        return None, "Candidate expression must be a non-empty string"
    expression = expression.strip()
    # Never truncate executable expressions: that would change their meaning.
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return None, f"Expression exceeds the {MAX_EXPRESSION_LENGTH}-character limit"

    compact: dict[str, Any] = {"expression": expression}
    for key, limit in TEXT_LIMITS.items():
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            compact[key] = value.strip()[:limit]
    risks = item.get("risks")
    if isinstance(risks, list):
        bounded_risks = [
            risk.strip()[:MAX_RISK_LENGTH]
            for risk in risks[:MAX_RISKS]
            if isinstance(risk, str) and risk.strip()
        ]
        if bounded_risks:
            compact["risks"] = bounded_risks
    for key in LIST_TEXT_FIELDS:
        value = item.get(key)
        if isinstance(value, list):
            bounded = [
                entry.strip()[:MAX_RULE_LENGTH]
                for entry in value[:MAX_RULES]
                if isinstance(entry, str) and entry.strip()
            ]
            if bounded:
                compact[key] = bounded
    return compact, None
