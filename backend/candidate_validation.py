"""Deterministic syntax/registry validation and provenance for alpha candidates."""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Iterable

VALIDATOR_VERSION = "alpha-expression-validator-v1"
_ALLOWED_CONSTANT_NAMES = {"True", "False", "None"}
_FORBIDDEN_NODES = (
    ast.Attribute,
    ast.Subscript,
    ast.Lambda,
    ast.DictComp,
    ast.ListComp,
    ast.GeneratorExp,
    ast.NamedExpr,
)


def catalog_fingerprint(fields: Iterable[str], operators: Iterable[str]) -> str:
    """Stable digest identifying the field/operator catalog used for validation."""
    payload = {
        "validator": VALIDATOR_VERSION,
        "fields": sorted(set(str(x) for x in fields)),
        "operators": sorted(set(str(x) for x in operators)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_expression(expression: str, fields: Iterable[str], operators: Iterable[str]) -> tuple[bool, str | None]:
    """Check parseability, allowed syntax, and registered field/function names.

    This is a static gate only; it does not execute the expression or guarantee
    that the engine can evaluate it for every universe/date window.
    """
    field_set = set(str(x) for x in fields)
    operator_set = set(str(x) for x in operators)
    try:
        tree = ast.parse(str(expression).strip(), mode="eval")
    except (SyntaxError, ValueError) as exc:
        return False, f"Invalid expression syntax: {getattr(exc, 'msg', str(exc))}"

    allowed_names = field_set | operator_set | _ALLOWED_CONSTANT_NAMES
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            return False, "Unsupported identifier: " + node.id
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            return False, "Only registered function calls are allowed"
        if isinstance(node, _FORBIDDEN_NODES):
            return False, "Unsupported expression syntax"

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    unknown_calls = called - operator_set
    if unknown_calls:
        return False, "Unsupported operator(s): " + ", ".join(sorted(unknown_calls))

    used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    unknown_fields = used_names - operator_set - field_set - _ALLOWED_CONSTANT_NAMES
    if unknown_fields:
        return False, "Unsupported field(s): " + ", ".join(sorted(unknown_fields))

    return True, None
