"""Build a bounded strategy handoff without claiming a backtest was run."""
from __future__ import annotations

import ast
from typing import Any


def _expression_inventory(expression: str, fields: set[str], operators: set[str]) -> tuple[list[str], list[str]]:
    tree = ast.parse(expression, mode="eval")
    calls = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return sorted(names & fields), sorted(calls & operators)


def build_strategy_handoff(
    candidate: dict[str, Any],
    research_context: dict[str, Any],
    fields: set[str],
    operators: set[str],
    catalog_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a handoff with expression-derived requirements and bounded RAG hints."""
    expression = str(candidate.get("expression", "")).strip()
    fields_used, operators_used = _expression_inventory(expression, fields, operators)
    context = research_context.get("strategy_context", {})
    cluster = research_context.get("cluster", "D1")
    universe = research_context.get("universe", "NIFTY50")
    entry_rules = candidate.get("entry_rules", [])
    exit_rules = candidate.get("exit_rules", [])
    execution_rules = candidate.get("execution_rules", [])
    questions = candidate.get("clarifying_questions", [])
    warnings = candidate.get("risks", [])
    complete_rules = all(isinstance(v, list) and v for v in (entry_rules, exit_rules, execution_rules))
    status = "ready_for_notebook2" if complete_rules and not questions and cluster == "D1" else "needs_clarification"
    catalog_context = catalog_context if isinstance(catalog_context, dict) else {}
    retrieved_fields = [
        str(row["name"]) for row in catalog_context.get("fields", [])
        if isinstance(row, dict) and row.get("name") in fields
    ][:8]
    retrieved_operators = [
        str(row["name"]) for row in catalog_context.get("operators", [])
        if isinstance(row, dict) and row.get("name") in operators
    ][:8]
    return {
        "schema_version": "1.0",
        "assistant_status": status,
        "hypothesis": candidate.get("hypothesis", ""),
        "alpha_expression": expression,
        "expression_validation": {
            "valid_syntax": True,
            "fields_used": fields_used,
            "operators_used": operators_used,
            "validation_scope": "registry validation plus deterministic synthetic runtime probe",
        },
        "required_operators": operators_used,
        "retrieved_capabilities": {
            "source": "active-engine-catalog-rag",
            "fields": retrieved_fields,
            "operators": retrieved_operators,
        },
        "cluster": cluster,
        "universe": universe,
        "required_fields": fields_used,
        "entry_rules": entry_rules,
        "exit_rules": exit_rules,
        "execution_rules": execution_rules,
        "strategy_context": context,
        "clarifying_questions": questions,
        "warnings": warnings,
        "assumptions": candidate.get("assumptions", []),
        "backtest_executed": False,
        "notebook2_must_revalidate": True,
    }
