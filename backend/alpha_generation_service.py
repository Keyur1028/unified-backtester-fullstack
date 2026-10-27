"""Gemini-facing prompt construction and invocation for alpha generation."""
from __future__ import annotations

import json
from typing import Any, Iterable

from retrieval_context import format_retrieval_context


def build_generation_prompt(
    idea: str,
    fields: Iterable[str],
    operators: Iterable[str],
    similar_ideas: list[dict[str, Any]] | None = None,
    research_context: dict[str, Any] | None = None,
    catalog_context: dict[str, Any] | None = None,
) -> str:
    """Build the constrained prompt from the current catalog and retrieval context."""
    schema = {
        "ideas": [{
            "name": "short readable name",
            "hypothesis": "testable market hypothesis",
            "expression": "engine expression",
            "rationale": "why the signal could work",
            "expected_behavior": "what a high/low score means",
            "risks": ["research caveat"],
            "required_fields": ["field names referenced by the expression"],
            "entry_rules": ["explicit signal/entry rule"],
            "exit_rules": ["explicit exit/holding rule"],
            "execution_rules": ["timing/price/position rule"],
            "clarifying_questions": ["question if required details are ambiguous"],
            "assumptions": ["explicit assumption"],
        }]
    }
    prompt = (
        "You are a quantitative research assistant for a daily-bar alpha backtester. "
        "Convert the user's strategy into 3 materially different, testable alpha candidates. "
        "Return ONLY valid JSON matching this shape: " + json.dumps(schema) +
        "\nUser strategy: " + idea +
        "\nSupported fields (only these): " + json.dumps(sorted(fields)) +
        "\nSupported operators (only these): " + json.dumps(sorted(operators)) +
        "\nRules: use only listed fields/operators; do not invent data, fundamentals, sentiment, intraday data, or undocumented functions. "
        "Expressions must be simple expression-language strings, not Python code. Prefer cross-sectional ranking where appropriate. "
        "Use positive integer lookback windows. Do not promise profitability. If the idea needs unavailable data or intraday timing, "
        "state that limitation in risks and propose the closest honest daily OHLCV-only proxy. Generate distinct hypotheses, not merely different window lengths. "
        "Keep entry, exit, and execution rules separate from the expression; do not claim the expression implements stops, targets, or sizing unless the engine explicitly supports them. "
        "Return concise arrays for required_fields, entry_rules, exit_rules, execution_rules, clarifying_questions, and assumptions. "
        "If key strategy details are missing or ambiguous, ask questions instead of inventing rules."
    )
    if research_context:
        # Keep the notebook's structured rationale/context separate from the core idea.
        prompt += "\nStructured research context (user-supplied; treat as hypotheses, not facts): " + json.dumps(
            research_context, sort_keys=True, ensure_ascii=False
        )
        prompt += (
            "\nUse the data rationale to understand why fields may be relevant and the operator rationale to understand intended calculations. "
            "The active engine registry remains authoritative; reject unavailable requirements instead of inventing substitutes."
        )
    if catalog_context:
        prompt += (
            "\\nRelevant catalog entries retrieved from the active engine contract (not a substitute for the full registry): "
            + json.dumps(catalog_context, ensure_ascii=False, sort_keys=True)
        )
    if similar_ideas:
        examples = format_retrieval_context(similar_ideas)
        if examples != "[]":
            prompt += (
                "\nRetrieved prior candidates (compact lexical matches; examples only, not proof of validity): "
                + examples
                + "\nUse these only as research context. Do not copy an expression unless it fits the current idea; "
                  "produce distinct candidates and still obey the active field/operator registry."
            )
    return prompt


def generate_candidates(api_key: str, model: str, prompt: str) -> str:
    """Invoke the configured Gemini model and return its raw JSON response text."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0.35},
    )
    return response.text or "{}"
