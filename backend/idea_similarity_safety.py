"""Conservative safeguards for near-duplicate alpha prompt reuse.

Lexical similarity alone can treat opposing trading directions as equivalent.
These checks are intentionally conservative: a detected conflict disables
automatic reuse and lets the normal model-generation path decide.
"""
from __future__ import annotations

# Each pair represents tokens that can signal opposing intent in a strategy
# brief. This is not a finance-language parser; it is a guardrail, not proof of
# semantic equivalence.
_OPPOSING_GROUPS = (
    frozenset({"bullish", "bearish", "bull", "bear"}),
    frozenset({"long", "short"}),
    frozenset({"buy", "sell", "buying", "selling"}),
    frozenset({"positive", "negative"}),
    frozenset({"rise", "fall", "rising", "falling", "rises", "falls"}),
    frozenset({"increase", "decrease", "increasing", "decreasing"}),
    frozenset({"strengthen", "weaken", "strengthening", "weakening"}),
    frozenset({"up", "down"}),
    frozenset({"outperform", "underperform"}),
    frozenset({"trend_continuation", "mean_reversion", "reversal"}),
)


def has_directional_conflict(left_tokens: list[str], right_tokens: list[str]) -> bool:
    """Return True when two token sets contain opposing direction/style terms.

    Tokens may come from `idea_tokens`; underscores are preserved by that
    tokenizer. If either side contains both sides of a pair, the check still
    errs on the side of disabling automatic reuse.
    """
    left = {str(token).casefold() for token in left_tokens}
    right = {str(token).casefold() for token in right_tokens}
    for group in _OPPOSING_GROUPS:
        left_present = left & group
        right_present = right & group
        if not left_present or not right_present:
            continue
        # For a group with more than two variants, compare semantic sides
        # rather than treating synonyms within the same side as conflicting.
        if group == _OPPOSING_GROUPS[0]:
            left_positive, right_positive = {"bullish", "bull"}, {"bullish", "bull"}
        elif group == _OPPOSING_GROUPS[2]:
            left_positive, right_positive = {"buy", "buying"}, {"buy", "buying"}
        elif group == _OPPOSING_GROUPS[4]:
            left_positive, right_positive = {"rise", "rising", "rises"}, {"rise", "rising", "rises"}
        elif group == _OPPOSING_GROUPS[5]:
            left_positive, right_positive = {"increase", "increasing"}, {"increase", "increasing"}
        elif group == _OPPOSING_GROUPS[6]:
            left_positive, right_positive = {"strengthen", "strengthening"}, {"strengthen", "strengthening"}
        else:
            # All remaining groups are either a pair or intentionally treated
            # as distinct strategy modes.
            left_positive, right_positive = set(), set()
        if left_positive:
            left_side = bool(left_present & left_positive)
            right_side = bool(right_present & right_positive)
            left_opposite = bool(left_present - left_positive)
            right_opposite = bool(right_present - right_positive)
            if (left_side and right_opposite) or (left_opposite and right_side):
                return True
        elif left_present != right_present:
            return True
    return False
