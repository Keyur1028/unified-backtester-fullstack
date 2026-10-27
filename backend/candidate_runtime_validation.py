"""Deterministic, offline runtime preflight for generated alpha expressions.

This probe catches engine-level incompatibilities without downloading market data
or claiming that an expression is economically useful or valid on real data.
"""
from __future__ import annotations

from typing import Iterable


def validate_candidate_runtime(engine, expression: str, field_names: Iterable[str]) -> tuple[bool, str | None]:
    """Evaluate an expression on deterministic synthetic OHLCV-shaped data."""
    clear_cache = getattr(engine, "clear_expression_cache", None)
    try:
        import numpy as np
        import pandas as pd

        row_count, asset_count = 260, 8
        dates = pd.bdate_range("2020-01-01", periods=row_count)
        tickers = [f"PROBE{i}" for i in range(asset_count)]
        rng = np.random.default_rng(20261027)
        returns = rng.normal(loc=0.0002, scale=0.012, size=(row_count, asset_count))
        close = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)
        open_ = close * (1.0 + rng.normal(0.0, 0.002, size=close.shape))
        spread = np.abs(rng.normal(0.004, 0.002, size=close.shape))
        high = pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()) * (1.0 + spread), index=dates, columns=tickers)
        low = pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()) * (1.0 - spread), index=dates, columns=tickers)
        volume = pd.DataFrame(np.exp(rng.normal(12.0, 0.8, size=close.shape)), index=dates, columns=tickers)
        ohlcv = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
        fields = dict(ohlcv)
        fields.update(engine.compute_features(ohlcv))

        for name in sorted(set(str(item) for item in field_names)):
            if name not in fields:
                fields[name] = pd.DataFrame(rng.normal(size=(row_count, asset_count)), index=dates, columns=tickers)

        group_map = {ticker: ("GROUP_A" if i < asset_count // 2 else "GROUP_B") for i, ticker in enumerate(tickers)}
        if callable(clear_cache):
            clear_cache()
        result = engine.evaluate_expression(expression, fields, group_map=group_map)
        if not isinstance(result, pd.DataFrame):
            return False, "Expression did not return a date-by-asset DataFrame."
        if result.shape != (row_count, asset_count):
            return False, "Expression returned an unexpected result shape."
        try:
            values = result.to_numpy(dtype=float)
        except (TypeError, ValueError):
            return False, "Expression returned non-numeric values."
        if not np.isfinite(values).any():
            return False, "Expression produced no finite values on synthetic probe data."
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:180]}"
    finally:
        if callable(clear_cache):
            try:
                clear_cache()
            except Exception:
                pass
