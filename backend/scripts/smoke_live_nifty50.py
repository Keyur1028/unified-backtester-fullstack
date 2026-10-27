#!/usr/bin/env python3
"""Opt-in live NIFTY 50 backtest smoke test against a running API and worker.

This intentionally is not part of offline CI: it requires a configured backend,
a worker, a Supabase access token, and live Yahoo Finance access from the worker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(base_url: str, token: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(base_url.rstrip("/") + path, data=body, headers=headers, method="GET" if body is None else "POST")
    try:
        with urlopen(request, timeout=30) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"API request {path} failed with HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"API request {path} failed: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"API request {path} returned a non-object response.")
    return decoded


def run_smoke_test(base_url: str, token: str, start: str, end: str, timeout_seconds: int = 1800) -> dict[str, Any]:
    """Queue a real-data NIFTY50 backtest and validate the persisted result."""
    configuration = {
        "expression": "rank(ts_mean(close, 20))",
        "universe": "NIFTY50",
        "start": start,
        "end": end,
        "holding_period": 5,
        "portfolio": "LONG_SHORT",
        "weighting": "EQUAL_WEIGHT",
        "transaction_cost_bps": 10,
        "slippage_bps": 5,
        "execution_lag_bars": 1,
        "execution_price": "close",
    }
    job = request_json(base_url, token, "/api/execution-jobs", configuration)
    job_id = job.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError(f"API did not return a queued job ID: {job!r}")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current = request_json(base_url, token, f"/api/execution-jobs/{job_id}")
        status = current.get("status")
        if status == "succeeded":
            result = current.get("result")
            if not isinstance(result, dict) or result.get("status") != "success":
                raise RuntimeError(f"Job succeeded without a valid backtest result: {current!r}")
            if result.get("data_mode") != "live-provider":
                raise RuntimeError(f"Expected live-provider data mode, got {result.get('data_mode')!r}")
            dates = result.get("series", {}).get("dates", [])
            summary = result.get("summary", {})
            sharpe = summary.get("sharpe_net")
            if not isinstance(dates, list) or len(dates) < 20:
                raise RuntimeError(f"Backtest returned insufficient daily observations: {len(dates) if isinstance(dates, list) else 'invalid'}")
            if not isinstance(sharpe, (int, float)):
                raise RuntimeError("Backtest result is missing a numeric net Sharpe ratio.")
            universe = result.get("universe", {})
            if universe.get("name") != "NIFTY50":
                raise RuntimeError(f"Backtest used an unexpected universe: {universe.get('name')!r}")
            return {
                "status": "passed",
                "job_id": job_id,
                "expression": configuration["expression"],
                "universe": universe,
                "requested_start": start,
                "requested_end": end,
                "observations": len(dates),
                "first_observation": dates[0],
                "last_observation": dates[-1],
                "summary": summary,
                "warnings": result.get("warnings", []),
                "note": "Review universe point-in-time/survivorship warnings before interpreting performance.",
            }
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"Backtest job {job_id} ended as {status}: {current.get('error')}")
        if status not in {"queued", "running"}:
            raise RuntimeError(f"Unexpected job status {status!r}: {current!r}")
        time.sleep(3)
    raise TimeoutError(f"Backtest job {job_id} did not finish within {timeout_seconds} seconds.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("BACKEND_URL", "http://127.0.0.1:3000"))
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    token = os.getenv("SUPABASE_ACCESS_TOKEN", "").strip()
    if not token:
        parser.error("Set SUPABASE_ACCESS_TOKEN to a valid user access token.")
    if args.start >= args.end:
        parser.error("--start must be earlier than --end.")
    try:
        report = run_smoke_test(args.url, token, args.start, args.end, args.timeout)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
