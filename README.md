# Unified Backtester — connected frontend + backend

This package keeps the previously created React frontend design and connects it to a separate Python Flask API wrapping the uploaded `alpha_backtesting_engine_production_hardened` notebook.

## Project structure

- `frontend/` — existing React + Vite + Recharts UI and CSS; no Tailwind.
- `backend/app.py` — HTTP API, validation boundary, result adapter for the frontend, and local experiment-log persistence.
- `backend/alpha_engine.py` — implementation cells exported from the notebook, including the original expression engine, feature/operator registry, backtest runner, `BacktestResult`, `describe_interface()` and `run_backtest_api()`.
- `backend/storage/experiment_log.json` — created after the first API run to persist the notebook tracker locally.
- `backend/data_cache/`, `backend/universes/`, `backend/external_scores/` — paths expected by the notebook engine.

## Start the backend

Python 3.10+ recommended.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
python app.py
```

The API listens at `http://127.0.0.1:3000`.

The engine uses live Yahoo Finance data through `yfinance` by default. Install `pyarrow` (included in requirements) for the notebook's parquet cache. The requested universe's supporting CSVs must exist where the engine contract requires them; for example, `NIFTY100` requires `backend/universes/nifty100.csv`, and `TOP250`/`TOP500` require their candidate-pool CSV. The engine intentionally does not silently substitute a different universe.

## Start the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite forwards `/api/*` to the Flask API on port 3000. The existing UI components and CSS are retained; the API adapter converts the notebook result's time-series dictionaries into the chart-friendly format used by the frontend.

## API routes

- `GET /api/health` — backend and engine availability.
- `GET /api/meta` — live engine interface contract (fields, operators, universes, portfolio/weighting options).
- `GET /api/examples` — starter alpha expressions.
- `GET /api/experiments` — notebook experiment tracker history.
- `POST /api/backtest` — validates the JSON request using the notebook's `run_backtest_api()`, executes the engine, adapts the actual result for the UI and persists the tracker snapshot.

Example request:

```bash
curl -X POST http://127.0.0.1:3000/api/backtest \\
  -H 'Content-Type: application/json' \\
  -d '{
    "expression": "rank(ts_zscore(momentum_20, 20))",
    "universe": "NIFTY50",
    "start": "2022-01-01",
    "end": "2025-12-31",
    "holding_period": 5,
    "portfolio": "LONG_SHORT",
    "weighting": "EQUAL_WEIGHT",
    "transaction_cost_bps": 10,
    "slippage_bps": 5,
    "execution_lag_bars": 1,
    "execution_price": "close"
  }'
```

## Important limitations

- Backtests run synchronously in the Flask request. This is a connected MVP, not a production job system; long-running jobs can tie up a web worker. Add a queue/worker, polling, cancellation, timeouts, quotas and bounded concurrency before multi-user deployment.
- Experiment tracking uses the notebook's local JSON tracker. It is not a database and does not provide multi-process locking, tenant ownership or concurrent-writer guarantees.
- Authentication, per-user authorization, managed secrets, request rate limiting, audit retention policy and production deployment configuration are not implemented.
- The notebook's regression/synthetic tests are not executed at API startup. A successful HTTP connection does not guarantee live data access for every configured universe/date range.
- Result data is real engine output when the API call succeeds. If the API is offline or the live-data run fails, the unchanged frontend falls back to its labelled illustrative preview.
- The notebook's own operator metadata explicitly marks some operator semantics as assumed/unverified. The UI does not override those engine semantics.

## Prompt-to-alpha with Gemini

The Create Alpha screen now includes **Gemini Alpha Assistant**. A user can describe a strategy in plain English, receive three distinct candidate alpha hypotheses and expressions, inspect each candidate's rationale/expected behavior/risks, and click **Backtest this idea**. Candidate backtests use the same `/api/backtest` path and the same notebook-derived engine as manually written expressions.

### Configure Gemini

1. Create a Gemini API key in Google AI Studio.
2. In `backend/`, copy `.env.example` to `.env` and set:

   ```env
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```

3. Install backend dependencies and restart `python app.py`.
4. Open Create Alpha, write the strategy in the Gemini prompt box, and select **Generate alpha ideas**.

The key is read only by the Python backend. Do not put it in a `VITE_*` variable or commit `.env` to source control. The endpoint rejects blank/oversized prompts and checks each returned expression against the engine's field/operator registry and a restricted AST shape. These checks are not a performance guarantee; the actual engine performs its own runtime validation when the candidate is backtested.

### API addition

`POST /api/generate-alpha`

Request:

```json
{"idea":"Find stocks with persistent momentum but penalize unusually high recent volatility."}
```

Response includes `ideas[]` with `name`, `hypothesis`, `expression`, `rationale`, `expected_behavior`, and `risks`, plus a note explaining that the ideas need backtesting and robustness checks. Missing Gemini configuration returns a clear `503` response.

### Important research limitations

- Gemini-generated expressions are research hypotheses, not recommendations or claims of profitability.
- Only fields and operators exposed by the backend engine are allowed. Intraday, fundamental, sentiment, and other unavailable inputs must not be invented.
- The candidate backtest buttons run candidates individually, using the currently selected universe, date range, portfolio and cost settings.
- Inspect data coverage, survivorship bias, point-in-time limitations, execution timing, costs and out-of-sample robustness before drawing conclusions.

The two supplied notebooks are included in the project root as references: `Review_Engine.ipynb` and `idea_to_expression_assistant_reference.ipynb`.
