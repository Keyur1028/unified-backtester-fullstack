# Unified Backtester — connected frontend + backend

This repository connects the React + Vite frontend to a Python Flask API that wraps the `alpha_backtesting_engine_production_hardened` notebook-derived engine. The notebook remains the source of truth for expression evaluation, portfolio construction, and backtest calculations.

## Project structure

- `frontend/` — React + Vite + Recharts UI and CSS.
- `backend/app.py` — HTTP API, request validation, result adaptation, alpha generation, and execution-job routes.
- `backend/alpha_engine.py` — notebook-derived expression engine, feature/operator registry, backtest runner, and interface metadata.
- `backend/backtest_service.py` — service boundary around the authoritative backtest execution call.
- `backend/execution_job_contract.py` and `backend/execution_job_store.py` — execution configuration/status contract and local SQLite persistence.
- `backend/storage/` — local runtime state, including the SQLite execution-job database and notebook experiment log.
- `backend/data_cache/`, `backend/universes/`, `backend/external_scores/` — data/cache paths expected by the engine.

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

The engine uses live Yahoo Finance data through `yfinance` by default. Install `pyarrow` (included in requirements) for the parquet cache. Required universe files must exist where the engine contract expects them; for example, `NIFTY100` requires `backend/universes/nifty100.csv`, and `TOP250`/`TOP500` require their candidate-pool CSVs. The engine does not silently substitute a different universe.

## Start the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite forwards `/api/*` to the Flask API on port 3000. The API adapter converts the notebook result's time-series dictionaries into the chart-friendly format used by the frontend.

## API routes

- `GET /api/health` — backend and engine availability.
- `GET /api/meta` — engine interface contract: fields, operators, universes, and portfolio/weighting options.
- `GET /api/examples` — starter alpha expressions.
- `GET /api/experiments` — recent notebook experiment-tracker entries.
- `POST /api/backtest` — legacy synchronous backtest endpoint; its existing response contract is retained.
- `POST /api/execution-jobs` — create a persisted execution record, run the backtest, and return its terminal job record.
- `GET /api/execution-jobs?limit=20` — list recent execution records. `limit` must be between 1 and 100.
- `GET /api/execution-jobs/<job_id>` — retrieve one persisted execution record by ID.
- `POST /api/generate-alpha` — generate candidate hypotheses and expressions with Gemini when configured.

### Execution-job lifecycle

The execution-job API persists a job ID, status, run configuration/fingerprint, timestamps, and either a result or error. Status values include `queued`, `running`, `succeeded`, and `failed` (the contract also defines `cancelled`). The current endpoint executes the backtest synchronously inside the HTTP request; the lifecycle record is persisted, but this is **not yet a background queue**.

The frontend's Runs & History page reads saved jobs and can inspect the result stored on a selected successful job. These records are stored in a local SQLite database by default at `backend/storage/execution_jobs.sqlite3`; override the path with `EXECUTION_JOB_DB`.

Example legacy backtest request:

```bash
curl -X POST http://127.0.0.1:3000/api/backtest \
  -H 'Content-Type: application/json' \
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

## Important limitations before multi-user deployment

- **Synchronous execution:** long backtests keep the HTTP request open and can occupy a web worker. A background queue/worker, polling strategy, timeouts, quotas, cancellation semantics, and bounded concurrency still need design and implementation.
- **Local persistence:** execution jobs use SQLite and the notebook experiment tracker uses a local JSON file. These are not shared multi-instance storage and do not provide tenant ownership or concurrent-writer guarantees for the experiment tracker.
- **Authentication and isolation:** authentication, per-user authorization, workspace isolation, managed secrets, request rate limiting, and an audit/retention policy are not implemented. Do not expose this API as a multi-user service until access controls and shared persistence are implemented.
- **Data availability:** regression/synthetic tests are not run automatically at API startup. A healthy HTTP response does not guarantee that live data is available for every universe/date range.
- **Preview behavior:** successful API runs display actual engine output. When the API is unavailable or a run fails, the frontend can show its labelled illustrative preview.
- **Research interpretation:** some operator semantics in the notebook metadata are marked assumed/unverified. Generated or backtested expressions are research hypotheses, not performance guarantees. Inspect data coverage, survivorship bias, point-in-time limitations, execution timing, costs, and out-of-sample robustness.

## Prompt-to-alpha with Gemini

The Create Alpha screen includes **Gemini Alpha Assistant**. A user can describe a strategy in plain English and receive candidate hypotheses/expressions with rationale, expected behavior, and risks. Generated candidates are checked against the engine's exposed field/operator registry and a restricted expression syntax before being shown. The checks do not establish runtime compatibility or profitability; backtest candidates through the same notebook-derived engine.

### Configure Gemini

1. Create a Gemini API key in Google AI Studio.
2. In `backend/`, copy `.env.example` to `.env` and set:

   ```env
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```

3. Install backend dependencies and restart `python app.py`.
4. Open Create Alpha, enter a strategy in the prompt box, and select **Generate alpha ideas**.

The key is read only by the Python backend. Do not put it in a `VITE_*` variable or commit `.env` to source control. The generation endpoint rejects blank/oversized prompts and validates returned expressions against the engine catalog. Similarity retrieval and response caching can reuse prior candidates to avoid some model calls; reuse is heuristic and still requires research validation.

### API request

`POST /api/generate-alpha`

```json
{"idea":"Find stocks with persistent momentum but penalize unusually high recent volatility."}
```

The response includes `ideas[]` and validation/provenance metadata. Missing Gemini configuration returns a `503` response.

The supplied notebooks `Review_Engine.ipynb` and `idea_to_expression_assistant_reference.ipynb` are retained as project references.
