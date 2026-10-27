# Unified Backtester — connected frontend + backend

This repository connects the React + Vite frontend to a Python Flask API that wraps the `alpha_backtesting_engine_production_hardened` notebook-derived engine. The notebook remains the source of truth for expression evaluation, portfolio construction, and backtest calculations.

## Project structure

- `frontend/` — React + Vite + Recharts UI and CSS.
- `backend/app.py` — HTTP API, request validation, result adaptation, alpha generation, and execution-job routes.
- `backend/alpha_engine.py` — notebook-derived expression engine, feature/operator registry, backtest runner, and interface metadata.
- `backend/backtest_service.py` — service boundary around the authoritative backtest execution call.
- `backend/execution_job_contract.py`, `backend/execution_job_store.py`, `backend/postgres_execution_job_store.py`, and `backend/execution_job_store_factory.py` — execution configuration/status contract and selectable SQLite/PostgreSQL persistence.
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

The engine resolves its default `data_cache/`, `universes/`, and `external_scores/` directories relative to `backend/alpha_engine.py`, so API and worker processes use the same locations even when launched from different working directories. The engine uses live Yahoo Finance data through `yfinance` by default. Install `pyarrow` (included in requirements) for the parquet cache. Required universe files must exist where the engine contract expects them; for example, `NIFTY100` requires `backend/universes/nifty100.csv`, and `TOP250`/`TOP500` require their candidate-pool CSVs. The engine does not silently substitute a different universe.

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
- `POST /api/execution-jobs` — validate and normalize the full backtest configuration before persisting it; valid requests return `202 Accepted` with a queued job record, while invalid requests return `400` without creating a job.
- `POST /api/execution-jobs/<job_id>/cancel` — cancel a job only while it is queued; returns `409` if it is already running or terminal.
- `GET /api/execution-jobs?limit=20` — list recent execution records. `limit` must be between 1 and 100.
- `GET /api/execution-jobs/<job_id>` — retrieve one persisted execution record by ID.
- `POST /api/generate-alpha` — generate candidate hypotheses and expressions with Gemini when configured.

### Execution-job lifecycle

The execution-job API persists a job ID, status, run configuration/fingerprint, timestamps, and either a result or error. Status values include `queued`, `running`, `succeeded`, and `failed` (the contract also defines `cancelled`). The HTTP endpoint returns immediately with a queued record. Run one or more worker processes using `python worker.py` from `backend/`; workers poll the shared job store, claim queued jobs through atomic state transitions, and persist results for the frontend to poll. Run the API and worker against the same database configuration. For multi-instance use, configure PostgreSQL; local SQLite is intended for development. Workers refresh each running job's `updated_at` heartbeat (default every 10 seconds). The worker periodically marks jobs failed if their heartbeat is older than `EXECUTION_JOB_STALE_MINUTES` (default 30 minutes, minimum 2). This avoids silently rerunning potentially expensive backtests; submit a failed job again to retry. Tune the timeout above expected scheduling pauses and transient database interruptions. If a worker is paused long enough to miss the threshold, a still-running job may be marked failed; use a generous timeout and monitor worker logs.

The frontend's Runs & History page reads saved jobs and can inspect the result stored on a selected successful job. These records use local SQLite by default at `backend/storage/execution_jobs.sqlite3`; override the path with `EXECUTION_JOB_DB`. To opt into shared PostgreSQL storage, set `EXECUTION_JOB_DATABASE_URL` to a PostgreSQL connection URL. When this variable is set, PostgreSQL is selected instead of SQLite; connection/schema errors fail startup rather than silently falling back. The database user needs permission to create the `execution_jobs` table and its index on first startup.

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

- **Background execution:** long backtests run in separate worker processes, not the Flask request. The frontend polls job status. Configure worker process count deliberately; concurrency limits, stopping an already-running backtest, per-user quotas, and request timeouts remain future work. Queued jobs can be cancelled before a worker claims them; abandoned running jobs are marked failed after a configurable heartbeat timeout rather than automatically rerun.
- **Persistence:** execution jobs can use PostgreSQL when `EXECUTION_JOB_DATABASE_URL` is configured, or local SQLite by default. The notebook experiment tracker still uses a local JSON file, so it is not shared across instances and does not provide tenant ownership or concurrent-writer guarantees. PostgreSQL job storage alone does not make the full application multi-instance safe.
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

The notebook-derived backtesting engine is exported into `backend/alpha_engine.py`. The original production notebooks are not currently committed in this repository; end-to-end parity with the production Idea→Expression notebook cannot be independently verified from this checkout.

## NIFTY 50 universe setup

The engine can use a bundled NIFTY 50 fallback list when `backend/universes/nifty50.csv` is absent. That fallback is explicitly not guaranteed current. For a reproducible run, refresh the current constituent snapshot from the official NSE Indices CSV before backtesting:

```bash
python backend/refresh_nifty50.py
```

This writes `backend/universes/nifty50.csv` and `nifty50.metadata.json`. The refresh script validates exactly 50 unique symbols and writes atomically only after validation succeeds. This is a **current-membership snapshot**, not point-in-time historical membership; historical runs can therefore have survivorship/reconstitution bias. The engine still allows its fallback when no CSV exists, so production/reproducible runs should execute this refresh step and verify the resulting metadata.

## NIFTY 100 universe setup

The engine requires an explicit constituents file for NIFTY 100 and deliberately does not silently substitute NIFTY 50. Bootstrap or refresh the current constituent snapshot from the official NSE Indices CSV before running NIFTY 100 backtests:

```bash
python backend/refresh_nifty100.py
```

This writes `backend/universes/nifty100.csv` and a source/checksum record in `nifty100.metadata.json`. The file is a **current-membership snapshot**, not a point-in-time historical universe; historical backtests using it carry survivorship/reconstitution bias unless you provide historical constituents separately. The script validates exactly 100 unique symbols and fails without replacing the existing file if the source response is invalid.

## Optional local retrieval models

Idea generation works without extra model downloads. By default, retrieval uses deterministic text normalization and hashed local vectors. For stronger local summarization and semantic embeddings, install the optional dependencies and pre-cache compatible models yourself. The backend never downloads models at request time.

### Local summarizer

```bash
python -m pip install transformers torch
hf download google/flan-t5-small --local-dir /models/flan-t5-small
```

Set `LOCAL_SUMMARIZER_MODEL=/models/flan-t5-small` in the backend environment and restart the backend. The model is used only to form a retrieval query; the full research brief is still passed to the generation prompt and exact-cache key. If the path or dependency is unavailable, the backend falls back to deterministic normalization.

### Local semantic embeddings

```bash
python -m pip install sentence-transformers
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir /models/all-MiniLM-L6-v2
```

Set `ALPHA_EMBEDDING_MODEL=/models/all-MiniLM-L6-v2` and restart the backend. The same local embedding model powers the SQLite vector store for prior candidate sets and the active data/operator catalog. If no usable local model is configured, the store uses deterministic hashed lexical vectors instead. No external embedding API is called.

These models run locally on CPU by default and consume additional memory/latency. Enable them only where the deployment has enough resources. The API key for Gemini remains a backend-only environment secret and is not used by the local summarizer or embedding store.


## Model-call reliability for Idea → Expression

Remote alpha generation now goes through `backend/generation_gateway.py`, which enforces a bounded request timeout, caps provider response size, and retries only likely transient network/rate-limit/server failures once. It does not log prompts, credentials, or response bodies. A retry after a timeout can still incur a provider charge if the provider accepted the original request.

Optional backend environment settings:

- `ALPHA_MODEL_TIMEOUT_MS`: request timeout in milliseconds (default `30000`; accepted range 1–120000).
- `ALPHA_MODEL_RETRIES`: currently bounded to at most one retry; default is `1`.

This is one production-hardening layer, not a claim that the whole Idea → Expression flow is production complete. The pipeline still requires integration tests against the real expression engine and data adapter, operational rate limiting/authentication for public deployments, and parity review against the production notebook once that notebook is available.


## Candidate response contract

The Idea → Expression endpoint now rejects malformed/non-object JSON payloads and provider responses that exceed the candidate-count or response-size bounds. Each candidate must pass the existing allowlisted payload compaction, registered-field/operator validation, and deterministic synthetic-data runtime probe. Exact duplicate expressions within one model response are rejected rather than returned as separate candidates.

A synthetic runtime probe is only a compatibility smoke test. It is not a substitute for a backtest on the requested NIFTY universe, historical prices, and out-of-sample dates.
