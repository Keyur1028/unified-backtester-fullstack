# Architecture and Integration Contract

This document records the current boundaries of Unified Backtester and the rules future changes should preserve. It describes the repository as it exists today; items marked **planned** are not implemented.

## 1. Current system boundaries

```text
React + Vite UI (frontend/)
        |
        | /api/* via Vite proxy or VITE_API_URL
        v
Flask HTTP/API layer (backend/app.py)
        |
        | validates requests, invokes engine, adapts results,
        | loads/saves the local experiment tracker
        v
Notebook-derived research engine (backend/alpha_engine.py)
        |
        +-- expression parser / field and operator registry
        +-- universe and market-data loading / cache
        +-- portfolio construction and backtest execution
        +-- result and experiment tracking
```

- **Frontend:** owns presentation, user input, and rendering of API results. It must not reimplement backtest calculations.
- **Flask API:** owns HTTP validation, error responses, orchestration, result-shape adaptation, and the current local persistence integration.
- **Research engine:** is the source of truth for expression semantics, supported fields/operators, universe behavior, execution assumptions, and backtest metrics. Keep engine logic in this layer rather than duplicating it in routes or React components.
- **Notebook:** the production engine was exported from the hardened notebook. Treat the notebook-derived engine as the executable implementation; update its interface deliberately if the notebook contract changes.
- **External services:** Yahoo Finance provides live market data when configured and reachable; Gemini is an optional external dependency for idea generation. Neither should be required for syntax/build-only CI.

## 2. Current API surface

All routes are under `/api`.

| Method and route | Purpose | Contract notes |
| --- | --- | --- |
| `GET /api/health` | Basic service/engine status | Does not prove every market-data request or backtest will succeed. |
| `GET /api/meta` | Engine interface metadata | Derive available fields, operators, universes, portfolio/weighting options and execution settings from `engine.describe_interface()`; do not maintain a second hard-coded registry in the UI. |
| `GET /api/examples` | Starter expressions | Examples must remain compatible with the engine registry. |
| `GET /api/experiments` | Experiment history | Currently backed by the engine tracker and local JSON snapshot. |
| `POST /api/backtest` | Validate and execute a backtest | Uses the engine's `run_backtest_api()`, adapts the result for the UI, and persists the tracker snapshot. |
| `POST /api/generate-alpha` | Generate candidate expressions from a natural-language idea | Uses Gemini when configured; candidates are checked against registered fields/operators and restricted expression syntax. This is not a performance or profitability validation. |

### Backtest request

The current request uses these fields where applicable:

- `expression`: expression-language string supported by the engine.
- `universe`, `start`, `end`: requested universe and date range.
- `holding_period`, `portfolio`, `weighting`: construction settings.
- `transaction_cost_bps`, `slippage_bps`: modeled costs.
- `execution_lag_bars`, `execution_price`: execution assumptions.

The engine/API metadata remains authoritative for accepted values and defaults. Validate at the API boundary and preserve the engine's own runtime validation.

### Error and result behavior

- Keep errors machine-readable and actionable; do not silently substitute an unsupported universe, field, or operator.
- A successful candidate-generation response means only that candidates passed the current syntax/registry checks. Runtime compatibility and research quality require backtesting and robustness analysis.
- The API adapter may convert the engine's dated dictionaries/series into chart-friendly arrays, but must not change the underlying metric definitions.
- Preserve warnings about point-in-time membership, survivorship bias, data coverage, and execution assumptions when returning results.

## 3. Data, persistence, and runtime constraints

Current behavior—not a claim of production readiness:

- Backtests execute synchronously in the Flask request.
- Experiment history is stored through the engine tracker and a local JSON file under `backend/storage/`; this is not a multi-user database and has no tenant isolation or multi-process write guarantees.
- Market data and universe files are resolved according to the engine's configured paths. Missing required universe files should result in a clear failure rather than an unannounced proxy universe.
- Gemini credentials belong on the backend only, loaded from environment configuration. Never expose provider keys through frontend variables or commit secrets.
- Authentication/authorization, durable multi-user persistence, job queuing, concurrency limits, quotas, and production deployment hardening are not currently implemented.

## 4. Change rules

For each change, keep the diff narrow and preserve these rules:

1. **One source of truth:** engine metadata and calculations come from `alpha_engine.py`; do not duplicate operator lists or financial calculations in the frontend.
2. **Stable API boundary:** when changing request/response fields, update the API route, frontend API consumer, documentation, and tests in the same PR.
3. **No silent data substitution:** do not replace missing symbols, dates, universes, or fields with unrelated data without an explicit user-visible warning.
4. **No false validation claims:** syntax checks, registry checks, and a successful HTTP response are not proof of alpha quality or full live-data correctness.
5. **Secrets and privacy:** no API keys in browser code, logs, fixtures, or committed environment files. Add authentication and ownership checks before presenting stored experiments as private user data.
6. **Keep slow work out of UI assumptions:** the current API is synchronous. If backtests move to background jobs, define job states and polling/cancellation contracts before changing the UI.
7. **Keep CI deterministic:** CI should test syntax/build and offline tests without requiring Gemini credentials, network market data, or a live provider account.

## 5. Recommended implementation sequence

These are proposed next steps, not current capabilities:

1. Add offline backend contract tests for health/meta, malformed requests, expression validation, and result adaptation using synthetic fixtures.
2. Add cache-first idea-to-expression generation: normalize the user's brief into a structured hypothesis, retrieve only from the engine's supported field/operator catalog, check exact/near-duplicate prior candidates, and call Gemini only on a cache miss. Keep the existing generation endpoint contract stable initially.
3. Add deterministic validation and provenance for generated candidates (catalog/version, cache hit or model, model identifier, and validation outcomes).
4. Design durable experiment persistence and user/workspace ownership before adding authentication-dependent UI behavior.
5. Introduce background jobs and bounded concurrency before relying on synchronous backtests for multiple users.

Do not implement future steps implicitly as part of a documentation-only change.
