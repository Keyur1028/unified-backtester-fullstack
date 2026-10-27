from __future__ import annotations
import os, time, json, ast
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass
import alpha_engine as engine
from idea_cache import make_cache_key, read_cached_response, write_cached_response
from candidate_validation import VALIDATOR_VERSION, catalog_fingerprint, validate_expression
from similarity_retrieval import idea_tokens, retrieve_similar, store_idea_response
from idea_similarity_safety import has_directional_conflict
from retrieval_context import format_retrieval_context
from candidate_payload import compact_generated_candidate
from candidate_runtime_validation import validate_candidate_runtime
from research_context import validate_research_context
from alpha_generation_service import build_generation_prompt, generate_candidates
from backtest_service import run_backtest
from execution_job_contract import ExecutionJobStatus, RunConfiguration
from execution_job_store import ExecutionJobNotFoundError, InvalidJobTransitionError
from execution_job_store_factory import create_execution_job_store

BASE = Path(__file__).resolve().parent
STORE = BASE / "storage"
STORE.mkdir(parents=True, exist_ok=True)
TRACKER_FILE = STORE / "experiment_log.json"
JOB_DATABASE = os.getenv("EXECUTION_JOB_DB", str(STORE / "execution_jobs.sqlite3"))
JOB_DATABASE_URL = os.getenv("EXECUTION_JOB_DATABASE_URL", "").strip()
JOB_STORE = create_execution_job_store(JOB_DATABASE_URL, JOB_DATABASE)
try:
    if TRACKER_FILE.exists():
        engine.TRACKER.load(str(TRACKER_FILE), replace=True)
except Exception as exc:
    print(f"[tracker] previous log not loaded: {exc}")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
CORS(app, resources={r"/api/*": {"origins": os.getenv("CORS_ORIGINS", "*").split(",")}})

def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def valid_cached_ideas(ideas, fields, operators):
    """Fail closed when cached candidate expressions no longer validate."""
    if not isinstance(ideas, list) or not ideas:
        return []
    accepted = []
    for item in ideas:
        if not isinstance(item, dict):
            return []
        expression = str(item.get("expression", "")).strip()
        try:
            valid, _reason = validate_expression(expression, fields, operators)
        except Exception:
            return []
        if not valid:
            return []
        candidate = dict(item)
        candidate["expression"] = expression
        accepted.append(candidate)
    return accepted


def pairs(raw):
    if not isinstance(raw, dict): return []
    out = []
    for k, v in raw.items():
        try: out.append((pd.Timestamp(k), num(v)))
        except Exception: pass
    return sorted(out, key=lambda x: x[0])

def frame_pairs(raw, column):
    if not isinstance(raw, dict): return []
    return sorted([(pd.Timestamp(d), num(row.get(column))) for d, row in raw.items()
                   if isinstance(row, dict)], key=lambda x: x[0])

def adapt_result(envelope, elapsed):
    r = envelope["result"]
    netp, grossp = pairs(r.get("net_returns")), pairs(r.get("gross_returns"))
    dates = [d.strftime("%Y-%m-%d") for d, _ in netp]
    net = pd.Series([v if v is not None else np.nan for _, v in netp],
                    index=pd.DatetimeIndex([d for d, _ in netp]), dtype=float)
    icp, ddp, turnp = pairs(r.get("ic_series")), pairs(r.get("drawdown_series")), pairs(r.get("turnover_series"))
    cnet, cgross = pairs(r.get("cum_net_returns")), pairs(r.get("cum_gross_returns"))
    def aligned(ps, transform=lambda v: v):
        m = {d.strftime("%Y-%m-%d"): (transform(v) if v is not None else None) for d, v in ps}
        return [m.get(d) for d in dates]
    roll = net.rolling(60, min_periods=20).apply(
        lambda x: np.nanmean(x) / np.nanstd(x, ddof=1) * np.sqrt(252)
        if np.nanstd(x, ddof=1) > 0 else np.nan, raw=True)
    roll_map = {d.strftime("%Y-%m-%d"): num(v) for d, v in roll.items()}
    ic_series = pd.Series({d: v for d, v in icp if v is not None}).sort_index()
    ic_roll = ic_series.rolling(20, min_periods=1).mean()
    ic_roll_map = {pd.Timestamp(d).strftime("%Y-%m-%d"): num(v) for d, v in ic_roll.items()}
    ex = r.get("exposures") or {}
    longp, shortp, netexp = (frame_pairs(ex, "long_exposure"),
                             frame_pairs(ex, "short_exposure"),
                             frame_pairs(ex, "net_exposure"))
    yearly = []
    for year, yr in net.groupby(net.index.year):
        yr = yr.dropna()
        if yr.empty: continue
        cum = (1 + yr).cumprod()
        dd = cum / cum.cummax() - 1
        vol = yr.std(ddof=1) * np.sqrt(252) if len(yr) > 1 else np.nan
        sharpe = yr.mean() / yr.std(ddof=1) * np.sqrt(252) if len(yr)>1 and yr.std(ddof=1)>0 else np.nan
        iv = [v for d,v in icp if d.year == year and v is not None]
        tv = [v for d,v in turnp if d.year == year and v is not None]
        yearly.append({"year": int(year), "return": num((cum.iloc[-1]-1)*100),
                       "sharpe": num(sharpe), "volatility": num(vol*100),
                       "maxDD": num(dd.min()*100), "turnover": num(np.mean(tv)*252) if tv else None,
                       "ic": num(np.mean(iv)) if iv else None, "days": int(len(yr))})
    qraw = r.get("quantile_summary") or {}
    quantiles = []
    for b, row in qraw.items():
        if isinstance(row, dict):
            try: bi = int(float(b)); label = f"Q{bi+1}"
            except Exception: label = str(b)
            quantiles.append({"bucket": label, "avg_return": num(row.get("avg_return")),
                              "cumulative_return": num(row.get("cumulative_return"))})
    quantiles.sort(key=lambda x: x["bucket"])
    meta = r.get("universe_metadata") or {}
    coverage = r.get("coverage") or {}
    warnings = []
    if r.get("survivorship_bias_risk"): warnings.append("Universe has survivorship-bias risk; inspect universe metadata.")
    if not meta.get("point_in_time", False): warnings.append("Point-in-time universe membership is not verified for this run.")
    if coverage.get("min_coverage") is not None and coverage["min_coverage"] < .7:
        warnings.append("Minimum cross-sectional signal coverage fell below 70%.")
    if not warnings: warnings.append("Review data freshness, universe methodology and execution assumptions before interpreting results.")
    return {
      "experiment_id": envelope.get("experiment_id"), "status": "success", "elapsed_s": round(elapsed, 3),
      "data_mode": "live-provider",
      "config": {"expression": r.get("expression"), "universe": r.get("universe"),
                 "start": r.get("start_date"), "end": r.get("end_date"),
                 "holding_period": r.get("holding_period"), "portfolio": r.get("portfolio_type"),
                 "weighting": r.get("weighting")},
      "summary": {
        "annual_return_net": num(r.get("annual_return_net")), "annual_return_gross": num(r.get("annual_return_gross")),
        "sharpe_net": num(r.get("sharpe_net")), "sharpe_gross": num(r.get("sharpe_gross")),
        "sortino": num(r.get("sortino")), "calmar": num(r.get("calmar")),
        "max_drawdown": num(r.get("max_drawdown")), "turnover_annual": num(r.get("turnover_annual")),
        "ic_mean": num(r.get("ic_mean")), "ic_ir": num(r.get("ic_ir")), "fitness": None,
        "coverage_avg": num(coverage.get("avg_coverage")), "win_rate": num(r.get("win_rate")),
        "profit_factor": num(r.get("profit_factor"))},
      "benchmark": r.get("benchmark") or {}, "warnings": warnings,
      "universe": {"name": r.get("universe"), "construction": meta.get("construction_method", "Preset"),
                   "point_in_time": bool(meta.get("point_in_time", False)),
                   "survivorship_bias_risk": bool(r.get("survivorship_bias_risk")),
                   "n_assets": meta.get("n_assets") or meta.get("asset_count")},
      "series": {"dates": dates, "cum_gross": aligned(cgross, lambda v:v-1),
                 "cum_net": aligned(cnet, lambda v:v-1), "drawdown": aligned(ddp),
                 "rolling_sharpe": [roll_map.get(d) for d in dates],
                 "turnover": aligned(turnp), "long": aligned(longp),
                 "short": aligned(shortp), "net": aligned(netexp)},
      "ic": {"dates": [d.strftime("%Y-%m-%d") for d,_ in icp], "values": [v for _,v in icp],
             "rolling": [ic_roll_map.get(d.strftime("%Y-%m-%d")) for d,_ in icp]},
      "quantiles": quantiles, "yearly": yearly,
      "raw_summary": {"annual_vol": num(r.get("annual_vol")), "turnover_mean": num(r.get("turnover_mean")),
                      "factor_exposures": r.get("factor_exposures") or {}, "metadata": r.get("metadata") or {}}
    }

@app.post("/api/generate-alpha")
def generate_alpha_ideas():
    """Turn a plain-language strategy brief into validated candidate expressions."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": {"type": "ValidationError", "message": "Request body must be a JSON object."}}), 400
    idea = str(payload.get("idea", "")).strip()
    if len(idea) < 20:
        return jsonify({"error": {"type": "ValidationError", "message": "Describe the strategy in at least 20 characters, including the behavior you want to capture."}}), 400
    if len(idea) > 5000:
        return jsonify({"error": {"type": "ValidationError", "message": "Strategy prompt must be 5,000 characters or fewer."}}), 400

    # Optional structured fields from the research-idea notebook. Bound them before
    # using them in model prompts, cache keys, or retrieval features.
    rationale_for_data_used = payload.get("rationale_for_data_used", "")
    rationale_for_operators_used = payload.get("rationale_for_operators_used", "")
    strategy_context = payload.get("strategy_context", {})
    for field_name, value in (
        ("rationale_for_data_used", rationale_for_data_used),
        ("rationale_for_operators_used", rationale_for_operators_used),
    ):
        if not isinstance(value, str):
            return jsonify({"error": {"type": "ValidationError", "message": f"{field_name} must be a string."}}), 400
        if len(value) > 2000:
            return jsonify({"error": {"type": "ValidationError", "message": f"{field_name} must be 2,000 characters or fewer."}}), 400
    if not isinstance(strategy_context, dict):
        return jsonify({"error": {"type": "ValidationError", "message": "strategy_context must be a JSON object."}}), 400
    try:
        strategy_context_json = json.dumps(strategy_context, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return jsonify({"error": {"type": "ValidationError", "message": "strategy_context must contain JSON-compatible values."}}), 400
    if len(strategy_context_json) > 2000:
        return jsonify({"error": {"type": "ValidationError", "message": "strategy_context must be 2,000 characters or fewer."}}), 400

    capability_interface = engine.describe_interface()
    normalized_context, context_errors, unsupported_context = validate_research_context(
        payload, strategy_context, capability_interface
    )
    if context_errors:
        return jsonify({"error": {"type": "ValidationError", "message": "Invalid research context.", "details": context_errors}}), 400
    if unsupported_context:
        return jsonify({
            "status": "unsupported",
            "assistant_status": "unsupported",
            "alpha_expression": None,
            "backtest_executed": False,
            "warnings": ["D0 intraday is not supported by the current daily-bar engine."],
            "error": {"type": "UnsupportedStrategyContext", "message": "The production engine does not support this strategy context."},
        }), 422
    strategy_context = normalized_context["strategy_context"]
    strategy_context_json = json.dumps(normalized_context, sort_keys=True, ensure_ascii=False)

    research_context = {
        "rationale_for_data_used": rationale_for_data_used.strip(),
        "rationale_for_operators_used": rationale_for_operators_used.strip(),
        **normalized_context,
    }
    research_brief = idea + "\nResearch context: " + strategy_context_json
    if research_context["rationale_for_data_used"]:
        research_brief += "\nRationale for data used: " + research_context["rationale_for_data_used"]
    if research_context["rationale_for_operators_used"]:
        research_brief += "\nRationale for operators used: " + research_context["rationale_for_operators_used"]
    if strategy_context:
        research_brief += "\nStrategy context: " + strategy_context_json
    fields_info = engine.describe_interface().get("fields", {})
    fields = set()
    for value in fields_info.values():
        if isinstance(value, list): fields.update(str(x) for x in value)
        elif isinstance(value, dict): fields.update(str(x) for x in value.keys())
    operators = set(engine.OPERATORS) | set(engine.GROUP_OPERATORS)
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    cache_key = make_cache_key(research_brief, sorted(fields), sorted(operators), model)
    cache_path = STORE / "idea_generation_cache.json"
    cached = read_cached_response(cache_path, cache_key)
    if cached is not None:
        cached_ideas = valid_cached_ideas(cached.get("ideas"), fields, operators)
        if cached_ideas:
            runtime_ok = True
            for candidate in cached_ideas:
                passed, _reason = validate_candidate_runtime(engine, candidate["expression"], fields)
                if not passed:
                    runtime_ok = False
                    break
                candidate["runtime_validation"] = "synthetic-data-probe-pass"
            if runtime_ok:
                cached["ideas"] = cached_ideas
                cached["provenance"] = {
                    "validator": VALIDATOR_VERSION,
                    "catalog_fingerprint": catalog_fingerprint(fields, operators),
                    "scope": "static-syntax-registry-and-synthetic-runtime-probe",
                }
                cached["cache"] = {"hit": True, "strategy": "exact-match"}
                return jsonify(cached)
            app.logger.warning("Ignoring exact-match alpha cache entry that failed runtime preflight")
        app.logger.warning("Ignoring invalid exact-match alpha cache entry")

    catalog_id = catalog_fingerprint(fields, operators)
    retrieval_path = STORE / "idea_retrieval_index.json"
    similar_ideas = retrieve_similar(
        retrieval_path, research_brief, model, catalog_id, limit=3, min_similarity=0.25
    )

    # Reuse a prior validated candidate set only for a very close lexical match.
    # Require enough query terms to reduce false positives from short generic ideas.
    query_tokens = idea_tokens(idea)
    near_duplicate = next(
        (
            row for row in similar_ideas
            if len(query_tokens) >= 5 and row.get("similarity", 0) >= 0.92
            and not has_directional_conflict(query_tokens, row.get("matched_tokens", []))
            and isinstance(row.get("ideas"), list) and row["ideas"]
        ),
        None,
    )
    if near_duplicate is not None:
        reusable_ideas = valid_cached_ideas(near_duplicate["ideas"], fields, operators)
        if reusable_ideas:
            runtime_checked = []
            for candidate in reusable_ideas:
                passed, _reason = validate_candidate_runtime(engine, candidate["expression"], fields)
                if not passed:
                    runtime_checked = []
                    break
                candidate["runtime_validation"] = "synthetic-data-probe-pass"
                runtime_checked.append(candidate)
            reusable_ideas = runtime_checked
        if reusable_ideas:
            result = {
                "ideas": reusable_ideas,
                "rejected": [],
                "model": model,
                "provenance": {
                    "validator": VALIDATOR_VERSION,
                    "catalog_fingerprint": catalog_id,
                    "scope": "static-syntax-registry-and-synthetic-runtime-probe",
                },
                "note": (
                    "Reused candidates from a highly similar prior prompt to avoid a model call. "
                    "Directional-conflict checks are heuristic and lexical similarity is not semantic; "
                    "verify the expressions and run backtests "
                    "before interpreting their research value."
                ),
                "retrieval": {
                    "strategy": "local-lexical-cosine",
                    "matches_used": 1,
                    "similarity": near_duplicate["similarity"],
                    "raw_prompt_persisted": False,
                },
                "cache": {"hit": True, "strategy": "similarity"},
            }
            try:
                write_cached_response(cache_path, cache_key, result)
            except OSError:
                app.logger.warning("Could not persist similarity cache hit", exc_info=True)
            return jsonify(result)
        app.logger.warning("Ignoring invalid similar-alpha candidates")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": {"type": "ConfigurationError", "message": "Gemini is not configured. Add GEMINI_API_KEY to backend/.env and restart the backend."}}), 503
    prompt = build_generation_prompt(idea, fields, operators, similar_ideas, research_context=research_context)
    try:
        response_text = generate_candidates(api_key, model, prompt)
        raw = json.loads(response_text)

        ideas = raw.get("ideas", []) if isinstance(raw, dict) else []
        accepted, rejected = [], []
        for item in ideas[:5]:
            candidate, compact_error = compact_generated_candidate(item)
            if candidate is None:
                rejected.append({
                    "expression": str(item.get("expression", ""))[:1000] if isinstance(item, dict) else "",
                    "reason": compact_error or "Invalid candidate payload",
                })
                continue
            expr = candidate["expression"]
            try:
                valid, reason = validate_expression(expr, fields, operators)
                if not valid:
                    raise ValueError(reason or "Expression validation failed")
                runtime_ok, runtime_reason = validate_candidate_runtime(engine, expr, fields)
                if not runtime_ok:
                    raise ValueError("Synthetic runtime probe failed: " + (runtime_reason or "unknown error"))
                candidate["id"] = "idea-" + str(len(accepted) + 1)
                candidate["validation"] = "syntax-and-registry-pass"
                candidate["runtime_validation"] = "synthetic-data-probe-pass"
                candidate["provenance"] = {
                    "validator": VALIDATOR_VERSION,
                    "catalog_fingerprint": catalog_fingerprint(fields, operators),
                    "scope": "static-syntax-registry-and-synthetic-runtime-probe",
                }
                accepted.append(candidate)
            except Exception as exc:
                rejected.append({"expression": expr, "reason": str(exc)[:250]})
        if not accepted:
            return jsonify({"error": {"type": "GenerationValidationError", "message": "Gemini returned no candidates that passed engine field/operator checks. Try a more specific OHLCV-based prompt.", "rejected": rejected}}), 422
        result = {"ideas": accepted, "rejected": rejected, "model": model,
                  "provenance": {"validator": VALIDATOR_VERSION,
                                 "catalog_fingerprint": catalog_fingerprint(fields, operators),
                                 "scope": "static-syntax-registry-and-synthetic-runtime-probe"},
                  "note": "Candidates passed syntax, registry, and a deterministic synthetic-data runtime probe. Real-data compatibility, data coverage, and out-of-sample robustness still require an actual backtest."}
        result["retrieval"] = {
            "strategy": "local-lexical-cosine",
            "matches_used": len(similar_ideas),
            "raw_prompt_persisted": False,
        }
        try:
            write_cached_response(cache_path, cache_key, result)
        except OSError:
            app.logger.warning("Could not persist idea-generation cache", exc_info=True)
        try:
            store_idea_response(retrieval_path, idea, model, catalog_id, result)
        except OSError:
            app.logger.warning("Could not persist idea retrieval index", exc_info=True)
        result["cache"] = {"hit": False, "strategy": "exact-match"}
        return jsonify(result)
    except ImportError:
        return jsonify({"error": {"type": "DependencyError", "message": "Install backend dependencies with: pip install -r requirements.txt (includes google-genai)."}}), 503
    except Exception:
        app.logger.exception("Gemini alpha generation failed")
        return jsonify({"error": {"type": "GeminiError", "message": "Gemini generation failed. Check the API key, model availability, quota, and backend logs."}}), 502

@app.get("/api/health")
def health():
    return jsonify({"status":"ok", "engine":"alpha_backtesting_engine_production_hardened",
                    "data_mode":"live-provider", "yfinance_available":engine.yf is not None,
                    "tracker_count":engine.TRACKER.count()})

@app.get("/api/meta")
def metadata():
    i = engine.describe_interface()
    f, ops = i.get("fields",{}), i.get("operators",{})
    return jsonify({"universes":list(i.get("universes",{}).keys()),
      "portfolio_types":i.get("portfolio_types",[]), "weighting_schemes":i.get("weighting_schemes",[]),
      "execution_settings":i.get("execution_settings",{}),
      "fields":{"native":f.get("native_ohlcv",[]), "engineered":f.get("engineered",[]), "external":f.get("external","")},
      "operators":ops.get("elementwise_and_timeseries",[])+ops.get("group_aware",[]),
      "expression_syntax":i.get("expression_syntax"), "universe_details":i.get("universes",{})})

@app.get("/api/examples")
def examples():
    return jsonify([
      {"name":"Momentum z-score","expression":"rank(ts_zscore(momentum_20, 20))","category":"Momentum"},
      {"name":"Short-term reversal","expression":"-rank(ts_delta(close, 5))","category":"Reversal"},
      {"name":"Volume surge","expression":"rank(ts_zscore(volume_ratio_20, 20)) * sign(momentum_5)","category":"Volume"},
      {"name":"RSI mean reversion","expression":"-rank(rsi_14)","category":"Mean reversion"},
      {"name":"Trend vs 50d average","expression":"rank(close / sma_50 - 1)","category":"Trend"}])

@app.get("/api/experiments")
def experiments():
    rows = engine.TRACKER.to_dict()
    return jsonify([{"experiment_id":x.get("experiment_id") or x.get("id"),
      "status":x.get("status","success"), "expression":x.get("expression"),
      "created_at":x.get("at"), "config":x.get("config",{})} for x in rows[-100:][::-1]])

@app.post("/api/backtest")
def backtest():
    if not request.is_json:
        return jsonify({"error":{"type":"ValidationError","message":"Content-Type must be application/json"}}),415
    payload = request.get_json(silent=True)
    if not isinstance(payload,dict):
        return jsonify({"error":{"type":"ValidationError","message":"Request body must be a JSON object"}}),400
    started = time.perf_counter()
    envelope = run_backtest(payload)
    elapsed = time.perf_counter()-started
    try: engine.TRACKER.save(str(TRACKER_FILE))
    except Exception as exc: app.logger.warning("Tracker persistence failed: %s",exc)
    if envelope.get("status") != "success":
        err = envelope.get("error") or {}
        return jsonify({"experiment_id":envelope.get("experiment_id"),"status":"error",
                        "error":err,"config":envelope.get("config")}), (400 if err.get("type")=="ValidationError" else 422)
    return jsonify(adapt_result(envelope,elapsed))

def execute_execution_job(job_id: str) -> dict[str, Any]:
    """Execute one queued job and persist a terminal result.

    Safe for worker retries: only a queued job can transition to running, so
    multiple workers cannot execute the same record concurrently.
    """
    job = JOB_STORE.get_job(job_id)
    JOB_STORE.transition(job_id, ExecutionJobStatus.RUNNING)
    started = time.perf_counter()
    try:
        envelope = run_backtest(job["config"])
        elapsed = time.perf_counter() - started
        try:
            engine.TRACKER.save(str(TRACKER_FILE))
        except Exception as exc:
            app.logger.warning("Tracker persistence failed: %s", exc)

        if not isinstance(envelope, dict) or envelope.get("status") != "success":
            error = (envelope.get("error") if isinstance(envelope, dict) else None) or {
                "type": "ExecutionError", "message": "Backtest execution failed."
            }
            return JOB_STORE.transition(job_id, ExecutionJobStatus.FAILED, error=error)

        result = adapt_result(envelope, elapsed)
        return JOB_STORE.transition(job_id, ExecutionJobStatus.SUCCEEDED, result=result)
    except Exception:
        app.logger.exception("Execution job failed")
        try:
            return JOB_STORE.transition(
                job_id, ExecutionJobStatus.FAILED,
                error={"type": "InternalError", "message": "Unexpected backtest execution failure."},
            )
        except Exception:
            app.logger.exception("Could not persist failed execution-job status")
            raise


@app.post("/api/execution-jobs")
def create_execution_job():
    """Persist a run request and return immediately for background processing."""
    if not request.is_json:
        return jsonify({"error": {"type": "ValidationError", "message": "Content-Type must be application/json"}}), 415
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": {"type": "ValidationError", "message": "Request body must be a JSON object"}}), 400

    try:
        # Reject malformed configurations before they consume queue/storage capacity.
        validated_payload = engine.validate_backtest_request(payload)

        fields_info = engine.describe_interface().get("fields", {})
        fields = set()
        for value in fields_info.values():
            if isinstance(value, list):
                fields.update(str(item) for item in value)
            elif isinstance(value, dict):
                fields.update(str(item) for item in value.keys())
        operators = set(engine.OPERATORS) | set(engine.GROUP_OPERATORS)
        expression_valid, expression_error = validate_expression(
            validated_payload["expression"], fields, operators
        )
        if not expression_valid:
            return jsonify({
                "error": {
                    "type": "ValidationError",
                    "message": expression_error or "Expression validation failed.",
                }
            }), 400

        configuration = RunConfiguration.from_payload(validated_payload)
    except (TypeError, ValueError) as exc:
        return jsonify({
            "error": {
                "type": "ValidationError",
                "message": str(exc)[:500] or "Run configuration is invalid.",
            }
        }), 400

    job = JOB_STORE.create_job(configuration)
    return jsonify(job), 202


@app.get("/api/execution-jobs")
def list_execution_jobs():
    """List recent persisted execution jobs; page size is limited to 100."""
    raw_limit = request.args.get("limit", "20")
    try:
        limit = int(raw_limit)
        jobs = JOB_STORE.list_jobs(limit=limit)
    except (TypeError, ValueError):
        return jsonify({
            "error": {
                "type": "ValidationError",
                "message": "limit must be an integer between 1 and 100.",
            }
        }), 400
    # Result time series can be large. Keep list/polling responses lightweight and
    # fetch the complete immutable result from GET /api/execution-jobs/<job_id>.
    summaries = []
    for job in jobs:
        summary = {key: value for key, value in job.items() if key != "result"}
        summary["result_available"] = job.get("result") is not None
        summaries.append(summary)
    return jsonify({"jobs": summaries, "count": len(summaries), "limit": limit}), 200


@app.post("/api/execution-jobs/<job_id>/cancel")
def cancel_execution_job(job_id):
    """Cancel a queued job before a worker claims it; running jobs cannot be stopped."""
    try:
        job = JOB_STORE.transition(job_id, ExecutionJobStatus.CANCELLED)
        return jsonify(job), 200
    except ExecutionJobNotFoundError:
        return jsonify({
            "error": {"type": "NotFound", "message": "Execution job was not found."}
        }), 404
    except InvalidJobTransitionError:
        return jsonify({
            "error": {
                "type": "InvalidJobState",
                "message": "Only queued execution jobs can be cancelled; this job may already be running or finished.",
            }
        }), 409


@app.get("/api/execution-jobs/<job_id>")
def get_execution_job(job_id):
    """Fetch a persisted execution-job record by identifier."""
    try:
        return jsonify(JOB_STORE.get_job(job_id)), 200
    except ExecutionJobNotFoundError:
        return jsonify({
            "error": {"type": "NotFound", "message": "Execution job was not found."}
        }), 404


@app.errorhandler(413)
def too_large(_err):
    return jsonify({"error":{"type":"RequestTooLarge","message":"Request body exceeds 256 KB."}}),413

@app.errorhandler(500)
def internal_error(_err):
    app.logger.exception("Unhandled backend error")
    return jsonify({"error":{"type":"InternalError","message":"Unexpected backend error. Check the server log."}}),500

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","127.0.0.1"), port=int(os.getenv("PORT","3000")),
            debug=os.getenv("FLASK_DEBUG","0")=="1")
