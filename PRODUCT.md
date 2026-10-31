# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Quantitative researchers and analysts who test trading hypotheses on Indian equity market data (NIFTY universes). They work at a desk, on a desktop browser, with dense tables and charts open for long sessions.

## Product Purpose
Unified Backtester is an alpha research workspace. A user states a financial hypothesis in plain language, gets candidate alpha expressions, validates them, runs them through the existing notebook backtesting engine, reviews the results, and saves reproducible versions. Success means a researcher can move from idea to a trustworthy, reproducible backtest without losing track of what was validated and what was only proposed.

## Positioning
The backtesting engine is the source of truth for every number. The interface separates a candidate expression, a validated expression, and an alpha that actually passed a backtest, and it never shows mock metrics as real results. A neighboring tool could copy the features, but not this honesty about provenance.

## Operating Context
Used in long research sessions on a desktop. Typical workflow: describe an idea, review generated candidates, edit one expression, run it on a date range and universe, compare runs, save the good ones. Results include equity curves, drawdown, turnover, and IC.

## Capabilities and Constraints
- Idea-to-expression assistant with cache and retrieval; candidates need validation and backtesting before any claim.
- Expression research screen with configurable universe, dates, holding period, costs, and execution settings.
- Alpha library with immutable versions and run history.
- Backend and notebook engine are the authoritative computation. The frontend must not duplicate or fake them.
- Missing metrics are shown as unavailable, never as fabricated defaults.
- Out of scope: screeners, investment strategy, portfolio simulation.

## Brand Commitments
- Product name: Unified Backtester. The user confirmed the visual identity is free to change completely; the name and features stay.

## Evidence on Hand
No testimonials, customers, case studies, or benchmark claims exist. Do not invent any.

## Product Principles
- Never present a number that did not come from an actual backtest run.
- Make the status of every candidate, validation, and run visible at a glance.
- Keep dense research data legible: numbers aligned, states obvious, nothing decorative competing with the data.
- Prefer plain labels for research terms over jargon-as-decoration.
