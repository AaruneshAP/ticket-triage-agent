# Specification: Ticket Triage Agent

## 1. Overview & Objective
The Ticket Triage Agent is an end-to-end reference implementation and portfolio project built for high-volume open-source software engineering repositories (specifically `home-assistant/core`). It ingests inbound issues, normalizes structured problem forms, predicts triage metrics (time-to-resolution survival curve, risk of closing as "not planned", integration label fallback), flags missing diagnostic information, retrieves historical duplicate tickets, and generates calibrated AI draft responses for human maintainer review.

## 2. Ingestion & Data Pipeline
- **Source**: GitHub REST API (`/repos/home-assistant/core/issues`), filtering out Pull Requests (where `pull_request` key is non-null).
- **Backfill Window**: Trailing 6 months (`MONTHS=6`), descending pagination with rate-limit backoff and checkpoint resumption.
- **Incremental Window**: Ingest issues updated since last checkpoint, upserting by issue number.
- **Storage Hierarchy**:
  - `data/raw/`: Immutable JSON dumps per page/batch.
  - `data/interim/`: Normalized Apache Parquet dataset.
  - `Snowflake RAW`: `RAW_ISSUES`, `RAW_COMMENTS`.
  - `dbt Bronze / Silver / Gold`: Cleaned staging views and curated analytical feature store (`fct_tickets`, `mart_weekly_volume`, `mart_model_monitoring`).

## 3. Modeling & Intelligence Layer
- **Survival Analysis (Time to Close)**:
  - Censoring: Open tickets right-censored at snapshot time.
  - Baselines: Historical median by integration.
  - Models: Lifelines Cox / Weibull AFT, XGBoost survival regression (`survival:aft`).
  - Metrics: Concordance Index, Brier score / calibration at 7d, 30d, 90d.
- **Resolution Quality Risk (Not Planned / No-Fix)**:
  - Target: Binary flag `closed_without_fix` (`state_reason == 'not_planned'` or duplicate closure without patch).
  - Model: Gradient boosted trees with text features (TF-IDF + embeddings) and metadata.
  - Explanations: TreeSHAP local and global feature attribution.
- **Label Prediction & Classification**:
  - Target: Top-N integration domains + "other".
  - Leakage Guard: Removes "Integration causing the issue" field from training text.
- **Retrieval & Duplicate Detection**:
  - Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`) on title and problem description.
  - Temporal Constraint: Query issue may only retrieve candidate matches created strictly prior to its creation.
  - Metric: Recall@5 on verified duplicate pairs.
- **Agentic Orchestration (LangGraph + Langfuse)**:
  - State graph executing: Form Parsing -> Completeness Rule Audit -> Model Inference -> Vector Retrieval -> Calibrated Draft Generation.
  - LLM Provider: Groq / Google GenAI with template fallback when credentials are absent.
  - Observability: Complete execution tracing via Langfuse.

## 4. Serving & User Interfaces
- **FastAPI Service**:
  - `GET /health`: Service health, model status, database connectivity.
  - `POST /triage`: Full inference payload (predicted p50/p90 days, risk score, SHAP drivers, duplicate matches, completeness checklist, draft reply).
- **Streamlit Maintainer App**:
  - Interactive triage review queue with Approve / Edit / Reject workflow.
  - Feedback written to database for model fine-tuning.
  - Volume forecasting and data drift dashboards.
- **Enterprise Integration**:
  - Power Automate alert triggers on high-risk or incomplete tickets.
  - Power Apps canvas application for mobile triage approvals.
  - Power BI executive reports (triage planner, backlog trends).
  - Looker Studio operational model health and feature drift monitoring.

## 5. Monitoring & Self-Healing
- Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests across rolling 14-day windows against baseline feature distributions.
- Evidently HTML and JSON report generation.
- Automated retraining evaluation trigger in Airflow when drift persists or performance drops.
