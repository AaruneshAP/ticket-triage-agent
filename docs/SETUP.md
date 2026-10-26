# External Accounts and Environment Setup

This project interfaces with external services for ingestion, data warehousing, LLM inference, telemetry, alerting, and interactive demonstration. Below is the list of required service accounts, keys, and environment variables.

## Required Environment Variables & Keys

| Service / Tool | Purpose | Environment Variable / Setting |
|---|---|---|
| **GitHub** | Issue ingestion & triage sandbox | `GITHUB_TOKEN`, `GITHUB_SANDBOX_TOKEN`, `SANDBOX_REPO` |
| **Databricks SQL Warehouse** | Medallion data warehouse (Bronze/Silver/Gold) | `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_HTTP_PATH` |
| **Groq Cloud** | Primary LLM inference (Llama / Mixtral) | `GROQ_API_KEY` |
| **Google Gemini API** | Secondary LLM inference fallback | `GEMINI_API_KEY` |
| **Langfuse Cloud** | LLM tracing and observability telemetry | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` |
| **Google Cloud Service Account** | Google Sheets bridge for Looker Studio | `GOOGLE_SHEETS_CREDENTIALS_JSON`, `GOOGLE_SHEET_ID` |
| **Telegram Bot** | Webhook alerting channel for n8n | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| **Hugging Face** | Interactive demo Space hosting | `HF_TOKEN` |

## Local Prerequisites

- **Python 3.12+** (managed via `uv` or standard venv)
- **Docker Desktop** (WSL2 backend on Windows or native engine on Linux) for running Apache Airflow and n8n services
- **Power BI Desktop** (optional, for viewing the offline `.pbix` report)
