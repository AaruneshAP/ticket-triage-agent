# Ticket Triage Documentation & Completeness Cohorts

This directory contains technical documentation, empirical benchmarks, and the 3 validation cohorts used to evaluate diagnostic completeness:

---

## 1. Completeness Evaluation Cohorts

To ensure statistical rigor and measure out-of-cohort generalization, the completeness rules are evaluated across three distinct cohorts ($N=50$ tickets each, 150 tickets total with 17 positives):

| Cohort File | Ticket Range | Description & Ground Truth | Evaluation Role |
| :--- | :---: | :--- | :--- |
| **`completeness_insample.csv`** | #13373 – #170262 | Historical intake tickets. Contains 8 positives and 42 negatives. | In-Sample Benchmark (Rule tuning) |
| **`completeness_heldout.csv`** | #176576 – #178271 | Mid-intake tickets. Contains 7 positives and 43 negatives. | Held-Out Test Cohort |
| **`completeness_unseen_cohort3.csv`** | #178272 – #178364 | Recent intake tickets. Contains 2 positives and 48 negatives. | Third Cohort (LOCO cross-validation) |

---

## 2. Directory Index

- **`SETUP.md`**: Required external accounts, service keys, and environment variables.
- **`HOW_IT_WORKS.md`**: Architectural reference, data flow, pipeline stages, and design decisions.
- **`evaluation.md`**: Evaluation report across all models (Survival Analysis, Not-Planned Risk, Label Recommender, Volume Forecaster, Completeness Rules, and Retrieval).
- **`data_dictionary.md`**: Schema and feature definitions for the Home Assistant issue triage dataset.
- **`google_sheets_looker_studio.md`**: Setup guide for Google Sheets and Looker Studio dashboards.
- **`powerbi_guide.md`**: Guide for connecting Power BI Desktop to Databricks gold tables.
- **`draft_review.md`**: LLM draft response evaluation rubric and qualitative inspection.
- **`warehouse_spike.md`**: Verification log and latency benchmarks for Databricks Free Edition.
- **`data_reconciliation.md`**: End-to-end count reconciliation across raw, bronze, silver, gold, and feature stores.
- **`images/`**: Verified visual artifacts for Power BI dashboards, Looker Studio reports, and Langfuse observability traces (see [README.md#dashboards-and-observability](../README.md#dashboards-and-observability)).
