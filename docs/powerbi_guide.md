# Power BI Desktop Production Dashboard Guide (Phase 9)

This guide provides a rapid (~15 minute) click-by-click implementation blueprint for connecting **Power BI Desktop** to Databricks Free Edition SQL Warehouse and constructing a 3-page operational triage and MLOps reporting suite.

All visuals and cards are driven strictly by dynamic DAX measures evaluated over live warehouse tables. **No typed constants or mock-up numbers are used.**

---

## 1. Stated Business Assumptions

Before building visuals or DAX measures, the following operational thresholds are formally defined as **stated assumptions**:

1. **Resolution SLA Target (14 Days)**:
   - **Assumption**: 14 calendar days from ticket creation to issue closure represents the standard Service Level Agreement (SLA) target for Home Assistant Core issues.
   - **Implementation**: Any closed issue where `duration_days > 14` is classified as an SLA breach.
2. **High-Risk Not-Planned Threshold (0.35)**:
   - **Assumption**: A predicted probability $\ge 0.35$ for `target_is_not_planned` (`p_not_planned >= 0.35`) classifies an issue as high-risk for closing without resolution (e.g. invalid report, stale unreproducible bug, or third-party device failure).
   - **Implementation**: Calibrated to optimize detection recall on the 12% ground-truth positive base rate.

---

## 2. Connecting Power BI Desktop to Databricks Free Edition

Power BI Desktop can connect directly to Databricks without requiring a Microsoft 365 work or school organizational account.

### Connection Steps:
1. Launch **Power BI Desktop** on Windows. If a sign-in dialog appears, click the **X** (close) icon in the top-right corner to proceed offline.
2. Click **Get Data** $\rightarrow$ **More...** $\rightarrow$ **Databricks** $\rightarrow$ Click **Connect**.
3. In the Databricks connection dialog, enter credentials from `.env`:
   - **Server Hostname**: Databricks workspace hostname (from `DATABRICKS_HOST` or `DATABRICKS_SERVER_HOSTNAME`).
   - **HTTP Path**: SQL Warehouse endpoint path (from `DATABRICKS_HTTP_PATH`).
4. Select Authentication: **Personal Access Token (PAT)**.
   - Enter `DATABRICKS_TOKEN` from `.env`.
5. In the **Navigator** window, expand:
   - Catalog: `workspace`
   - Schema: `gold`
6. Select and check the following 5 tables to load:
   - `workspace.gold.fct_tickets` (issue grain fact table)
   - `workspace.gold.fct_model_monitoring` (feature PSI drift metrics)
   - `workspace.gold.fct_model_performance` (out-of-sample evaluation history)
   - `workspace.gold.fct_retrain_decisions` (candidate vs. champion promotion records)
   - `workspace.gold.mart_weekly_volume` (historical weekly intake & forecaster feeds)
7. Click **Load** (or **Import** mode).

*(Offline Fallback: If the Databricks warehouse is paused or unreachable, use **Get Data $\rightarrow$ Folder** and point to `data/exports/powerbi/` to load the exact CSV counterparts).*

---

## 3. Production DAX Measures Reference

Create a dedicated measures container table (`_Measures`) or attach these measures directly to their parent fact tables:

### Measures on `fct_tickets`:
```dax
Total Tickets = COUNTROWS(fct_tickets)

Open Tickets = 
CALCULATE(
    COUNTROWS(fct_tickets),
    fct_tickets[state] = "open"
)

Closed Tickets = 
CALCULATE(
    COUNTROWS(fct_tickets),
    fct_tickets[state] = "closed"
)

Resolution Rate Pct = 
DIVIDE([Closed Tickets], [Total Tickets], 0) * 100

Median Days to Close = 
CALCULATE(
    MEDIAN(fct_tickets[duration_days]),
    fct_tickets[state] = "closed"
)

Completeness Rate Pct = 
100 - (DIVIDE(CALCULATE(COUNTROWS(fct_tickets), fct_tickets[is_incomplete] = 1), [Total Tickets], 0) * 100)

SLA Breach Rate Pct = 
DIVIDE(
    CALCULATE(
        COUNTROWS(fct_tickets),
        fct_tickets[duration_days] > 14,
        fct_tickets[state] = "closed"
    ),
    [Closed Tickets],
    0
) * 100

High Risk Not Planned Tickets = 
CALCULATE(
    COUNTROWS(fct_tickets),
    fct_tickets[p_not_planned] >= 0.35
)

High Risk Prevalence Pct = 
DIVIDE([High Risk Not Planned Tickets], [Total Tickets], 0) * 100

Empirical Brier Score = 
CALCULATE(
    AVERAGEX(
        fct_tickets,
        (fct_tickets[p_not_planned] - fct_tickets[target_is_not_planned]) ^ 2
    ),
    fct_tickets[state] = "closed"
)
```

### Measures on `fct_model_monitoring`:
```dax
Max Feature PSI = 
MAX(fct_model_monitoring[psi_score])

Active Drift Alerts = 
CALCULATE(
    COUNTROWS(fct_model_monitoring),
    fct_model_monitoring[alert_status] = "ALERT"
)

Pipeline Health Status = 
SWITCH(
    TRUE(),
    [Max Feature PSI] >= 0.25, "Severe Drift (Retrain Required)",
    [Max Feature PSI] >= 0.10, "Warning (Moderate Shift)",
    "Healthy (Normal Distribution)"
)
```

### Measures on `fct_model_performance`:
```dax
Out of Sample C Index = 
CALCULATE(
    MAX(fct_model_performance[survival_c_index]),
    fct_model_performance[sample_type] = "OUT_OF_SAMPLE"
)

Out of Sample PR AUC = 
CALCULATE(
    MAX(fct_model_performance[not_planned_pr_auc]),
    fct_model_performance[sample_type] = "OUT_OF_SAMPLE"
)
```

### Measures on `fct_retrain_decisions`:
```dax
Retrain Promotion Status = 
SELECTEDVALUE(fct_retrain_decisions[final_decision], "NO_DECISION")

Candidate Champion Delta = 
MAX(fct_retrain_decisions[delta])
```

---

## 4. 3-Page Dashboard Layout Specification

### Page 1: Triage Volume & Operations
*Focus: Overall repository velocity, issue backlog, and intake health.*

- **Top Filter Slicers**:
  - `fct_tickets[state]` (Tile or Dropdown: `open`, `closed`)
  - `fct_tickets[integration_key]` (Dropdown with Search)
  - `fct_tickets[created_date_key]` (Between Date Slider)
- **Top Metric Cards** (4 single-value Card visuals):
  1. Card: `[Total Tickets]`
  2. Card: `[Open Tickets]`
  3. Card: `[Median Days to Close]` (Format: `0.0 days`)
  4. Card: `[Completeness Rate Pct]` (Format: `0.0%`)
- **Main Visuals**:
  1. **Clustered Column Chart**: Intake Volume vs Resolution Throughput
     - X-Axis: `fct_tickets[created_date_key]` (or `dim_date[MonthName]`)
     - Y-Axis: `[Total Tickets]` and `[Closed Tickets]`
  2. **Horizontal Bar Chart**: Top 15 Integrations by Ticket Volume
     - Y-Axis: `fct_tickets[integration_key]`
     - X-Axis: `[Total Tickets]`
     - Tooltip: `[Open Tickets]`, `[Completeness Rate Pct]`

---

### Page 2: Resolution Survival & SLA Performance
*Focus: Survival duration forecasting, SLA compliance, and closing risk classification.*

- **Top Filter Slicers**:
  - `fct_tickets[integration_key]` (Dropdown)
  - `fct_tickets[has_traceback]` (Binary toggle: `0`, `1`)
- **Top Metric Cards** (4 single-value Card visuals):
  1. Card: `[SLA Breach Rate Pct]` (Format: `0.0%` — Stated assumption: >14 days)
  2. Card: `[High Risk Prevalence Pct]` (Format: `0.0%` — Stated assumption: p >= 0.35)
  3. Card: `[Empirical Brier Score]` (Format: `0.0000`)
  4. Card: `[Out of Sample C Index]` (Format: `0.0000`)
- **Main Visuals**:
  1. **Scatter Chart / Binned Line Chart**: Actual vs Predicted Resolution Duration
     - X-Axis: `fct_tickets[predicted_p50_days]`
     - Y-Axis: `fct_tickets[duration_days]`
     - Legend / Filter: `fct_tickets[state] = "closed"`
  2. **Table Visual**: High-Risk Triage Queue
     - Columns: `ticket_number`, `integration_key`, `p_not_planned`, `predicted_p50_days`, `state`
     - Visual-level filter: `p_not_planned >= 0.35`
     - Sort: Descending by `p_not_planned`

---

### Page 3: MLOps Drift & Retraining Governance
*Focus: Distribution stability, automated retraining evaluation, and continuous pipeline health.*

- **Top Filter Slicers**:
  - `fct_model_monitoring[alert_status]` (Tile: `NORMAL`, `ALERT`)
  - `fct_retrain_decisions[path_name]` (Dropdown: `path_a_stationary`, `path_b_expanded_retrain`)
- **Top Metric Cards** (4 single-value Card visuals):
  1. Card: `[Max Feature PSI]` (Format: `0.0000`)
  2. Card: `[Active Drift Alerts]` (Integer count)
  3. Card: `[Pipeline Health Status]` (Text badge: Healthy / Warning / Severe Drift)
  4. Card: `[Candidate Champion Delta]` (Format: `+0.0000`)
- **Main Visuals**:
  1. **Horizontal Bar Chart**: Feature Drift Ranking (PSI)
     - Y-Axis: `fct_model_monitoring[feature_name]`
     - X-Axis: `fct_model_monitoring[psi_score]`
     - Color formatting: Green if $< 0.10$, Yellow if $\ge 0.10$, Red if $\ge 0.25$.
  2. **Table Visual**: Retraining Promotion Decisions
     - Source: `fct_retrain_decisions`
     - Columns: `path_name`, `training_dataset_size`, `champion_metric`, `candidate_metric`, `delta`, `margin_required`, `final_decision`
  3. **Line Chart**: Model Evaluation History
     - Source: `fct_model_performance`
     - X-Axis: `window_start`
     - Y-Axis: `survival_c_index`, `not_planned_pr_auc`
