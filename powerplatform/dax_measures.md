# Power BI Production DAX Measures Library (Phase 9)

This document contains production-tested Data Analysis Expressions (DAX) measures categorized for executive reporting, operational triage monitoring, and machine learning health tracking in **Power BI**.

All measures are dynamically computed on real Databricks gold tables (`fct_tickets`, `fct_model_monitoring`, `fct_model_performance`, `fct_retrain_decisions`). No typed constants or mock-up numbers are used.

### Stated Business Assumptions
1. **SLA Target**: **14 Days**. Resolution elapsed time $> 14$ days is treated as an SLA breach.
2. **High-Risk Threshold**: **0.35**. Predicted not-planned probability $\ge 0.35$ flags an issue as high-risk.

---

## 1. Core Volume & Throughput Measures

### Total Tickets
```dax
Total Tickets = COUNTROWS(fct_tickets)
```

### Open Tickets
```dax
Open Tickets = 
CALCULATE(
    COUNTROWS(fct_tickets),
    fct_tickets[state] = "open"
)
```

### Closed Tickets
```dax
Closed Tickets = 
CALCULATE(
    COUNTROWS(fct_tickets),
    fct_tickets[state] = "closed"
)
```

### Resolution Rate %
```dax
Resolution Rate Pct = 
DIVIDE([Closed Tickets], [Total Tickets], 0) * 100
```

---

## 2. Issue Quality & Completeness Measures

### Incomplete Tickets
```dax
Incomplete Tickets = 
CALCULATE(
    COUNTROWS(fct_tickets),
    fct_tickets[is_incomplete] = 1
)
```

### Completeness Rate %
```dax
Completeness Rate Pct = 
100 - (DIVIDE([Incomplete Tickets], [Total Tickets], 0) * 100)
```

### Traceback Attachment %
```dax
Traceback Attachment Pct = 
DIVIDE(
    CALCULATE(COUNTROWS(fct_tickets), fct_tickets[has_traceback] = 1),
    [Total Tickets],
    0
) * 100
```

### Log Snippet Attachment %
```dax
Log Attachment Pct = 
DIVIDE(
    CALCULATE(COUNTROWS(fct_tickets), fct_tickets[has_logs] = 1),
    [Total Tickets],
    0
) * 100
```

### Terse One-Liner Issue Rate %
```dax
Terse Issue Rate Pct = 
DIVIDE(
    CALCULATE(COUNTROWS(fct_tickets), fct_tickets[body_len] < 50),
    [Total Tickets],
    0
) * 100
```

---

## 3. SLA & Survival Time-to-Close Measures

### Average Resolution Days (Actual)
```dax
Avg Days to Close = 
CALCULATE(
    AVERAGE(fct_tickets[duration_days]),
    fct_tickets[state] = "closed"
)
```

### Median Predicted P50 Days
```dax
Median Predicted P50 Days = 
MEDIAN(fct_tickets[predicted_p50_days])
```

### 90th Percentile Resolution Forecast (P90)
```dax
Predicted P90 Days = 
PERCENTILE.INC(fct_tickets[predicted_p90_days], 0.90)
```

### SLA Breach Rate % (Issues Taking > 14 Days)
```dax
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
```

---

## 4. Machine Learning & Risk Metrics

### High Risk Not-Planned Tickets
```dax
High Risk Not Planned Tickets = 
CALCULATE(
    COUNTROWS(fct_tickets),
    fct_tickets[p_not_planned] >= 0.35
)
```

### High Risk Prevalence %
```dax
High Risk Prevalence Pct = 
DIVIDE([High Risk Not Planned Tickets], [Total Tickets], 0) * 100
```

### Actual Not-Planned Tickets
```dax
Actual Not Planned Tickets = 
CALCULATE(
    COUNTROWS(fct_tickets),
    fct_tickets[target_is_not_planned] = 1
)
```

### Empirical Brier Score
```dax
Empirical Brier Score = 
CALCULATE(
    AVERAGEX(
        fct_tickets,
        (fct_tickets[p_not_planned] - fct_tickets[target_is_not_planned]) ^ 2
    ),
    fct_tickets[state] = "closed"
)
```

---

## 5. Drift Monitoring & MLOps Health KPIs

### Maximum Feature PSI
```dax
Max Feature PSI = 
MAX(fct_model_monitoring[psi_score])
```

### Active Drift Alert Count
```dax
Active Drift Alerts = 
CALCULATE(
    COUNTROWS(fct_model_monitoring),
    fct_model_monitoring[alert_status] = "ALERT"
)
```

### Pipeline Health Status Indicator
```dax
Pipeline Health Status = 
SWITCH(
    TRUE(),
    [Max Feature PSI] >= 0.25, "🔴 SEVERE DRIFT (Retrain Required)",
    [Max Feature PSI] >= 0.10, "🟡 WARNING (Moderate Shift)",
    "🟢 HEALTHY (Normal Distribution)"
)
```

### Out-of-Sample C-Index
```dax
Out of Sample C Index = 
CALCULATE(
    MAX(fct_model_performance[survival_c_index]),
    fct_model_performance[sample_type] = "OUT_OF_SAMPLE"
)
```

### Out-of-Sample PR-AUC
```dax
Out of Sample PR AUC = 
CALCULATE(
    MAX(fct_model_performance[not_planned_pr_auc]),
    fct_model_performance[sample_type] = "OUT_OF_SAMPLE"
)
```

### Retrain Promotion Decision Status
```dax
Retrain Promotion Status = 
SELECTEDVALUE(fct_retrain_decisions[final_decision], "NO_DECISION")
```

### Candidate vs Champion Metric Delta
```dax
Candidate Champion Delta = 
MAX(fct_retrain_decisions[delta])
```

