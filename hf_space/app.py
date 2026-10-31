"""Streamlit Public Interactive Demo for Ticket Triage Agent (Hugging Face Space).

Operates strictly in read-only demo mode over curated repository snapshot data.
No live database or warehouse credentials required.
"""

import json
import os

import pandas as pd
import streamlit as st
from triage_logic import parse_issue, predict_triage_outcomes

# Page configuration
st.set_page_config(
    page_title="Ticket Triage Agent | Home Assistant Core",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, polished appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-alert {
        color: #DC2626;
        font-weight: 600;
    }
    .badge-ok {
        color: #16A34A;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎫 Home Assistant Core | Ticket Triage Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Issue Completeness Verification, Component Routing, Survival Forecasting & Drift Governance</div>', unsafe_allow_html=True)

# Pre-defined sample tickets for fast demo testing
SAMPLE_PRESETS = {
    "1. MQTT Crash with Full Traceback (Complete)": {
        "title": "MQTT integration crashes on broker disconnect during startup",
        "body": """### The problem
When Home Assistant Core starts and the Mosquitto broker is restarting, the MQTT integration crashes with an unhandled exception instead of retrying the connection.

### What version of Home Assistant Core has the issue?
core-2026.8.0

### What type of installation are you running?
Home Assistant OS

### Integration causing the issue
mqtt

### Traceback/logs
```python
Traceback (most recent call last):
  File "/usr/src/homeassistant/homeassistant/components/mqtt/client.py", line 412, in async_connect
    await self._client.connect(host, port, keepalive)
ConnectionRefusedError: [Errno 111] Connection refused
```"""
    },
    "2. Vague One-Liner Issue (Incomplete - Rule C Flagged)": {
        "title": "Automations stopped triggering",
        "body": """### The problem
None of my automations work after the update. Please fix it.

### What version of Home Assistant Core has the issue?
all versions

### Integration causing the issue
automation"""
    },
    "3. ZHA Zigbee Device Drop (No Logs - Diagnostic Request)": {
        "title": "ZHA Aqara door sensor goes unavailable after 2 hours",
        "body": """### The problem
My Aqara contact sensor (lumi.sensor_magnet.aq2) pairs successfully via Sonoff Zigbee 3.0 dongle, but stops sending state updates and becomes unavailable after exactly 2 hours.

### What version of Home Assistant Core has the issue?
2026.8.1

### What type of installation are you running?
Home Assistant Supervised

### Integration causing the issue
zha"""
    },
}

# Navigation tabs
tab_playground, tab_performance, tab_drift, tab_retrain, tab_architecture = st.tabs([
    "🎯 Interactive Playground",
    "📊 Model Performance & Calibration",
    "🔍 Feature Drift Monitor (PSI)",
    "🔄 Retraining Governance",
    "🏗️ Architecture & Stated Assumptions",
])

# ---------------------------------------------------------
# TAB 1: INTERACTIVE TRIAGE PLAYGROUND
# ---------------------------------------------------------
with tab_playground:
    st.subheader("Live Ticket Intake & Triage Simulation")
    st.write("Select a pre-loaded GitHub issue scenario or paste a custom submission to evaluate automated triage:")

    col_select, col_empty = st.columns([2, 1])
    with col_select:
        selected_preset = st.selectbox("Choose a sample ticket:", list(SAMPLE_PRESETS.keys()) + ["Custom Ticket"])

    if selected_preset == "Custom Ticket":
        default_title = "HomeKit climate entity heating temperature step inverted"
        default_body = "### The problem\nChanging target temperature sends inverted delta to climate device.\n\n### What version of Home Assistant Core has the issue?\ncore-2026.8.0\n\n### Integration causing the issue\nhomekit"
    else:
        default_title = SAMPLE_PRESETS[selected_preset]["title"]
        default_body = SAMPLE_PRESETS[selected_preset]["body"]

    input_title = st.text_input("Issue Title:", value=default_title)
    input_body = st.text_area("Issue Markdown Body:", value=default_body, height=220)

    if st.button("🚀 Run Triage Pipeline", type="primary", use_container_width=True):
        parsed = parse_issue(input_body, input_title)
        outcomes = predict_triage_outcomes(parsed)

        st.markdown("---")
        st.subheader("Triage Assessment Results")

        # 4 Key Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Identified Component", f"`{outcomes['integration']}`")
        with c2:
            status_text = "❌ Incomplete" if outcomes["is_incomplete"] else "✅ Complete"
            st.metric("Rule C Completeness", status_text)
        with c3:
            st.metric("Resolution Forecast", "No reliable signal", help="Negative Result: Out-of-sample survival C-index is 0.514 (CI [0.492, 0.536]), fixed-horizon ROC-AUC 0.46–0.53 (crosses chance 0.50). Resolution time cannot be reliably predicted from ticket text; the cause was not investigated.")
        with c4:
            risk_label = "⚠️ High Risk" if outcomes["is_high_risk"] else "Normal Risk"
            st.metric("Not-Planned Risk (p)", f"{outcomes['p_not_planned']:.3f}", delta=risk_label, delta_color="inverse" if outcomes["is_high_risk"] else "normal")

        # Details & Draft Comment
        col_draft, col_diag = st.columns([3, 2])

        with col_draft:
            st.markdown("##### Automated GitHub Response Draft")
            st.code(outcomes["draft_comment"], language="markdown")
            st.caption("Prefixed with `[AI draft, needs review]` per open-source maintainer guidelines.")

        with col_diag:
            st.markdown("##### Parser & Diagnostic Diagnostics")
            diag_data = {
                "Signal": [
                    "Home Assistant Version Detected",
                    "Python Traceback Detected",
                    "System Log Snippets Detected",
                    "Template Structure Parsed",
                    "Resolution Latency Signal",
                ],
                "Value": [
                    str(parsed.get("version") or "None detected"),
                    "Yes" if parsed["has_traceback"] else "No",
                    "Yes" if parsed["has_logs"] else "No",
                    "Yes" if parsed["is_template"] else "No (Raw Text)",
                    "No reliable signal (suppressed)",
                ]
            }
            st.table(pd.DataFrame(diag_data))

            if outcomes["missing_items"]:
                st.warning("⚠️ **Missing Diagnostic Items Flagged:**\n- " + "\n- ".join(outcomes["missing_items"]))

# ---------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & CALIBRATION
# ---------------------------------------------------------
with tab_performance:
    st.subheader("Model Validation & Calibration Across Cohorts")
    st.write("Rigorous out-of-sample performance metrics verified across post-cutoff evaluation cohorts:")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Survival C-Index (Out-of-Sample)",
            "0.514",
            delta="95% CI [0.492, 0.536]",
            delta_color="inverse",
            help="Strictly out-of-sample post-training test C-index: 0.514 [0.492, 0.536] (crosses chance 0.50, negative result). Validation (tuning) split C-index: 0.5426.",
        )
    with m2:
        st.metric(
            "Not-Planned PR-AUC (Out-of-Sample)",
            "0.195",
            delta="vs 0.145 test prev",
            help="Strictly out-of-sample test: ROC-AUC 0.610 [0.546, 0.674], PR-AUC 0.1950 [0.147, 0.276] vs test prevalence 0.145 on N=496 closed tickets. Validation (tuning) split: PR-AUC 0.222, ROC-AUC 0.549.",
        )
    with m3:
        st.metric(
            "Empirical Brier Score (Out-of-Sample)",
            "0.1681",
            delta="14.5% test prevalence",
            help="Strictly out-of-sample test: 0.1681 on N=496 closed tickets. Validation (tuning) split: 0.1997 on 15.3% prevalence.",
        )
    with m4:
        st.metric(
            "Integration Top-1 (Out-of-Sample)",
            "80.0%",
            delta="vs 76.72% always-other",
            help="Strictly out-of-sample test (N=915 labeled, 26 classes): Top-1 80.0%, Top-3 97.16%, Macro-F1 0.644 vs always-'other' baseline 76.72% (Macro-F1 0.033). Validation (tuning) split: Top-1 86.6%, Macro-F1 0.832.",
        )

    # Load performance history
    perf_path = "data/performance_history.json"
    if os.path.exists(perf_path):
        with open(perf_path, "r", encoding="utf-8") as f:
            perf_json = json.load(f)
        st.markdown("##### Cohort Evaluation History")
        st.dataframe(pd.DataFrame(perf_json.get("cohort_metrics", [])), use_container_width=True)

    st.markdown("---")
    st.markdown("##### Leave-One-Cohort-Out (LOCO) Completeness Rule Results")
    loco_data = {
        "Test Cohort": ["Cohort 3 (Unseen)", "Cohort 2 (Held-Out)", "Cohort 1 (In-Sample)", "Unweighted Mean"],
        "Selected Rule": ["Rule C (<25w)", "Rule B/C (<20w/<25w)", "Rule C (<25w)", "Rule C Consensus"],
        "Precision": ["100.0%", "33.3% - 50.0%", "50.0%", "61.1% - 66.7%"],
        "Recall": ["100.0%", "28.6% - 57.1%", "62.5%", "63.7% - 73.2%"],
        "F1-Score": ["1.000", "0.308 - 0.533", "0.556", "0.621 - 0.696"],
    }
    st.table(pd.DataFrame(loco_data))
    st.info("ℹ️ **Methodology Disclosure**: All 150 available labeled tickets were used during LOCO cross-validation. No unseen data remains in the project. Any future tuning will require a 4th unseen cohort.")

# ---------------------------------------------------------
# TAB 3: FEATURE DRIFT MONITOR (PSI)
# ---------------------------------------------------------
with tab_drift:
    st.subheader("Feature Stability & Distribution Drift Tracking")
    st.write("Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test tracking between training baseline and production intake:")

    drift_path = "data/drift_metrics.json"
    if os.path.exists(drift_path):
        with open(drift_path, "r", encoding="utf-8") as f:
            drift_json = json.load(f)

        drift_records = []
        for feat, info in drift_json.get("features_analyzed", {}).items():
            drift_records.append({
                "Feature / Signal": feat,
                "Reference Mean": round(info.get("ref_mean", 0.0), 3),
                "Production Mean": round(info.get("curr_mean", 0.0), 3),
                "PSI Score": round(info.get("psi", 0.0), 4),
                "KS Statistic": round(info.get("ks_statistic", 0.0), 4),
                "Status": info.get("status", "NORMAL").upper(),
            })
        drift_df = pd.DataFrame(drift_records)

        col_tbl, col_chart = st.columns([3, 2])
        with col_tbl:
            st.dataframe(drift_df, use_container_width=True)
        with col_chart:
            st.bar_chart(drift_df.set_index("Feature / Signal")["PSI Score"])

        st.caption("Drift Thresholds: 🟢 Normal (PSI < 0.10) | 🟡 Warning (0.10 ≤ PSI < 0.25) | 🔴 Alert / Retrain Required (PSI ≥ 0.25)")

# ---------------------------------------------------------
# TAB 4: RETRAINING GOVERNANCE
# ---------------------------------------------------------
with tab_retrain:
    st.subheader("Automated Retraining Demonstration & Margin Gating")
    st.write("Demonstration of production model promotion policy with strict margin requirements:")

    retrain_path = "data/promotion_decisions.json"
    if os.path.exists(retrain_path):
        with open(retrain_path, "r", encoding="utf-8") as f:
            retrain_json = json.load(f)

        paths = retrain_json.get("demonstration_paths", {})

        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("#### Path A: Stationary Data Window")
            p_a = paths.get("path_a_stationary", {})
            st.write(p_a.get("description", ""))
            st.metric("Champion Metric", f"{p_a.get('champion_metric', 0.0):.4f}")
            st.metric("Candidate Metric", f"{p_a.get('candidate_metric', 0.0):.4f}")
            st.metric("Decision", p_a.get("final_decision", "REJECT"), delta="No Statistical Gain", delta_color="inverse")

        with c_b:
            st.markdown("#### Path B: Expanded Window Retraining")
            p_b = paths.get("path_b_expanded_retrain", {})
            st.write(p_b.get("description", ""))
            st.metric("Champion Metric", f"{p_b.get('champion_metric', 0.0):.4f}")
            st.metric("Candidate Metric", f"{p_b.get('candidate_metric', 0.0):.4f}")
            st.metric("Decision", p_b.get("final_decision", "PROMOTED"), delta=f"+{p_b.get('delta', 0.0):.4f} (Exceeds Margin)", delta_color="normal")

# ---------------------------------------------------------
# TAB 5: ARCHITECTURE & ASSUMPTIONS
# ---------------------------------------------------------
with tab_architecture:
    st.subheader("End-to-End System Architecture & Operating Assumptions")

    st.markdown(r"""
    ### Data Pipeline & Control Flow
    ```
    GitHub API (Core Issues)
        │
        ▼
    Hourly Ingestion & Raw Parquet
        │
        ▼
    Databricks SQL Warehouse (Free Edition)
        │
        ▼
    dbt (Bronze -> Silver -> Gold Marts)
        │
        ▼
    Multi-Model Inference (Classifier + XGBoost + AFT Survival + Vector Embeddings)
        │
        ▼
    LangGraph Triage Agent + FastAPI Gateway (Langfuse Traces)
        ├── Review App (Human-in-the-loop Streamlit)
        ├── n8n Webhook Alerts (High-Risk Triage)
        ├── Power BI Desktop (Support Velocity & Capacity Planning)
        └── Looker Studio (Real-Time Model Health via Google Sheets Bridge)
    ```

    ### Formally Stated Operating Assumptions:
    1. **SLA Assumption (14 Calendar Days)**: Target resolution threshold for Home Assistant issues. Duration $> 14$ days is tracked as an SLA breach.
    2. **High-Risk Assumption (0.35 Probability)**: Predicted probability $p \ge 0.35$ of closing as `not_planned` flags tickets for proactive triage intervention.
    3. **Sandbox Allow-List Guard**: Writes (issue comments and labels) are strictly restricted in code to designated sandbox repositories (`ticket-triage-sandbox`). The main `home-assistant/core` repository is strictly read-only.
    """)
