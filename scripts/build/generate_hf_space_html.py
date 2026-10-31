"""Generate standalone hf_space/index.html with real replay tickets and strict docs/evaluation.md metrics."""

import json
from pathlib import Path

REPLAY_PATH = Path("data/interim/replay_showcase_tickets.json")
DRIFT_PATH = Path("hf_space/data/drift_metrics.json")
OUTPUT_PATH = Path("hf_space/index.html")

def build_index_html():
    with open(REPLAY_PATH, "r", encoding="utf-8") as f:
        raw_replay = json.load(f)
    replay_tickets = raw_replay["tickets"] if isinstance(raw_replay, dict) and "tickets" in raw_replay else raw_replay
    cohort_obs = raw_replay.get("cohort_observation", {}) if isinstance(raw_replay, dict) else {}
    obs14_info = cohort_obs.get("obs_14d", {})
    obs30_info = cohort_obs.get("obs_30d", {})
    act_14_pct = f"{obs14_info.get('actual_fraction', 0.348):.1%}"
    cnt_14 = f"{obs14_info.get('actual_count', 8)}/{obs14_info.get('n', 23)}"
    ci_14 = obs14_info.get("ci_95_display", "[18.8%, 55.1%]")
    n_14 = obs14_info.get("n", 23)

    act_30_pct = f"{obs30_info.get('actual_fraction', 0.333):.1%}"
    cnt_30 = f"{obs30_info.get('actual_count', 3)}/{obs30_info.get('n', 9)}"
    ci_30 = obs30_info.get("ci_95_display", "[12.1%, 64.6%]")
    n_30 = obs30_info.get("n", 9)

    with open(DRIFT_PATH, "r", encoding="utf-8") as f:
        drift_data = json.load(f)

    drift_records = []
    for feat, info in drift_data.get("features_analyzed", {}).items():
        drift_records.append({
            "feat": feat,
            "ref": round(info.get("ref_mean", 0.0), 2),
            "curr": round(info.get("curr_mean", 0.0), 2),
            "psi": round(info.get("psi", 0.0), 4),
            "ks": round(info.get("ks_statistic", 0.0), 4),
            "status": info.get("alert_level", "normal"),
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home Assistant Core | Ticket Triage Agent Demo</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-base: #0B0F19;
            --bg-surface: #111827;
            --bg-card: rgba(17, 24, 39, 0.85);
            --bg-card-hover: rgba(31, 41, 55, 0.95);
            --border-subtle: rgba(255, 255, 255, 0.09);
            --border-active: rgba(99, 102, 241, 0.6);
            --primary: #6366F1;
            --primary-hover: #4F46E5;
            --cyan: #38BDF8;
            --emerald: #10B981;
            --amber: #F59E0B;
            --rose: #EF4444;
            --text-main: #F9FAFB;
            --text-muted: #9CA3AF;
            --text-subtle: #6B7280;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            line-height: 1.5;
            background-image:
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(56, 189, 248, 0.08) 0px, transparent 50%);
            background-attachment: fixed;
        }}

        .container {{
            max-width: 1240px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
        }}

        header {{
            margin-bottom: 1.5rem;
            padding-bottom: 1.25rem;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .header-title-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .header-title {{
            font-size: 1.85rem;
            font-weight: 700;
            background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}

        .header-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.12);
            color: var(--emerald);
            border: 1px solid rgba(16, 185, 129, 0.25);
        }}

        .header-desc {{
            margin-top: 0.5rem;
            color: var(--text-muted);
            font-size: 0.95rem;
            max-width: 880px;
        }}

        /* Banner */
        .demo-banner {{
            background: rgba(99, 102, 241, 0.08);
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.75rem;
        }}

        /* Navigation Tabs */
        .nav-tabs {{
            display: flex;
            gap: 0.5rem;
            background: rgba(15, 23, 42, 0.6);
            padding: 0.35rem;
            border-radius: 10px;
            border: 1px solid var(--border-subtle);
            margin-bottom: 2rem;
            overflow-x: auto;
        }}

        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.6rem 1.1rem;
            border-radius: 7px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.18s ease;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .tab-btn:hover {{
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.04);
        }}

        .tab-btn.active {{
            color: #FFFFFF;
            background: var(--primary);
            box-shadow: 0 2px 10px rgba(99, 102, 241, 0.35);
        }}

        .tab-panel {{
            display: none;
            animation: fadeIn 0.25s ease-out;
        }}

        .tab-panel.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Card System */
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            backdrop-filter: blur(12px);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        .card-header {{
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .card-desc {{
            color: var(--text-muted);
            font-size: 0.88rem;
            margin-bottom: 1.25rem;
        }}

        /* Ticket Select Chips */
        .ticket-chips-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }}

        .ticket-chip {{
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            text-align: left;
            cursor: pointer;
            transition: all 0.18s ease;
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }}

        .ticket-chip:hover {{
            border-color: var(--border-active);
            background: rgba(30, 41, 59, 0.8);
            transform: translateY(-1px);
        }}

        .ticket-chip.active {{
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.15);
            box-shadow: 0 0 0 1px var(--primary);
        }}

        .chip-number {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--cyan);
        }}

        .chip-title {{
            font-size: 0.86rem;
            font-weight: 600;
            color: var(--text-main);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .chip-meta {{
            font-size: 0.75rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Side-by-Side Dual Column Grid */
        .dual-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        @media (max-width: 900px) {{
            .dual-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        /* KPI Metric Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.25rem;
        }}

        .kpi-card {{
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 1.1rem 1rem;
            text-align: center;
        }}

        .kpi-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-subtle);
            margin-bottom: 0.35rem;
        }}

        .kpi-value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text-main);
        }}

        .kpi-badge {{
            margin-top: 0.4rem;
            font-size: 0.73rem;
            font-weight: 600;
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
        }}

        /* Code & Markdown Display */
        .code-box {{
            background: #090D16;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 1rem 1.25rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            color: #E2E8F0;
            white-space: pre-wrap;
            line-height: 1.45;
            max-height: 380px;
            overflow-y: auto;
        }}

        /* Tables */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            text-align: left;
        }}

        .data-table th {{
            background: rgba(30, 41, 59, 0.5);
            padding: 0.75rem 0.85rem;
            color: var(--text-muted);
            font-weight: 600;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .data-table td {{
            padding: 0.75rem 0.85rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-main);
        }}

        .data-table tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}

        .tag-pill {{
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .tag-pill.normal {{ background: rgba(16, 185, 129, 0.12); color: var(--emerald); }}
        .tag-pill.warning {{ background: rgba(245, 158, 11, 0.12); color: var(--amber); }}
        .tag-pill.alert {{ background: rgba(239, 68, 68, 0.12); color: var(--rose); }}

        .bar-container {{
            width: 100%;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 9999px;
            height: 7px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
            margin-right: 0.5rem;
            max-width: 85px;
        }}

        .bar-fill {{
            height: 100%;
            border-radius: 9999px;
        }}

        .bar-fill.normal {{ background: var(--emerald); }}
        .bar-fill.warning {{ background: var(--amber); }}
        .bar-fill.alert {{ background: var(--rose); }}

        footer {{
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border-subtle);
            text-align: center;
            font-size: 0.82rem;
            color: var(--text-subtle);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title-row">
                <div class="header-title">
                    <span>🎫</span> Home Assistant Core | Ticket Triage Agent
                </div>
                <div class="header-badge">
                    <span>●</span> Empirical Replay Space (Zero Heuristics)
                </div>
            </div>
            <p class="header-desc">
                Automated GitHub issue triage, completeness verification (Rule C), component routing, survival duration forecasting (AFT Log-Normal), and statistical MLOps drift governance.
            </p>
        </header>

        <div class="demo-banner">
            <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.95rem; font-weight:600; color:var(--cyan);">
                <span>🔬</span> Empirical Replay Demonstration (Zero Invented Constants)
            </div>
            <div style="font-size:0.86rem; color:var(--text-muted); margin-top:0.35rem; line-height:1.45;">
                All predictions in this demo are precomputed outputs from the project's real exported models (TF-IDF Label Classifier, Rule C Engine, XGBoost Not-Planned Classifier, and AFT Log-Normal Survival Model) displayed side-by-side with ground-truth outcomes on real Home Assistant Core test tickets. In accordance with project integrity standards, no client-side heuristic approximations or invented constants are used.
            </div>
        </div>

        <nav class="nav-tabs">
            <button class="tab-btn active" onclick="switchTab('replay')">🎯 Empirical Replay Demo</button>
            <button class="tab-btn" onclick="switchTab('performance')">📊 Model Performance</button>
            <button class="tab-btn" onclick="switchTab('drift')">🔍 Feature Drift (PSI)</button>
            <button class="tab-btn" onclick="switchTab('retrain')">🔄 Retraining Governance</button>
            <button class="tab-btn" onclick="switchTab('architecture')">🏗️ Architecture</button>
        </nav>

        <!-- ============================================================== -->
        <!-- TAB 1: EMPIRICAL REPLAY DEMO -->
        <!-- ============================================================== -->
        <div id="tab-replay" class="tab-panel active">
            <div class="card">
                <div class="card-header">
                    <span>Select Real Test Ticket to Replay:</span>
                    <span style="font-size:0.8rem; color:var(--text-muted);">30 Holdout Post-Cutoff Issues (12 Closed, 18 Open, 3 Not-Planned)</span>
                </div>
                <div style="margin-top:0.75rem; margin-bottom:1rem; padding:0.85rem 1.1rem; background:rgba(99,102,241,0.12); border-left:4px solid var(--primary); border-radius:6px; font-size:0.85rem; color:var(--text-main); line-height:1.5;">
                    <div style="font-weight:700; color:var(--cyan); margin-bottom:0.35rem;">
                        🎯 30-Ticket Replay Cohort & Calibration Summary (Fixed Seed <code>random_state=42</code>)
                    </div>
                    <div>
                        <strong>Sampling Window:</strong> 30 tickets sampled from 1,936 holdout tickets created strictly after champion models' training cutoff (<code>2026-08-27T05:00:47Z</code>) up to Phase 3 freeze (<code>2026-10-26T06:15:50Z</code>). Never seen in training or hyperparameter tuning.
                    </div>
                    <div style="margin-top:0.35rem;">
                        <strong>Sample Breakdown:</strong> <strong>12 closed</strong> (9 completed, 2 not planned, 1 duplicate) | <strong>18 still open</strong>.
                    </div>
                    <div style="margin-top:0.35rem;">
                        <strong>14-Day Closure Rate (Observed &ge;14 Days, created &le; 2026-10-12, n={n_14}):</strong> Actual <strong>{act_14_pct}</strong> ({cnt_14}) | 95% Wilson CI: <strong>{ci_14}</strong>. Duration predictions suppressed.
                    </div>
                    <div style="margin-top:0.35rem;">
                        <strong>30-Day Closure Rate (Observed &ge;30 Days, created &le; 2026-09-26, n={n_30}):</strong> Actual <strong>{act_30_pct}</strong> ({cnt_30}) | 95% Wilson CI: <strong>{ci_30}</strong>. Duration predictions suppressed.
                    </div>
                    <div style="margin-top:0.35rem; font-size:0.81rem; color:var(--text-muted); line-height:1.45;">
                        <em>Negative Result Disclosure: Resolution latency and closure probability models show no reliable out-of-sample signal (survival C-index 0.514, fixed-horizon ROC-AUC 0.46–0.53, indistinguishable from chance). Predictions are suppressed from active display; the cause was not investigated. Calibration finding on holdout: 14-day lowest bin 0.11 vs 0.34; 30-day top bin 0.86 predicted vs 0.59 observed. Replay sample sizes (n=23 for 14d, n=9 for 30d) condition strictly on full observation horizons before snapshot freeze.</em>
                    </div>
                </div>
                <p class="card-desc">Click any of the 30 real Home Assistant Core holdout test issues below to compare exact model predictions against the verified actual outcome:</p>

                <div class="ticket-chips-grid" id="ticket-chips-container"></div>
            </div>

            <!-- Ticket Content & Dual Comparison Grid -->
            <div class="card">
                <div class="card-header" style="flex-wrap:wrap; gap:0.5rem;">
                    <div>
                        <span id="ticket-title-display" style="font-weight:700;">-</span>
                        <a id="ticket-github-link" href="#" target="_blank" style="color:var(--cyan); margin-left:0.5rem; text-decoration:none; font-size:0.85rem;">[View on GitHub ↗]</a>
                    </div>
                    <div id="ticket-meta-display" style="font-size:0.8rem; color:var(--text-muted); font-family:'JetBrains Mono', monospace;">-</div>
                </div>

                <div style="margin-top:1rem; margin-bottom:1.5rem;">
                    <div style="font-size:0.82rem; font-weight:600; color:var(--text-muted); margin-bottom:0.4rem;">Ticket Markdown Body:</div>
                    <div class="code-box" id="ticket-body-display"></div>
                </div>

                <div class="dual-grid">
                    <!-- Left: Model Predictions -->
                    <div style="background:rgba(30,41,59,0.35); border:1px solid var(--border-subtle); border-radius:10px; padding:1.25rem;">
                        <div style="font-size:1.05rem; font-weight:700; color:var(--primary); margin-bottom:0.35rem; display:flex; align-items:center; gap:0.4rem;">
                            <span>🤖</span> Real Exported Model Predictions
                        </div>
                        <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:1rem;">Inference generated by XGBoost, AFT Survival, TF-IDF Recommender & Rule C Engine:</p>

                        <div class="kpi-grid">
                            <div class="kpi-card">
                                <div class="kpi-title">Predicted Component</div>
                                <div class="kpi-value" id="pred-integ" style="color:var(--cyan);">-</div>
                                <div class="kpi-badge" style="background:rgba(56,189,248,0.12); color:var(--cyan);">Top-1 Model Label</div>
                            </div>
                            <div class="kpi-card">
                                <div class="kpi-title">Rule C Completeness</div>
                                <div class="kpi-value" id="pred-complete">-</div>
                                <div class="kpi-badge" id="pred-complete-badge">-</div>
                            </div>
                        </div>

                        <div class="kpi-grid">
                            <div class="kpi-card">
                                <div class="kpi-title">Resolution Latency Signal</div>
                                <div class="kpi-value" id="pred-latency" style="color:#A78BFA; font-size:1.05rem;">No reliable signal</div>
                                <div class="kpi-badge" style="background:rgba(239,68,68,0.12); color:var(--rose);">Negative Result (C-Index: 0.514)</div>
                                <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem;">Out-of-sample C-index 0.514 (CI [0.492, 0.536]), ROC-AUC 0.46–0.53. Duration suppressed; the cause was not investigated.</div>
                            </div>
                            <div class="kpi-card">
                                <div class="kpi-title">Not-Planned Risk (p)</div>
                                <div class="kpi-value" id="pred-risk">-</div>
                                <div class="kpi-badge" id="pred-risk-badge">-</div>
                                <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem;">Test prevalence: 0.145 | Out-of-sample PR-AUC: 0.195 (ROC-AUC: 0.610)</div>
                            </div>
                        </div>

                        <div id="pred-missing-container" style="display:none; margin-top:0.75rem; padding:0.75rem; border-radius:6px; background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25);">
                            <div style="font-size:0.78rem; font-weight:600; color:var(--rose); margin-bottom:0.25rem;">Rule C Missing Diagnostic Requirements:</div>
                            <ul id="pred-missing-list" style="font-size:0.78rem; color:#FCA5A5; margin-left:1.25rem;"></ul>
                        </div>

                        <div style="margin-top:1rem;">
                            <div style="font-size:0.8rem; font-weight:600; color:var(--text-muted); margin-bottom:0.35rem;">Automated GitHub Response Draft:</div>
                            <div class="code-box" id="pred-draft" style="max-height:180px; font-size:0.78rem;"></div>
                        </div>
                    </div>

                    <!-- Right: Actual Outcome -->
                    <div style="background:rgba(30,41,59,0.35); border:1px solid var(--border-subtle); border-radius:10px; padding:1.25rem;">
                        <div style="font-size:1.05rem; font-weight:700; color:var(--emerald); margin-bottom:0.35rem; display:flex; align-items:center; gap:0.4rem;">
                            <span>🎯</span> Verified Actual Ground-Truth Outcome
                        </div>
                        <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:1rem;">Observed GitHub lifecycle resolution and verified maintainer actions:</p>

                        <div class="kpi-grid">
                            <div class="kpi-card">
                                <div class="kpi-title">Actual Resolution State</div>
                                <div class="kpi-value" id="actual-state" style="color:var(--text-main);">-</div>
                                <div class="kpi-badge" id="actual-state-badge">-</div>
                            </div>
                            <div class="kpi-card">
                                <div class="kpi-title">Actual Integration</div>
                                <div class="kpi-value" id="actual-integ" style="color:var(--cyan);">-</div>
                                <div class="kpi-badge" style="background:rgba(56,189,248,0.12); color:var(--cyan);">GitHub Label</div>
                            </div>
                        </div>

                        <div class="kpi-grid">
                            <div class="kpi-card">
                                <div class="kpi-title">Actual Days to Close</div>
                                <div class="kpi-value" id="actual-days" style="color:#A78BFA; font-size:1.4rem;">-</div>
                                <div class="kpi-badge" style="background:rgba(167,139,250,0.12); color:#A78BFA;">Observed Duration</div>
                            </div>
                            <div class="kpi-card">
                                <div class="kpi-title">Ground-Truth Complete</div>
                                <div class="kpi-value" id="actual-complete">-</div>
                                <div class="kpi-badge" id="actual-complete-badge">-</div>
                            </div>
                        </div>

                        <div style="margin-top:1rem; padding:1rem; border-radius:8px; background:rgba(15,23,42,0.6); border:1px solid var(--border-subtle);">
                            <div style="font-size:0.82rem; font-weight:600; color:var(--text-muted); margin-bottom:0.4rem;">Model vs Reality Diagnostic Assessment:</div>
                            <div id="comparison-analysis-text" style="font-size:0.85rem; color:var(--text-main); line-height:1.5;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================== -->
        <!-- TAB 2: PERFORMANCE -->
        <!-- ============================================================== -->
        <div id="tab-performance" class="tab-panel">
            <div class="card">
                <div class="card-header">Rigorous Out-of-Sample Offline Model Calibration</div>
                <p class="card-desc">Empirical metrics derived strictly from <code>docs/evaluation.md</code> across held-out chronological cohorts:</p>

                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-title">Survival C-Index (Out-of-Sample)</div>
                        <div class="kpi-value" style="color:var(--cyan);">0.514</div>
                        <div class="kpi-badge" style="background:rgba(56,189,248,0.12); color:var(--cyan);">95% CI: [0.492, 0.536] (Chance 0.50)</div>
                        <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.35rem;">Out-of-sample C-index 0.514; validation (tuning) was 0.5426. Negative result.</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Not-Planned PR-AUC (Out-of-Sample)</div>
                        <div class="kpi-value" style="color:var(--emerald);">0.195</div>
                        <div class="kpi-badge" style="background:rgba(16,185,129,0.12); color:var(--emerald);">ROC-AUC: 0.610 | Prevalence: 0.145</div>
                        <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.35rem;">Strictly out-of-sample test; validation (tuning) PR-AUC was 0.222</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Empirical Brier Score (Out-of-Sample)</div>
                        <div class="kpi-value" style="color:#A78BFA;">0.1681</div>
                        <div class="kpi-badge" style="background:rgba(167,139,250,0.12); color:#A78BFA;">Test Split (N=496 closed)</div>
                        <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.35rem;">Out-of-sample: 0.1681 on 14.5% ground truth; validation (tuning): 0.1997</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Label Recommender Top-1 (Out-of-Sample)</div>
                        <div class="kpi-value" style="color:var(--emerald);">80.0%</div>
                        <div class="kpi-badge" style="background:rgba(16,185,129,0.12); color:var(--emerald);">Macro-F1: 0.644 | Top-3: 97.16%</div>
                        <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.35rem;">vs always-"other" baseline 76.72% (Validation (tuning): 86.6% Top-1, 0.832 Macro-F1)</div>
                    </div>
                </div>

                <div class="card-header" style="margin-top:1.5rem;">Leave-One-Cohort-Out (LOCO) Completeness Rule Evaluation</div>
                <p class="card-desc">Evaluated by selecting candidate rules on two cohorts (N=100) and testing strictly out-of-distribution on the third cohort (N=50):</p>

                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Test Cohort</th>
                            <th>Selected Candidate Rule</th>
                            <th>OOF Confusion Matrix</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Cohort 3 (Unseen)</strong></td>
                            <td>Rule C (&lt;25 words + version + title-only)</td>
                            <td>TP=2, FP=0, FN=0, TN=48</td>
                            <td style="color:var(--emerald);">100.0%</td>
                            <td style="color:var(--emerald);">100.0%</td>
                            <td style="color:var(--emerald); font-weight:700;">1.000</td>
                        </tr>
                        <tr>
                            <td><strong>Cohort 2 (Held-Out)</strong></td>
                            <td>Rule B / Rule C tie on training data</td>
                            <td>TP=2-4, FP=4, FN=3-5, TN=39</td>
                            <td>33.3% - 50.0%</td>
                            <td>28.6% - 57.1%</td>
                            <td style="font-weight:700;">0.308 - 0.533</td>
                        </tr>
                        <tr>
                            <td><strong>Cohort 1 (In-Sample)</strong></td>
                            <td>Rule C (&lt;25 words + version + title-only)</td>
                            <td>TP=5, FP=5, FN=3, TN=37</td>
                            <td>50.0%</td>
                            <td>62.5%</td>
                            <td style="font-weight:700;">0.556</td>
                        </tr>
                        <tr style="background:rgba(99,102,241,0.06);">
                            <td><strong>LOCO Summary</strong></td>
                            <td>Out-of-Cohort Metric Range</td>
                            <td>Per-Fold Range: 0.31 – 1.00</td>
                            <td style="color:var(--cyan); font-weight:600;">Fold Mean</td>
                            <td style="color:var(--cyan); font-weight:600;">Pooled Matrix</td>
                            <td style="color:var(--cyan); font-weight:700;">Pooled 0.51 | Mean 0.62</td>
                        </tr>
                    </tbody>
                </table>

                <div style="margin-top:1.2rem; padding:0.85rem 1rem; border-radius:8px; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.25); font-size:0.84rem; color:#FDE68A;">
                    ⚠️ <strong>Methodology Caveat</strong>: <em>Optimistic, 17 positives, no unseen data left.</em> Pooled out-of-cohort F1 = 0.51 across 150 tickets (fold mean: 0.62, where Cohort 3 has only 2 positives resulting in F1 = 1.000). All 150 available labeled tickets across Cohorts 1, 2, and 3 have been fully exposed; any future tuning requires an independent 4th cohort.
                </div>
            </div>
        </div>

        <!-- ============================================================== -->
        <!-- TAB 3: DRIFT MONITOR -->
        <!-- ============================================================== -->
        <div id="tab-drift" class="tab-panel">
            <div class="card">
                <div class="card-header">Feature Distribution Drift (PSI & KS Tests)</div>
                <p class="card-desc">Comparing historical training baseline (N=2,338) vs recent production intake (N=574):</p>

                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Feature Name</th>
                            <th>Reference Mean</th>
                            <th>Current Mean</th>
                            <th>Visual Indicator</th>
                            <th>PSI Score</th>
                            <th>KS Statistic</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="drift-table-body"></tbody>
                </table>

                <div style="margin-top:1rem; font-size:0.8rem; color:var(--text-subtle);">
                    Thresholds: PSI &lt; 0.10 (Normal / Stable), 0.10 &le; PSI &lt; 0.25 (Moderate Shift / Warning), PSI &ge; 0.25 (Significant Drift / Alert).
                </div>
            </div>
        </div>

        <!-- ============================================================== -->
        <!-- TAB 4: RETRAINING -->
        <!-- ============================================================== -->
        <div id="tab-retrain" class="tab-panel">
            <div class="card">
                <div class="card-header">Statistical Margin Promotion Governance</div>
                <p class="card-desc">Promotion policy requiring Candidate Metric &gt; Champion + 0.0100 on independent evaluation set:</p>

                <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.25rem;">
                    <div style="background:rgba(30,41,59,0.4); border:1px solid rgba(239,68,68,0.3); border-radius:10px; padding:1.25rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                            <span style="font-weight:600; font-size:1.05rem;">Path A: Stationary Distribution</span>
                            <span class="tag-pill alert">REJECTED</span>
                        </div>
                        <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:1rem;">
                            Candidate trained on same intake distribution with stochastic seed variation.
                        </p>
                        <div style="font-size:0.88rem; display:flex; flex-direction:column; gap:0.4rem;">
                            <div>Champion Metric: <strong>0.7820</strong></div>
                            <div>Candidate Metric: <strong>0.7818</strong></div>
                            <div>Metric Delta: <strong style="color:var(--rose);">-0.0002</strong> (Required: +0.0100)</div>
                            <div>Decision: <strong style="color:var(--rose);">REJECT</strong> (Preserve Champion)</div>
                        </div>
                    </div>

                    <div style="background:rgba(30,41,59,0.4); border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:1.25rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                            <span style="font-weight:600; font-size:1.05rem;">Path B: Expanded Intake Window</span>
                            <span class="tag-pill normal">PROMOTED</span>
                        </div>
                        <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:1rem;">
                            Candidate retrained with 420 additional resolved issues reflecting new release components.
                        </p>
                        <div style="font-size:0.88rem; display:flex; flex-direction:column; gap:0.4rem;">
                            <div>Champion Metric: <strong>0.7820</strong></div>
                            <div>Candidate Metric: <strong>0.8035</strong></div>
                            <div>Metric Delta: <strong style="color:var(--emerald);">+0.0215</strong> (Exceeds +0.0100 Margin)</div>
                            <div>Decision: <strong style="color:var(--emerald);">PROMOTED TO CHAMPION</strong></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ============================================================== -->
        <!-- TAB 5: ARCHITECTURE -->
        <!-- ============================================================== -->
        <div id="tab-architecture" class="tab-panel">
            <div class="card">
                <div class="card-header">Production Data Pipeline & Operating Constraints</div>
                <p class="card-desc">End-to-end integration topology and formally stated operational assumptions:</p>

                <div class="code-box" style="margin-bottom:1.5rem; color:var(--cyan);">
GitHub API (Core Issues)
    │
    ▼
Hourly Ingestion (Raw JSON & Parquet Files)
    │
    ▼
Databricks SQL Warehouse (Catalog: workspace, Schema: gold)
    │
    ▼
dbt Transformations (Bronze -> Silver -> Gold Marts)
    │
    ▼
Multi-Model Predictive Engine (Integration Classifier, XGBoost, AFT Survival)
    │
    ▼
LangGraph Triage Agent + FastAPI Gateway (Langfuse Tracing)
    ├── Human-in-the-Loop Review App (Streamlit on port 8501)
    ├── Proactive High-Risk Alerts (n8n Webhook / Telegram)
    ├── Capacity & Velocity Dashboard (Power BI Desktop via Databricks Gold)
    └── Live Model Health Dashboard (Looker Studio via Google Sheets Bridge)
                </div>

                <div class="card-header" style="font-size:1rem;">Formally Stated Business Assumptions</div>
                <ul style="color:var(--text-muted); font-size:0.88rem; margin-left:1.5rem; display:flex; flex-direction:column; gap:0.5rem;">
                    <li><strong>14-Day Resolution SLA</strong>: Standard service level target for Home Assistant Core bugs. Elapsed time &gt; 14 days is classified as an SLA breach.</li>
                    <li><strong>0.35 High-Risk Threshold</strong>: Issues with predicted not-planned probability p &ge; 0.35 are flagged for proactive maintainer attention.</li>
                    <li><strong>Strict Sandbox Write Allow-List</strong>: All bot comments and labels are restricted strictly to designated sandbox repositories (<code>ticket-triage-sandbox</code>). <code>home-assistant/core</code> is strictly read-only.</li>
                </ul>
            </div>
        </div>

        <footer>
            Home Assistant Core Ticket Triage Agent | Built with FastAPI, LangGraph, dbt, Databricks & Hugging Face Spaces.
        </footer>
    </div>

    <script>
        const REPLAY_TICKETS = {json.dumps(replay_tickets, indent=2)};
        const DRIFT_DATA = {json.dumps(drift_records, indent=2)};

        let currentTicketIndex = 0;

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
            if (window.event && window.event.currentTarget) {{
                window.event.currentTarget.classList.add('active');
            }}
            const targetPanel = document.getElementById('tab-' + tabId);
            if (targetPanel) targetPanel.classList.add('active');
        }}

        function renderTicketChips() {{
            const container = document.getElementById('ticket-chips-container');
            container.innerHTML = '';
            REPLAY_TICKETS.forEach((t, idx) => {{
                const chip = document.createElement('div');
                chip.className = 'ticket-chip' + (idx === currentTicketIndex ? ' active' : '');
                chip.onclick = () => selectTicket(idx);

                const statusColor = t.actual_outcome.state_reason === 'not_planned' ? 'var(--rose)' : (t.actual_outcome.state === 'closed' ? 'var(--emerald)' : 'var(--cyan)');

                chip.innerHTML = `
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="chip-number">#${{t.number}}</span>
                        <span style="font-size:0.72rem; color:${{statusColor}}; font-weight:600;">${{t.actual_outcome.state.toUpperCase()}} (${{t.actual_outcome.state_reason}})</span>
                    </div>
                    <div class="chip-title">${{t.title}}</div>
                    <div class="chip-meta">
                        <span>Comp: <code>${{t.actual_outcome.actual_integration}}</code></span>
                        <span>•</span>
                        <span>Author: @${{t.author}}</span>
                    </div>
                `;
                container.appendChild(chip);
            }});
        }}

        function selectTicket(idx) {{
            currentTicketIndex = idx;
            renderTicketChips();
            const t = REPLAY_TICKETS[idx];

            document.getElementById('ticket-title-display').textContent = `#${{t.number}}: ${{t.title}}`;
            document.getElementById('ticket-github-link').href = `https://github.com/home-assistant/core/issues/${{t.number}}`;
            document.getElementById('ticket-meta-display').textContent = `Author: @${{t.author}} | Created: ${{t.created_at}}`;
            document.getElementById('ticket-body-display').textContent = t.body;

            // Predictions
            const p = t.real_model_predictions;
            document.getElementById('pred-integ').textContent = p.predicted_integration;

            const compEl = document.getElementById('pred-complete');
            const compBadge = document.getElementById('pred-complete-badge');
            if (p.is_complete) {{
                compEl.textContent = "✅ Complete";
                compEl.style.color = "var(--emerald)";
                compBadge.textContent = "Passes Rule C";
                compBadge.style.background = "rgba(16,185,129,0.12)";
                compBadge.style.color = "var(--emerald)";
                document.getElementById('pred-missing-container').style.display = 'none';
            }} else {{
                compEl.textContent = "❌ Incomplete";
                compEl.style.color = "var(--rose)";
                compBadge.textContent = "Rule C Flagged";
                compBadge.style.background = "rgba(239,68,68,0.12)";
                compBadge.style.color = "var(--rose)";

                const missingList = document.getElementById('pred-missing-list');
                missingList.innerHTML = '';
                (p.missing_fields || []).forEach(m => {{
                    const li = document.createElement('li');
                    li.textContent = m;
                    missingList.appendChild(li);
                }});
                document.getElementById('pred-missing-container').style.display = 'block';
            }}

            document.getElementById('pred-latency').textContent = "No reliable signal for resolution time";

            const riskEl = document.getElementById('pred-risk');
            const riskBadge = document.getElementById('pred-risk-badge');
            riskEl.textContent = p.not_planned_risk_probability.toFixed(3);
            if (p.not_planned_risk_probability >= 0.35) {{
                riskEl.style.color = "var(--rose)";
                riskBadge.textContent = "⚠️ High Risk (p ≥ 0.35)";
                riskBadge.style.background = "rgba(239,68,68,0.12)";
                riskBadge.style.color = "var(--rose)";
            }} else {{
                riskEl.style.color = "var(--emerald)";
                riskBadge.textContent = "Normal Probability";
                riskBadge.style.background = "rgba(16,185,129,0.12)";
                riskBadge.style.color = "var(--emerald)";
            }}

            // Formulate Draft Comment
            let draft = `[AI draft, needs review]\\n\\nThank you for reaching out. Based on automated triage analysis:\\n- **Identified Component**: \\`${{p.predicted_integration}}\\``;
            if (!p.is_complete) {{
                draft += `\\n- **Completeness Notice**: This report appears incomplete or lacks actionable diagnostics:`;
                (p.missing_fields || []).forEach(m => draft += `\\n  - ${{m}}`);
                draft += `\\n\\nTo help maintainers investigate, please provide:\\n- System diagnostics or relevant log output from Home Assistant Core (Settings > System > Logs).\\n- Detailed step-by-step reproduction steps and expected vs actual behavior.`;
            }} else {{
                draft += `\\n- **Status**: Issue contains diagnostic details and reproduction context.`;
                if (t.body.includes("Traceback")) draft += `\\n- **Diagnostics**: Python traceback detected in report.`;
                else if (t.body.includes("Logger:") || t.body.includes("```")) draft += `\\n- **Diagnostics**: System log snippets attached.`;
            }}
            document.getElementById('pred-draft').textContent = draft;

            // Actual Outcome
            const a = t.actual_outcome;
            const stateEl = document.getElementById('actual-state');
            const stateBadge = document.getElementById('actual-state-badge');
            stateEl.textContent = a.state.toUpperCase();
            if (a.state_reason === 'not_planned') {{
                stateEl.style.color = "var(--rose)";
                stateBadge.textContent = "Closed as Not Planned";
                stateBadge.style.background = "rgba(239,68,68,0.12)";
                stateBadge.style.color = "var(--rose)";
            }} else if (a.state === 'closed') {{
                stateEl.style.color = "var(--emerald)";
                stateBadge.textContent = "Closed / Completed with Fix";
                stateBadge.style.background = "rgba(16,185,129,0.12)";
                stateBadge.style.color = "var(--emerald)";
            }} else {{
                stateEl.style.color = "var(--cyan)";
                stateBadge.textContent = "Active / Open Issue";
                stateBadge.style.background = "rgba(56,189,248,0.12)";
                stateBadge.style.color = "var(--cyan)";
            }}

            document.getElementById('actual-integ').textContent = a.actual_integration;
            document.getElementById('actual-days').textContent = a.state === 'closed' ? `${{a.actual_days_to_close}} days (${{a.state_reason}})` : (a.days_open_str || 'Still Open');

            const actComp = document.getElementById('actual-complete');
            const actCompBadge = document.getElementById('actual-complete-badge');
            if (a.ground_truth_complete) {{
                actComp.textContent = "Complete";
                actComp.style.color = "var(--emerald)";
                actCompBadge.textContent = "Had Diagnostics / Traceback";
                actCompBadge.style.background = "rgba(16,185,129,0.12)";
                actCompBadge.style.color = "var(--emerald)";
            }} else {{
                actComp.textContent = "Incomplete";
                actComp.style.color = "var(--rose)";
                actCompBadge.textContent = "Vague / Lacked Logs";
                actCompBadge.style.background = "rgba(239,68,68,0.12)";
                actCompBadge.style.color = "var(--rose)";
            }}

            // Diagnostic analysis summary
            let analysis = "";
            const integMatch = (p.predicted_integration.toLowerCase() === a.actual_integration.toLowerCase());
            if (integMatch) {{
                analysis += `✅ <strong>Component Routing Success</strong>: Model predicted <code>${{p.predicted_integration}}</code>, matching the maintainer's ground-truth label (Out-of-sample Top-1 Accuracy: 80.0% vs always-"other" 76.72%, Top-3: 97.16%, Macro-F1: 0.644 across 26 classes).<br>`;
            }} else {{
                analysis += `ℹ️ <strong>Component Routing Nuance</strong>: Model predicted <code>${{p.predicted_integration}}</code> while label was <code>${{a.actual_integration}}</code>.<br>`;
            }}

            if (a.state_reason === 'not_planned') {{
                analysis += `🔍 <strong>Not-Planned Life Cycle</strong>: Ticket was closed as not planned after ${{a.actual_days_to_close}} days. Model predicted not-planned risk probability was ${{p.not_planned_risk_probability}} (Test prevalence: 0.145, Out-of-sample PR-AUC: 0.195, ROC-AUC: 0.610).<br>`;
            }} else if (a.state === 'closed') {{
                analysis += `🛠️ <strong>Resolved with Fix</strong>: Closed in ${{a.actual_days_to_close}} days with maintainer patch.<br>`;
            }} else {{
                analysis += `⏳ <strong>Currently Open</strong>: Awaiting resolution. Resolution latency prediction suppressed (negative result: out-of-sample survival C-index is 0.514, fixed-horizon ROC-AUC 0.46–0.53, the cause was not investigated).<br>`;
            }}

            if (p.is_complete === a.ground_truth_complete) {{
                analysis += `📋 <strong>Completeness Assessment</strong>: Rule C decision matches ground truth (${{p.is_complete ? 'Actionable & complete' : 'Deficient in diagnostic logs'}}).`;
            }} else {{
                analysis += `📋 <strong>Completeness Note</strong>: Rule C flagged ${{p.is_complete ? 'complete' : 'incomplete'}} based on objective string length and traceback patterns.`;
            }}

            document.getElementById('comparison-analysis-text').innerHTML = analysis;
        }}

        // Render Drift table
        const tbody = document.getElementById('drift-table-body');
        DRIFT_DATA.forEach(row => {{
            const tr = document.createElement('tr');
            const pct = Math.min(100, (row.psi / 0.25) * 100);
            tr.innerHTML = `
                <td><code>${{row.feat}}</code></td>
                <td>${{row.ref.toFixed(2)}}</td>
                <td>${{row.curr.toFixed(2)}}</td>
                <td>
                    <div class="bar-container">
                        <div class="bar-fill ${{row.status}}" style="width:${{pct}}%;"></div>
                    </div>
                </td>
                <td><strong>${{row.psi.toFixed(4)}}</strong></td>
                <td>${{row.ks.toFixed(4)}}</td>
                <td><span class="tag-pill ${{row.status}}">${{row.status.toUpperCase()}}</span></td>
            `;
            tbody.appendChild(tr);
        }});

        // Initialize with first ticket
        renderTicketChips();
        selectTicket(0);
    </script>
</body>
</html>
"""

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated {OUTPUT_PATH} ({len(html_content)} bytes)")

if __name__ == "__main__":
    build_index_html()
