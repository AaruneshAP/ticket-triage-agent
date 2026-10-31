---
title: Home Assistant Ticket Triage Agent
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: apache-2.0
short_description: Issue triage, completeness check & MLOps demo
---

# Home Assistant Core | Ticket Triage Agent Demo

This Space provides an interactive, standalone demonstration of the **Ticket Triage Agent** (portfolio project and reference implementation) built for Home Assistant Core.

## Key Capabilities Demonstrated:
1. **Empirical Replay Demonstration**: Live side-by-side comparison of real model predictions (XGBoost, Survival AFT, TF-IDF Recommender, Rule C Engine) against actual GitHub resolution outcomes on real Home Assistant Core test tickets (zero heuristic constants).
2. **Offline Calibration & Performance**: Rigorous out-of-sample metrics strictly matching `docs/evaluation.md` (Survival C-Index: strictly out-of-sample 0.514 [0.492, 0.536], validation (tuning) split 0.5426; Not-Planned Risk: strictly out-of-sample ROC-AUC 0.610, PR-AUC 0.195 vs test prevalence 0.145, validation (tuning) PR-AUC 0.222; Label Recommender: strictly out-of-sample Top-1 80.0% vs always-"other" baseline 76.72%, Top-3 97.16%, Macro-F1 0.644, validation (tuning) Top-1 86.6%, Macro-F1 0.832; Completeness LOCO F1 range 0.31 to 1.00, conservative mean 0.62, pooled 0.51 - flagged optimistic, 17 positives, no unseen data left; Fixed-Horizon: 14-day under-predicts with mean 0.32 vs 0.44 observed and 0.11 vs 0.34 in lowest bin, 30-day is non-monotonic and over-confident at top with 0.86 vs 0.59 observed; strictly out-of-sample ROC-AUC 0.4621 and 0.5329).
3. **Distribution Drift Monitoring**: Feature stability tracking via Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests.
4. **Retraining Governance**: Statistical margin-gated model promotion policy (demonstrating stationary rejection vs expanded window promotion).

*All demonstrations in this Space operate over curated repository snapshot datasets and do not expose live warehouse credentials.*
