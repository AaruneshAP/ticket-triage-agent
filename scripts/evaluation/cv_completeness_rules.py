"""Stratified 5-Fold Cross-Validation for Completeness Rule Selection on Pooled N=150 Dataset.

Candidate Features Evaluated:
- Word count thresholds: 10, 15, 20, 25, 30, 40, 50
- has_logs, has_traceback
- Repro-step keywords (steps to reproduce, repro, expected, behavior)
- Missing / invalid Core version
- Title-only terse descriptions ('as in the title', 'see title')
- Screenshot-only reports
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from triage.parse.form_parser import parse_issue

# 1. Load the pooled dataset (150 tickets, 17 positives)
c1 = pd.read_csv("docs/completeness_insample.csv")
c2 = pd.read_csv("docs/completeness_heldout.csv")
c3 = pd.read_csv("docs/completeness_unseen_cohort3.csv")

c1["cohort"] = "Cohort 1 (In-sample, N=50, 8 pos)"
c2["cohort"] = "Cohort 2 (Held-out, N=50, 7 pos)"
c3["cohort"] = "Cohort 3 (Unseen, N=50, 2 pos)"

pooled_df = pd.concat([c1, c2, c3], ignore_index=True)

# Fetch full text from interim/issues.parquet if available
issues_path = "data/interim/issues.parquet"
if os.path.exists(issues_path):
    idf = pd.read_parquet(issues_path)
    t_map = {int(r["number"]): (str(r.get("body", "")), str(r.get("title", ""))) for _, r in idf.iterrows()}
    bodies, titles = [], []
    for _, r in pooled_df.iterrows():
        t_num = int(r["ticket_number"])
        if t_num in t_map:
            bodies.append(t_map[t_num][0])
            titles.append(t_map[t_num][1])
        else:
            bodies.append(str(r["ticket_text"]))
            titles.append(str(r["title"]))
    pooled_df["body"] = bodies
    pooled_df["title"] = titles
else:
    pooled_df["body"] = pooled_df["ticket_text"]

# 2. Extract feature signals
records = []
for _, r in pooled_df.iterrows():
    p = parse_issue(r["body"], r["title"])
    p_text = str(p.get("problem") or "")
    cl = re.sub(r"<img[\s\S]*?>", "", p_text)
    cl = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", cl)
    cl = re.sub(r"https?://\S+", "", cl)
    cl = re.sub(r"###\s+[A-Za-z0-9\s\?]+\s*", "", cl).strip()
    words = [w for w in cl.split() if len(w) > 1]
    word_count = len(words)
    char_count = len(cl)

    # Title-only or empty description check
    is_title_only = bool(re.search(r"\b(as in the title|see title|same as title|title says it all|as above)\b", cl, re.I)) or (word_count <= 5 and "title" in cl.lower())

    # Repro keywords check
    has_repro_keywords = bool(re.search(r"\b(steps to reproduce|repro|reproduce|expected behavior|steps:)\b", r["body"], re.I))

    # Logs & traceback
    has_tb = bool(p.get("has_traceback", False))
    has_lg = bool(p.get("has_logs", False))
    has_err_snippet = bool(re.search(r"Traceback|Logger:|TypeError|KeyError|ValueError|AttributeError|Exception:", r["body"], re.I))

    # Version check (missing or invalid)
    v_raw = str(p.get("version") or r.get("form_version") or "").strip().lower()
    is_ver_missing = v_raw in ["", "none", "nan", "unknown", "_no response_"] or bool(re.search(r"\b(all versions|all|any|latest|current|na|n/a)\b", v_raw))

    # Integration check
    integ = str(p.get("integration_in_form") or r.get("integration") or "").strip().lower()
    is_integ_missing = integ in ["", "none", "nan", "unknown", "_no response_", "unlabeled"]

    # Screenshot only
    is_screenshot = ("<img" in p_text or "![" in p_text) and word_count < 20 and not (has_tb or has_lg or has_err_snippet)

    records.append({
        "ticket_number": r["ticket_number"],
        "label": int(r["label"] if "label" in r else r["human_label"]),
        "word_count": word_count,
        "char_count": char_count,
        "is_title_only": is_title_only,
        "has_repro_keywords": has_repro_keywords,
        "has_traceback": has_tb,
        "has_logs": has_lg,
        "has_err_snippet": has_err_snippet,
        "has_any_diagnostics": has_tb or has_lg or has_err_snippet,
        "is_ver_missing": is_ver_missing,
        "is_integ_missing": is_integ_missing,
        "is_screenshot": is_screenshot,
        "raw_text": cl,
    })

feat_df = pd.DataFrame(records)
y = feat_df["label"].values

# 3. Define candidate rule rule-generators
def eval_rule(rule_fn, df, y_arr, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(df), dtype=int)
    fold_f1s = []

    for train_idx, val_idx in skf.split(df, y_arr):
        val_df = df.iloc[val_idx]
        preds = np.array([rule_fn(row) for _, row in val_df.iterrows()], dtype=int)
        oof_preds[val_idx] = preds
        fold_f1 = f1_score(y_arr[val_idx], preds, zero_division=0)
        fold_f1s.append(fold_f1)

    tp = int(((oof_preds == 1) & (y_arr == 1)).sum())
    fp = int(((oof_preds == 1) & (y_arr == 0)).sum())
    fn = int(((oof_preds == 0) & (y_arr == 1)).sum())
    tn = int(((oof_preds == 0) & (y_arr == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # 1000-sample bootstrap for 95% CI on out-of-fold predictions
    np.random.seed(seed)
    b_prec, b_rec, b_f1 = [], [], []
    for _ in range(1000):
        b_idx = np.random.choice(len(y_arr), size=len(y_arr), replace=True)
        b_tp = ((oof_preds[b_idx] == 1) & (y_arr[b_idx] == 1)).sum()
        b_fp = ((oof_preds[b_idx] == 1) & (y_arr[b_idx] == 0)).sum()
        b_fn = ((oof_preds[b_idx] == 0) & (y_arr[b_idx] == 1)).sum()
        p_val = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
        r_val = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
        f_val = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        b_prec.append(p_val)
        b_rec.append(r_val)
        b_f1.append(f_val)

    ci_f1 = (np.percentile(b_f1, 2.5), np.percentile(b_f1, 97.5))
    ci_p = (np.percentile(b_prec, 2.5), np.percentile(b_prec, 97.5))
    ci_r = (np.percentile(b_rec, 2.5), np.percentile(b_rec, 97.5))

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "ci_f1": (round(ci_f1[0], 4), round(ci_f1[1], 4)),
        "ci_p": (round(ci_p[0], 4), round(ci_p[1], 4)),
        "ci_r": (round(ci_r[0], 4), round(ci_r[1], 4)),
        "fold_f1_mean": round(float(np.mean(fold_f1s)), 4),
        "oof_preds": oof_preds,
    }


# Evaluate candidate rule configurations
candidates = {}

# Baseline 1: Prior tuned rule (<50 words or <250 chars without logs)
def rule_prior_tuned(row):
    if row["is_ver_missing"]:
        return 1
    if row["is_screenshot"]:
        return 1
    if not row["has_any_diagnostics"] and (row["word_count"] < 50 or row["char_count"] < 250):
        return 1
    return 0
candidates["Rule A: Prior Tuned (<50 words or <250 chars)"] = rule_prior_tuned

# Candidate B: Conservative Word Count (<20 words without logs) + missing version
def rule_word_20(row):
    if row["is_ver_missing"]:
        return 1
    if row["is_title_only"]:
        return 1
    if row["is_screenshot"]:
        return 1
    if not row["has_any_diagnostics"] and row["word_count"] < 20:
        return 1
    return 0
candidates["Rule B: Word Count < 20 words + Version + Title-Only"] = rule_word_20

# Candidate C: Word Count < 25 words + missing version + title-only
def rule_word_25(row):
    if row["is_ver_missing"]:
        return 1
    if row["is_title_only"]:
        return 1
    if row["is_screenshot"]:
        return 1
    if not row["has_any_diagnostics"] and row["word_count"] < 25:
        return 1
    return 0
candidates["Rule C: Word Count < 25 words + Version + Title-Only"] = rule_word_25

# Candidate D: Word Count < 30 words + missing version + title-only
def rule_word_30(row):
    if row["is_ver_missing"]:
        return 1
    if row["is_title_only"]:
        return 1
    if row["is_screenshot"]:
        return 1
    if not row["has_any_diagnostics"] and row["word_count"] < 30:
        return 1
    return 0
candidates["Rule D: Word Count < 30 words + Version + Title-Only"] = rule_word_30

# Candidate E: Multi-tier Rule (<20 words OR (<35 words without repro keywords or diagnostics))
def rule_multitier(row):
    if row["is_ver_missing"]:
        return 1
    if row["is_title_only"]:
        return 1
    if row["is_screenshot"]:
        return 1
    if not row["has_any_diagnostics"]:
        if row["word_count"] < 20:
            return 1
        if row["word_count"] < 35 and not row["has_repro_keywords"]:
            return 1
    return 0
candidates["Rule E: Multi-tier (<20 words OR <35 words without repro keywords)"] = rule_multitier

# Run CV on all candidate rules
results = []
for name, fn in candidates.items():
    res = eval_rule(fn, feat_df, y)
    results.append({
        "rule_name": name,
        "tp": res["tp"], "fp": res["fp"], "fn": res["fn"], "tn": res["tn"],
        "precision": res["precision"], "recall": res["recall"], "f1": res["f1"],
        "ci_f1": res["ci_f1"],
        "ci_p": res["ci_p"],
        "ci_r": res["ci_r"],
    })

print("\n=== STRATIFIED 5-FOLD CROSS-VALIDATION ON POOLED N=150 TICKETS ===")
for r in results:
    print(f"\n{r['rule_name']}:")
    print(f"  Confusion Matrix: TP={r['tp']}, FP={r['fp']}, FN={r['fn']}, TN={r['tn']}")
    print(f"  Precision: {r['precision']:.4f} (95% CI: [{r['ci_p'][0]:.4f}, {r['ci_p'][1]:.4f}])")
    print(f"  Recall:    {r['recall']:.4f} (95% CI: [{r['ci_r'][0]:.4f}, {r['ci_r'][1]:.4f}])")
    print(f"  F1 Score:  {r['f1']:.4f} (95% CI: [{r['ci_f1'][0]:.4f}, {r['ci_f1'][1]:.4f}])")
