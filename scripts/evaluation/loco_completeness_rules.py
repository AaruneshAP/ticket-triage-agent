"""Leave-One-Cohort-Out (LOCO) Cross-Validation for Completeness Rule Selection.

Protocol:
- Fold 1: Train on Cohorts 1+2, select optimal rule, test on Cohort 3.
- Fold 2: Train on Cohorts 1+3, select optimal rule, test on Cohort 2.
- Fold 3: Train on Cohorts 2+3, select optimal rule, test on Cohort 1.
- Report all three test results and their average.
"""

import os
import re

import numpy as np
import pandas as pd

from triage.parse.form_parser import parse_issue

# 1. Load the 3 cohorts
c1 = pd.read_csv("docs/completeness_insample.csv")
c2 = pd.read_csv("docs/completeness_heldout.csv")
c3 = pd.read_csv("docs/completeness_unseen_cohort3.csv")

c1["cohort_idx"] = 1
c2["cohort_idx"] = 2
c3["cohort_idx"] = 3

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

    is_title_only = bool(re.search(r"\b(as in the title|see title|same as title|title says it all|as above)\b", cl, re.I)) or (word_count <= 5 and "title" in cl.lower())
    has_repro_keywords = bool(re.search(r"\b(steps to reproduce|repro|reproduce|expected behavior|steps:)\b", r["body"], re.I))
    has_tb = bool(p.get("has_traceback", False))
    has_lg = bool(p.get("has_logs", False))
    has_err_snippet = bool(re.search(r"Traceback|Logger:|TypeError|KeyError|ValueError|AttributeError|Exception:", r["body"], re.I))

    v_raw = str(p.get("version") or r.get("form_version") or "").strip().lower()
    is_ver_missing = v_raw in ["", "none", "nan", "unknown", "_no response_"] or bool(re.search(r"\b(all versions|all|any|latest|current|na|n/a)\b", v_raw))

    is_screenshot = ("<img" in p_text or "![" in p_text) and word_count < 20 and not (has_tb or has_lg or has_err_snippet)

    records.append({
        "ticket_number": r["ticket_number"],
        "cohort_idx": r["cohort_idx"],
        "label": int(r["label"] if "label" in r else r["human_label"]),
        "word_count": word_count,
        "char_count": char_count,
        "is_title_only": is_title_only,
        "has_repro_keywords": has_repro_keywords,
        "has_any_diagnostics": has_tb or has_lg or has_err_snippet,
        "is_ver_missing": is_ver_missing,
        "is_screenshot": is_screenshot,
    })

df = pd.DataFrame(records)

# Candidate rules
candidates = {
    "Rule A (<50w or <250c)": lambda r: 1 if (r["is_ver_missing"] or r["is_screenshot"] or (not r["has_any_diagnostics"] and (r["word_count"] < 50 or r["char_count"] < 250))) else 0,
    "Rule B (<20w)": lambda r: 1 if (r["is_ver_missing"] or r["is_title_only"] or r["is_screenshot"] or (not r["has_any_diagnostics"] and r["word_count"] < 20)) else 0,
    "Rule C (<25w)": lambda r: 1 if (r["is_ver_missing"] or r["is_title_only"] or r["is_screenshot"] or (not r["has_any_diagnostics"] and r["word_count"] < 25)) else 0,
    "Rule D (<30w)": lambda r: 1 if (r["is_ver_missing"] or r["is_title_only"] or r["is_screenshot"] or (not r["has_any_diagnostics"] and r["word_count"] < 30)) else 0,
    "Rule E (Multi-tier)": lambda r: 1 if (r["is_ver_missing"] or r["is_title_only"] or r["is_screenshot"] or (not r["has_any_diagnostics"] and (r["word_count"] < 20 or (r["word_count"] < 35 and not r["has_repro_keywords"])))) else 0,
}

def calc_metrics(sub_df, rule_fn):
    preds = sub_df.apply(rule_fn, axis=1)
    y = sub_df["label"]
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "p": p, "r": r, "f1": f1}

def main():
    print("================================================================================")
    print("LEAVE-ONE-COHORT-OUT (LOCO) CROSS-VALIDATION FOR COMPLETENESS RULES")
    print("================================================================================")

    test_cohorts = [3, 2, 1]
    loco_results = []

    for test_c in test_cohorts:
        train_c = [c for c in [1, 2, 3] if c != test_c]
        train_df = df[df["cohort_idx"].isin(train_c)]
        test_df = df[df["cohort_idx"] == test_c]

        print(f"\n--- FOLD (Train Cohorts: {train_c}, Test Cohort: {test_c}) ---")
        print("Candidate Rule Selection on Training Cohorts:")
        best_rule_name = None
        best_train_f1 = -1
        best_rule_fn = None

        for name, fn in candidates.items():
            m = calc_metrics(train_df, fn)
            print(f"  {name:25s}: TP={m['tp']:2d}, FP={m['fp']:2d}, FN={m['fn']:2d} | P={m['p']:.3f}, R={m['r']:.3f}, F1={m['f1']:.3f}")
            if m["f1"] > best_train_f1:
                best_train_f1 = m["f1"]
                best_rule_name = name
                best_rule_fn = fn

        print(f"  => Selected Rule: {best_rule_name} (Train F1 = {best_train_f1:.3f})")

        # Test on the held-out cohort
        test_m = calc_metrics(test_df, best_rule_fn)
        pos_cnt = int(test_df["label"].sum())
        print(f"Out-of-Distribution Test on Cohort {test_c} (N={len(test_df)}, {pos_cnt} positives):")
        print(f"  TP={test_m['tp']}, FP={test_m['fp']}, FN={test_m['fn']}, TN={test_m['tn']} | Precision={test_m['p']:.3f}, Recall={test_m['r']:.3f}, F1={test_m['f1']:.3f}")

        loco_results.append({
            "test_cohort": test_c,
            "train_cohorts": str(train_c),
            "selected_rule": best_rule_name,
            "train_f1": best_train_f1,
            "tp": test_m["tp"],
            "fp": test_m["fp"],
            "fn": test_m["fn"],
            "tn": test_m["tn"],
            "precision": test_m["p"],
            "recall": test_m["r"],
            "f1": test_m["f1"],
        })

    print("\n" + "="*80)
    print("LOCO RESULTS SUMMARY TABLE")
    print("="*80)
    print(f"{'Test Cohort':<15} {'Selected Rule':<25} {'TP/FP/FN/TN':<16} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 86)

    prec_list = [r["precision"] for r in loco_results]
    rec_list = [r["recall"] for r in loco_results]
    f1_list = [r["f1"] for r in loco_results]

    for r in loco_results:
        cm = f"{r['tp']}/{r['fp']}/{r['fn']}/{r['tn']}"
        print(f"Cohort {r['test_cohort']:<8} {r['selected_rule']:<25} {cm:<16} {r['precision']:.3f}      {r['recall']:.3f}     {r['f1']:.3f}")

    print("-" * 86)
    unweighted_p = float(np.mean(prec_list))
    unweighted_r = float(np.mean(rec_list))
    unweighted_f1 = float(np.mean(f1_list))
    print(f"{'Unweighted Mean':<41} {'-':<16} {unweighted_p:.3f}      {unweighted_r:.3f}     {unweighted_f1:.3f}")

    total_tp = sum(r["tp"] for r in loco_results)
    total_fp = sum(r["fp"] for r in loco_results)
    total_fn = sum(r["fn"] for r in loco_results)
    total_tn = sum(r["tn"] for r in loco_results)
    pooled_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    pooled_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    pooled_f1 = 2 * pooled_p * pooled_r / (pooled_p + pooled_r) if (pooled_p + pooled_r) > 0 else 0.0
    pooled_cm = f"{total_tp}/{total_fp}/{total_fn}/{total_tn}"
    print(f"{'Pooled Total':<41} {pooled_cm:<16} {pooled_p:.3f}      {pooled_r:.3f}     {pooled_f1:.3f}")
    print("="*80)

if __name__ == "__main__":
    main()
