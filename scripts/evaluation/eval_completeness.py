"""Completeness Diagnostic Evaluation Script.

Evaluates rule_flag and llm_flag in a completeness CSV against ground-truth human_label.
Prints exact per-flag TP/FP/FN ticket-number lists, counts, precision, recall, and F1 score.

Usage:
    python scripts/evaluation/eval_completeness.py [--csv path/to/cohort.csv]
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd


def evaluate_flag(
    df: pd.DataFrame, flag_col: str, label_col: str = "label"
) -> Tuple[List[int], List[int], List[int], int, float, float, float]:
    """Compute exact TP/FP/FN ticket lists, TN count, and Precision/Recall/F1."""
    if label_col not in df.columns and "human_label" in df.columns:
        label_col = "human_label"
    # Filter to rows with valid ground-truth labels
    labeled = df[df[label_col].notna() & (df[label_col].astype(str).isin(["0", "1", "0.0", "1.0"]))].copy()
    y_true = labeled[label_col].astype(float).astype(int)
    y_pred = labeled[flag_col].fillna(0).astype(float).astype(int)

    tp_tickets = sorted(labeled[(y_true == 1) & (y_pred == 1)]["ticket_number"].astype(int).tolist())
    fp_tickets = sorted(labeled[(y_true == 0) & (y_pred == 1)]["ticket_number"].astype(int).tolist())
    fn_tickets = sorted(labeled[(y_true == 1) & (y_pred == 0)]["ticket_number"].astype(int).tolist())
    tn_count = int(((y_true == 0) & (y_pred == 0)).sum())

    tp_c = len(tp_tickets)
    fp_c = len(fp_tickets)
    fn_c = len(fn_tickets)

    precision = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
    recall = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return tp_tickets, fp_tickets, fn_tickets, tn_count, precision, recall, f1


def run_evaluation(csv_path: str) -> Dict[str, Dict]:
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else "human_label"
    total_rows = len(df)
    labeled_rows = len(df[df[label_col].notna() & (df[label_col].astype(str).isin(["0", "1", "0.0", "1.0"]))])
    positives = int(df[label_col].astype(float).sum()) if labeled_rows > 0 else 0

    print("=" * 80)
    print(f"COMPLETENESS EVALUATION REPORT: {csv_path}")
    print(f"Total Tickets: {total_rows} | Labeled: {labeled_rows} | Ground-Truth Incomplete: {positives}")
    print("=" * 80)

    results = {}

    for flag_col in ["rule_flag", "llm_flag"]:
        if flag_col not in df.columns:
            print(f"Notice: Column '{flag_col}' not found in CSV.", file=sys.stderr)
            continue

        tp, fp, fn, tn, p, r, f1 = evaluate_flag(df, flag_col)
        results[flag_col] = {
            "tp_tickets": tp,
            "fp_tickets": fp,
            "fn_tickets": fn,
            "tn_count": tn,
            "precision": p,
            "recall": r,
            "f1": f1,
        }

        print(f"\n--- Flag: {flag_col} ---")
        print(f"Confusion Matrix: TP={len(tp)}, FP={len(fp)}, FN={len(fn)}, TN={tn}")
        print(f"Precision:        {p:.4f} ({len(tp)} / {len(tp)+len(fp)})")
        print(f"Recall:           {r:.4f} ({len(tp)} / {len(tp)+len(fn)})")
        print(f"F1 Score:         {f1:.4f}")
        print(f"True Positives  ({len(tp)}): {tp}")
        print(f"False Positives ({len(fp)}): {fp}")
        print(f"False Negatives ({len(fn)}): {fn}")

    # Combined Rule OR LLM Flag
    if "rule_flag" in df.columns and "llm_flag" in df.columns:
        comb_df = df.copy()
        r_f = comb_df["rule_flag"].fillna(0).astype(float).astype(int)
        l_f = comb_df["llm_flag"].fillna(0).astype(float).astype(int)
        comb_df["combined_flag"] = ((r_f == 1) | (l_f == 1)).astype(int)
        tp, fp, fn, tn, p, r, f1 = evaluate_flag(comb_df, "combined_flag")
        results["combined_flag"] = {
            "tp_tickets": tp,
            "fp_tickets": fp,
            "fn_tickets": fn,
            "tn_count": tn,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
        print("\n--- Combined (rule_flag OR llm_flag) ---")
        print(f"Confusion Matrix: TP={len(tp)}, FP={len(fp)}, FN={len(fn)}, TN={tn}")
        print(f"Precision:        {p:.4f} ({len(tp)} / {len(tp)+len(fp)})")
        print(f"Recall:           {r:.4f} ({len(tp)} / {len(tp)+len(fn)})")
        print(f"F1 Score:         {f1:.4f}")
        print(f"True Positives  ({len(tp)}): {tp}")
        print(f"False Positives ({len(fp)}): {fp}")
        print(f"False Negatives ({len(fn)}): {fn}")

    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate completeness flags against human labels.")
    default_csv = (
        "docs/completeness_heldout.csv"
        if os.path.exists("docs/completeness_heldout.csv")
        else "docs/completeness_labeling_heldout.csv"
    )
    parser.add_argument("--csv", default=default_csv, help="Path to completeness cohort CSV")
    args = parser.parse_args()
    run_evaluation(args.csv)
