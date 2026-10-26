"""Re-run refined completeness rules and subtle-gap detector on docs/completeness_labeling_heldout.csv.

Updates the CSV columns: rule_flag, rule_missing_items, llm_flag, llm_missing_items.
Computes and reports precision, recall, and F1 against human_label.
"""

import os

import pandas as pd
from dotenv import load_dotenv

from triage.agent.completeness import check_ticket_completeness_rules
from triage.agent.llm import detect_subtle_gaps

load_dotenv()

CSV_PATH = "docs/completeness_heldout.csv"
FEAT_PATH = "data/interim/features.parquet"


def main():
    print("Loading held-out CSV and features...", flush=True)
    df_csv = pd.read_csv(CSV_PATH)
    label_col = "label" if "label" in df_csv.columns else "human_label"
    df_feat = pd.read_parquet(FEAT_PATH) if os.path.exists(FEAT_PATH) else pd.DataFrame()

    print(f"Total tickets in held-out: {len(df_csv)}", flush=True)
    positives = df_csv[df_csv[label_col] == 1]["ticket_number"].tolist()
    print(f"Ground-truth positives ({len(positives)}): {positives}\n", flush=True)

    rule_flags = []
    rule_missings = []
    llm_flags = []
    llm_missings = []

    for idx, row in df_csv.iterrows():
        t = int(row["ticket_number"])
        row_dict = row.to_dict()

        # Merge with full feature text from features.parquet if available
        if not df_feat.empty and t in df_feat["ticket_number"].values:
            feat_row = df_feat[df_feat["ticket_number"] == t].iloc[0].to_dict()
            for k, v in feat_row.items():
                if v is not None and not pd.isna(v):
                    row_dict[k] = v

        # 1. Deterministic rules pass
        rule_missing = check_ticket_completeness_rules(row_dict)
        r_flag = 1 if len(rule_missing) > 0 else 0
        rule_flags.append(r_flag)
        rule_missings.append("; ".join(rule_missing) if rule_missing else "")

        # 2. LLM subtle-gap pass
        title = str(row_dict.get("title") or "")
        prob = str(row_dict.get("form_problem") or row_dict.get("body") or row_dict.get("ticket_text") or "")
        llm_missing = detect_subtle_gaps(title, prob, row_dict)
        l_flag = 1 if len(llm_missing) > 0 else 0
        llm_flags.append(l_flag)
        llm_missings.append("; ".join(llm_missing) if llm_missing else "")

    df_csv["rule_flag"] = rule_flags
    df_csv["rule_missing_items"] = rule_missings
    df_csv["llm_flag"] = llm_flags
    df_csv["llm_missing_items"] = llm_missings
    df_csv.to_csv(CSV_PATH, index=False)
    print("Updated docs/completeness_labeling_heldout.csv successfully!\n", flush=True)

    # Compute metrics
    for name, flag_col in [("Deterministic Rules", "rule_flag"), ("LLM Subtle-Gap", "llm_flag")]:
        tp = len(df_csv[(df_csv[flag_col] == 1) & (df_csv[label_col] == 1)])
        fp = len(df_csv[(df_csv[flag_col] == 1) & (df_csv[label_col] == 0)])
        fn = len(df_csv[(df_csv[flag_col] == 0) & (df_csv[label_col] == 1)])
        tn = len(df_csv[(df_csv[flag_col] == 0) & (df_csv[label_col] == 0)])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        flagged = df_csv[df_csv[flag_col] == 1]["ticket_number"].tolist()
        fps = df_csv[(df_csv[flag_col] == 1) & (df_csv[label_col] == 0)]["ticket_number"].tolist()
        fns = df_csv[(df_csv[flag_col] == 0) & (df_csv[label_col] == 1)]["ticket_number"].tolist()

        print(f"=== {name} ===")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"Flagged ({len(flagged)}): {flagged}")
        print(f"FPs: {fps}")
        print(f"FNs: {fns}\n")


if __name__ == "__main__":
    main()
