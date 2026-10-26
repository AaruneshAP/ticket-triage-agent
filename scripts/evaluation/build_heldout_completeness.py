"""Generate docs/completeness_labeling_heldout.csv on a fresh held-out set of 50 newer tickets.

Tickets df.iloc[80:130] from data/interim/features.parquet.
Columns: ticket_number, title, ticket_text, form_install_type, form_version, has_traceback,
         rule_flag, rule_missing_items, llm_flag, llm_missing_items, human_label, human_notes.
human_label and human_notes are left empty.
"""

import os

import pandas as pd
from dotenv import load_dotenv

from triage.agent.completeness import check_ticket_completeness_rules
from triage.agent.llm import detect_subtle_gaps

load_dotenv()

FEATURES_PATH = os.path.join("data", "interim", "features.parquet")
HELDOUT_CSV = os.path.join("docs", "completeness_heldout.csv")


def main():
    print("Loading features...", flush=True)
    df = pd.read_parquet(FEATURES_PATH)
    heldout_df = df.iloc[80:130].copy()
    print(f"Held-out set size: {len(heldout_df)}, tickets {heldout_df['ticket_number'].min()} to {heldout_df['ticket_number'].max()}", flush=True)

    records = []
    rule_flags = []
    llm_flags = []

    for idx, (_, row) in enumerate(heldout_df.iterrows()):
        t_dict = row.to_dict()
        t_num = int(row.get("ticket_number", row.get("number", 0)))
        title = str(row.get("title", ""))
        problem = str(row.get("clean_problem_text", row.get("body", "")))[:600]
        ticket_text_snippet = problem.replace("\r", " ").replace("\n", " ").strip()[:300]

        # Run rules without manual inspection
        rules_missing = check_ticket_completeness_rules(t_dict)
        rule_flag = 1 if len(rules_missing) > 0 else 0

        # Run LLM subtle gaps check
        llm_missing = detect_subtle_gaps(title=title, problem_text=problem, form_fields=t_dict)
        llm_flag = 1 if len(llm_missing) > 0 else 0

        rule_flags.append(rule_flag)
        llm_flags.append(llm_flag)

        records.append({
            "ticket_number": t_num,
            "title": title,
            "ticket_text": ticket_text_snippet,
            "form_install_type": row.get("form_install_type", "unknown"),
            "form_version": row.get("form_version", "unknown"),
            "has_traceback": int(row.get("has_traceback", 0)),
            "rule_flag": rule_flag,
            "rule_missing_items": "; ".join(rules_missing) if rules_missing else "None",
            "llm_flag": llm_flag,
            "llm_missing_items": "; ".join(llm_missing) if llm_missing else "None",
            "label": "",
            "notes": "",
        })
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/50 tickets...", flush=True)

    out_df = pd.DataFrame(records)
    out_df.to_csv(HELDOUT_CSV, index=False)
    print(f"Saved {len(out_df)} held-out tickets to {HELDOUT_CSV}", flush=True)

    rule_count = sum(rule_flags)
    llm_count = sum(llm_flags)
    both_count = sum(1 for r_flag, l_flag in zip(rule_flags, llm_flags) if r_flag == 1 and l_flag == 1)
    either_count = sum(1 for r_flag, l_flag in zip(rule_flags, llm_flags) if r_flag == 1 or l_flag == 1)

    print("\n--- Held-Out Set Summary (N=50) ---")
    print(f"Rule Flagged:   {rule_count}/50 ({rule_count/50*100:.1f}%)")
    print(f"LLM Flagged:    {llm_count}/50 ({llm_count/50*100:.1f}%)")
    print(f"Both Flagged:   {both_count}/50 ({both_count/50*100:.1f}%)")
    print(f"Either Flagged: {either_count}/50 ({either_count/50*100:.1f}%)")


if __name__ == "__main__":
    main()
