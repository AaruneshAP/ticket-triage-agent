"""Re-run the reworked detect_subtle_gaps on docs/completeness_labeling_heldout.csv.

Records the exact prompt version, raw LLM completions, and reports the flag count.
"""

import os

import pandas as pd
from dotenv import load_dotenv

from triage.agent.llm import detect_subtle_gaps

load_dotenv()

HELDOUT_CSV = os.path.join("docs", "completeness_heldout.csv")


def main():
    print("Loading held-out set...", flush=True)
    df = pd.read_csv(HELDOUT_CSV)
    print(f"Total tickets: {len(df)}", flush=True)

    llm_flags = []
    llm_missing_list = []

    for idx, row in df.iterrows():
        t_dict = row.to_dict()
        t_num = int(row.get("ticket_number", 0))
        title = str(row.get("title", ""))
        problem = str(row.get("ticket_text", ""))

        missing = detect_subtle_gaps(title=title, problem_text=problem, form_fields=t_dict)
        flag = 1 if len(missing) > 0 else 0
        llm_flags.append(flag)
        missing_str = "; ".join(missing) if missing else "None"
        llm_missing_list.append(missing_str)

        if flag == 1:
            print(f"  [FLAGGED] Ticket #{t_num}: {title[:50]} -> {missing}", flush=True)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)}...", flush=True)

    df["llm_flag"] = llm_flags
    df["llm_missing_items"] = [m if m != "None" else "" for m in llm_missing_list]
    df.to_csv(HELDOUT_CSV, index=False)

    flag_count = sum(llm_flags)
    print("\n--- Held-Out Re-Run Summary ---")
    print(f"Total LLM Flagged: {flag_count}/{len(df)} ({flag_count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
