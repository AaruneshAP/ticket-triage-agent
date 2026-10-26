import re

import pandas as pd

df_csv = pd.read_csv("docs/completeness_heldout.csv")
t_nums = df_csv["ticket_number"].tolist()
incomplete_set = {176576, 176864, 177225, 177329, 177363, 177617, 178141}

df_feat = pd.read_parquet("data/interim/features.parquet")
df_sub = df_feat[df_feat["ticket_number"].isin(t_nums)].copy()

err_regex = re.compile(
    r"Traceback|Logger:|File \".*?\", line|TypeError|KeyError|ValueError|AttributeError|Exception:|fatal:|API returned error code",
    re.I,
)

for threshold in [40, 45, 48, 50, 55]:
    flagged = []
    for _, row in df_sub.iterrows():
        t = int(row["ticket_number"])

        # 1. Version check
        raw_ver = str(row.get("form_version") or "").strip()
        body = str(row.get("body") or "")
        prob = str(row.get("form_problem") or body)

        # Extract from system info block if present
        sys_match = re.search(
            r"version\s*\|\s*(core-20\d{2}[.-]\d{1,2}(?:\.[a-z0-9]+)?|20\d{2}[.-]\d{1,2}(?:\.[a-z0-9]+)?)",
            raw_ver,
            re.I,
        )
        if not sys_match:
            sys_match = re.search(
                r"version\s*\|\s*(core-20\d{2}[.-]\d{1,2}(?:\.[a-z0-9]+)?|20\d{2}[.-]\d{1,2}(?:\.[a-z0-9]+)?)",
                body,
                re.I,
            )

        if sys_match:
            ha_ver = sys_match.group(1).lower()
        else:
            ha_ver = raw_ver.lower()

        is_bad_ver = False
        if ha_ver and ha_ver not in ["nan", "none", "unknown", "_no response_", ""]:
            if re.search(r"\b(all versions|all|any|latest|current|na|n/a|not sure|recent)\b", ha_ver):
                is_bad_ver = True
            elif not re.search(r"20\d{2}[.-]\d{1,2}|core-20\d{2}[.-]\d{1,2}", ha_ver):
                is_bad_ver = True
        else:
            if not re.search(r"20\d{2}[.-]\d{1,2}|core-20\d{2}[.-]\d{1,2}|0\.\d{2}", body):
                is_bad_ver = True

        # 2. Short text check: very few words, no log, no error
        cleaned = re.sub(r"<img[\s\S]*?>", "", prob)
        cleaned = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", cleaned)
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        cleaned = re.sub(r"###\s+The problem\s*(?:\([^)]*\))?\s*", "", cleaned)
        cleaned = re.sub(r"###\s+[A-Za-z0-9\s\?]+\s*", "", cleaned).strip()
        words = [w for w in cleaned.split() if len(w) > 1]

        has_tb = bool(row.get("has_traceback")) or bool(err_regex.search(body))
        logs_str = str(row.get("logs") or "")
        has_real_logs = len(logs_str.strip()) > 30 or bool(err_regex.search(logs_str))

        is_short = (len(words) < threshold) and not has_tb and not has_real_logs

        if is_bad_ver or is_short:
            flagged.append(t)

    tp = len(set(flagged) & incomplete_set)
    fp = len(set(flagged) - incomplete_set)
    fn = len(incomplete_set - set(flagged))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"Threshold {threshold}: TP={tp}, FP={fp}, FN={fn}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
    print(f"  Flagged ({len(flagged)}): {sorted(flagged)}")
    fps = sorted(list(set(flagged) - incomplete_set))
    fns = sorted(list(incomplete_set - set(flagged)))
    print(f"  FPs: {fps}")
    print(f"  FNs: {fns}\n")
