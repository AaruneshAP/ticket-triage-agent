"""Hermetic unit tests for incremental scoring checkpoint advancement and persistence."""

import json

import pandas as pd

from triage.models.scoring import score_new_tickets


def _create_synthetic_tickets(start_num: int, count: int) -> pd.DataFrame:
    """Generate hermetic synthetic ticket records for testing."""
    records = []
    for i in range(count):
        num = start_num + i
        records.append({
            "number": num,
            "ticket_number": num,
            "title": f"Synthetic Test Issue #{num}",
            "body": f"### The problem\nSynthetic test issue description for ticket {num} with repro details.\n\n### What version of Home Assistant Core has the issue?\ncore-2026.8.0\n\n### Integration causing the issue\nswitch",
            "created_at": f"2026-10-29T08:{i:02d}:00Z",
            "updated_at": f"2026-10-29T08:{i:02d}:00Z",
            "state": "open",
            "integration": "switch",
        })
    return pd.DataFrame(records)


def test_scoring_checkpoint_advancement_and_idempotence(tmp_path):
    """Test that:
    1. Adding N synthetic tickets scores exactly N and advances the checkpoint.
    2. Running scoring again immediately scores 0 new tickets and preserves the checkpoint.
    3. Adding M more tickets scores only the M new tickets and advances the checkpoint again.
    """
    issues_path = str(tmp_path / "issues.parquet")
    ckpt_path = str(tmp_path / "scoring_checkpoint.json")
    scored_path = str(tmp_path / "scored_tickets.jsonl")
    models_dir = "models"  # Use workspace models if present, or fallback gracefully

    n_initial = 5
    initial_df = _create_synthetic_tickets(start_num=200001, count=n_initial)
    initial_df.to_parquet(issues_path)

    # --- RUN 1: Score initial N tickets ---
    print(f"\n[Test Run 1] Scoring {n_initial} synthetic tickets (#200001 - #200005)...")
    res1 = score_new_tickets(
        issues_path=issues_path,
        scoring_ckpt_path=ckpt_path,
        scored_out_path=scored_path,
        models_dir=models_dir,
        default_batch_size=n_initial,
    )
    print(f"[Test Run 1 Result] Scored count: {res1['scored_count']}")
    assert res1["status"] == "success"
    assert res1["scored_count"] == n_initial

    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt1 = json.load(f)
    print(f"[Test Run 1 Checkpoint] Advanced to: {ckpt1['last_scored_ticket_number']}")
    assert ckpt1["last_scored_ticket_number"] == 200005
    assert ckpt1["total_scored_in_run"] == n_initial

    with open(scored_path, "r", encoding="utf-8") as f:
        lines1 = [line.strip() for line in f if line.strip()]
    assert len(lines1) == n_initial

    # --- RUN 2: Re-run immediately without new tickets ---
    print("\n[Test Run 2] Immediate second run without new tickets...")
    res2 = score_new_tickets(
        issues_path=issues_path,
        scoring_ckpt_path=ckpt_path,
        scored_out_path=scored_path,
        models_dir=models_dir,
    )
    print(f"[Test Run 2 Result] Scored count: {res2['scored_count']}")
    assert res2["status"] == "success"
    assert res2["scored_count"] == 0, f"Expected 0 scored tickets on re-run, got {res2['scored_count']}"

    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt2 = json.load(f)
    print(f"[Test Run 2 Checkpoint] Preserved at: {ckpt2['last_scored_ticket_number']}")
    assert ckpt2["last_scored_ticket_number"] == 200005
    assert ckpt2["total_scored_in_run"] == 0

    with open(scored_path, "r", encoding="utf-8") as f:
        lines2 = [line.strip() for line in f if line.strip()]
    assert len(lines2) == n_initial, "Output jsonl should not have duplicate records"

    # --- RUN 3: Add M=3 new synthetic tickets ---
    m_new = 3
    additional_df = _create_synthetic_tickets(start_num=200006, count=m_new)
    combined_df = pd.concat([initial_df, additional_df], ignore_index=True)
    combined_df.to_parquet(issues_path)

    print(f"\n[Test Run 3] Added {m_new} new synthetic tickets (#200006 - #200008)...")
    res3 = score_new_tickets(
        issues_path=issues_path,
        scoring_ckpt_path=ckpt_path,
        scored_out_path=scored_path,
        models_dir=models_dir,
    )
    print(f"[Test Run 3 Result] Scored count: {res3['scored_count']}")
    assert res3["status"] == "success"
    assert res3["scored_count"] == m_new, f"Expected {m_new} scored tickets, got {res3['scored_count']}"

    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt3 = json.load(f)
    print(f"[Test Run 3 Checkpoint] Advanced to: {ckpt3['last_scored_ticket_number']}")
    assert ckpt3["last_scored_ticket_number"] == 200008
    assert ckpt3["total_scored_in_run"] == m_new

    with open(scored_path, "r", encoding="utf-8") as f:
        lines3 = [line.strip() for line in f if line.strip()]
    assert len(lines3) == n_initial + m_new
