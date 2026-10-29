import os

import pandas as pd

os.makedirs("tests/fixtures", exist_ok=True)

issues_data = {
    "ticket_number": [1001, 1002],
    "title": ["Shelly device disconnects", "Hue lamp unreachable"],
    "body": ["Sample body 1", "Sample body 2"],
    "state": ["closed", "open"],
    "state_reason": ["completed", None],
    "labels": ['["integration: shelly"]', '["integration: hue"]'],
    "integration": ["shelly", "hue"],
    "author": ["alice", "bob"],
    "author_type": ["User", "User"],
    "author_association": ["NONE", "CONTRIBUTOR"],
    "assignee": [None, None],
    "comments_count": [2, 0],
    "created_at": ["2026-05-01T12:00:00Z", "2026-05-02T12:00:00Z"],
    "closed_at": ["2026-05-03T12:00:00Z", None],
    "updated_at": ["2026-05-03T12:00:00Z", "2026-05-02T12:00:00Z"],
    "closed_by": ["home-assistant[bot]", None],
    "closed_by_bot": [True, False],
    "is_template": [True, True],
    "form_problem": ["Problem 1", "Problem 2"],
    "form_version": ["2026.5.0", "2026.5.1"],
    "ha_version_year": [2026, 2026],
    "ha_version_month": [5, 5],
    "ha_version_patch": [0, 1],
    "has_last_working_version": [True, False],
    "form_last_working_version": ["2026.4.0", None],
    "form_install_type": ["Home Assistant OS", "Home Assistant Container"],
    "form_integration": ["shelly", "hue"],
    "has_traceback": [False, True],
    "has_logs": [True, False],
    "n_code_blocks": [1, 0],
    "n_urls": [0, 1],
    "n_missing_form_fields": [0, 1],
    "refers_to_issue": [False, False],
}
df_issues = pd.DataFrame(issues_data)
df_issues.to_parquet("tests/fixtures/sample_issues.parquet", index=False)

comments_data = {
    "comment_id": [5001, 5002],
    "issue_number": [1001, 1001],
    "author": ["charlie", "home-assistant[bot]"],
    "author_association": ["NONE", "BOT"],
    "is_bot": [False, True],
    "body": ["Can you provide diagnostics?", "Duplicate of #999"],
    "created_at": ["2026-05-01T13:00:00Z", "2026-05-03T12:00:00Z"],
    "updated_at": ["2026-05-01T13:00:00Z", "2026-05-03T12:00:00Z"],
    "duplicate_of_issue": [None, 999],
}
df_comments = pd.DataFrame(comments_data)
df_comments.to_parquet("tests/fixtures/sample_comments.parquet", index=False)
print("Fixtures created successfully!")
