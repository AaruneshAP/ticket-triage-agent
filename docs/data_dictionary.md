# Data Dictionary: Ticket Triage Intelligence System

This document formalizes the schemas, field definitions, data types, and temporal safety boundaries across all layers (RAW, Silver, Gold, and Feature Store).

## 1. Temporal Safety & Leakage Guard Policy

To prevent temporal data leakage in predictive modeling, features are strictly partitioned:
- **Creation-Time Features (Allowed)**: Information known at ticket creation timestamp (`created_at`).
- **Forbidden Target / Post-Creation Columns**: `comments`, `comments_count`, `assignee`, `state`, `state_reason`, `closed_at`, `closed_by`, `closed_by_bot`, `updated_at`, `edits`.
- **Target Leakage Prevention**: When training integration label classifiers, the `Integration causing the issue` heading and text is stripped via `clean_text_for_label_prediction()` because the upstream bot copies this form field into the label.

## 2. Feature Store Schema (`fct_tickets` / `data/interim/features.parquet`)

| Column Name | Type | Temporal Boundary | Description |
|---|---|---|---|
| `ticket_number` | BIGINT | Creation | Unique GitHub issue number identifier |
| `created_at` | TIMESTAMP | Creation | UTC timestamp when issue was submitted |
| `title_len` | INT | Creation | Length of issue title in characters |
| `body_len` | INT | Creation | Length of issue markdown body in characters |
| `n_code_blocks` | INT | Creation | Count of triple-backtick (```) code/log blocks |
| `n_urls` | INT | Creation | Count of hyperlinks in body |
| `has_traceback` | BOOLEAN | Creation | True if traceback headers (`Traceback (most recent call last)`) detected |
| `has_logs` | BOOLEAN | Creation | True if log prefixes (e.g. `ERROR`, `WARNING`) detected |
| `refers_to_issue` | BOOLEAN | Creation | True if issue references `#N` or URL to another issue |
| `n_missing_form_fields` | INT | Creation | Count of standard template headings containing `_No response_` or empty |
| `has_last_working_version` | BOOLEAN | Creation | True if user provided a previous working version |
| `ha_version_year` | INT | Creation | Extracted major release year of Home Assistant Core |
| `ha_version_month` | INT | Creation | Extracted minor release month of Home Assistant Core |
| `ha_version_patch` | INT | Creation | Extracted patch release number of Home Assistant Core |
| `version_gap_months` | FLOAT | Creation | Months lag behind latest release seen up to ticket creation |
| `install_type` | STRING | Creation | Reported environment (`Home Assistant OS`, `Container`, `Supervised`, `Core`) |
| `author_association` | STRING | Creation | Relationship of author (`MEMBER`, `CONTRIBUTOR`, `NONE`) |
| `created_hour` | INT | Creation | Hour of day (0-23 UTC) |
| `created_day_of_week` | INT | Creation | Day of week (0=Monday, 6=Sunday) |
| `is_weekend` | INT | Creation | 1 if created Saturday or Sunday, else 0 |
| `author_prior_tickets` | INT | Creation | Count of prior tickets opened by this author strictly before this ticket |
| `integration_volume_90d` | INT | Creation | Count of tickets opened for this integration in preceding 90 days |
| `integration_median_close_prior` | FLOAT | Creation | Median days to close for tickets of this integration closed BEFORE this creation |
| `clean_problem_text` | STRING | Creation | Cleaned problem section text for embedding and TF-IDF |
| `clean_text_for_label` | STRING | Creation | Text with integration section stripped for leak-free label prediction |
| `target_days_to_close_or_censor` | FLOAT | Target / Post | Event duration or right-censored observation window (Days) |
| `target_is_closed_event` | INT | Target / Post | Survival event indicator (1 if closed, 0 if censored/open) |
| `target_is_not_planned` | INT | Target / Post | 1 if closed as `not_planned` or `duplicate`, 0 if resolved with fix, NULL if open |
| `target_integration` | STRING | Target / Post | Downstream bot-assigned integration label |
