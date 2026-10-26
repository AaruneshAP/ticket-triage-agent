{{ config(materialized='view') }}

WITH source AS (
    SELECT * FROM {{ source('databricks_raw', 'raw_issues') }}
),
deduped AS (
    SELECT *,
           row_number() OVER (PARTITION BY ticket_number ORDER BY coalesce(updated_at, created_at) DESC) AS rn
    FROM source
)

SELECT
    ticket_number,
    title,
    body,
    state,
    state_reason,
    labels,
    integration,
    author,
    author_type,
    author_association,
    assignee,
    comments_count,
    to_timestamp(created_at) AS created_at,
    to_timestamp(closed_at) AS closed_at,
    to_timestamp(updated_at) AS updated_at,
    closed_by,
    closed_by_bot,
    is_template,
    form_problem,
    form_version,
    ha_version_year,
    ha_version_month,
    ha_version_patch,
    has_last_working_version,
    form_last_working_version,
    form_install_type,
    form_integration,
    has_traceback,
    has_logs,
    n_code_blocks,
    n_urls,
    n_missing_form_fields,
    refers_to_issue
FROM deduped
WHERE rn = 1

