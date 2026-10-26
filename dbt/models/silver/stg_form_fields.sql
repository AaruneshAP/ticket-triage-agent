{{ config(materialized='view') }}

WITH raw AS (
    SELECT * FROM {{ ref('raw_issues') }}
)

SELECT
    ticket_number,
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
FROM raw
