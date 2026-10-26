{{ config(
    materialized='incremental',
    unique_key='ticket_number',
    incremental_strategy='merge'
) }}

WITH issues AS (
    SELECT * FROM {{ ref('stg_issues') }}
    {% if is_incremental() %}
    WHERE updated_at >= (SELECT date_sub(coalesce(max(created_at), to_timestamp('1970-01-01')), 7) FROM {{ this }})
       OR ticket_number NOT IN (SELECT ticket_number FROM {{ this }})
    {% endif %}
),
forms AS (
    SELECT * FROM {{ ref('stg_form_fields') }}
)

SELECT
    i.ticket_number,
    -- Creation-time textual and metadata features (STRICTLY NO POST-CREATION LEAKAGE)
    length(coalesce(i.title, '')) AS title_len,
    length(coalesce(i.body, '')) AS body_len,
    coalesce(f.n_code_blocks, 0) AS n_code_blocks,
    coalesce(f.n_urls, 0) AS n_urls,
    coalesce(f.has_traceback, FALSE) AS has_traceback,
    coalesce(f.has_logs, FALSE) AS has_logs,
    coalesce(f.refers_to_issue, FALSE) AS refers_to_issue,
    coalesce(f.n_missing_form_fields, 0) AS n_missing_form_fields,
    coalesce(f.has_last_working_version, FALSE) AS has_last_working_version,
    f.ha_version_year,
    f.ha_version_month,
    f.ha_version_patch,
    coalesce(f.form_install_type, 'unknown') AS install_type,
    coalesce(i.integration, 'unlabeled') AS integration,
    i.author_association,
    hour(i.created_at) AS created_hour,
    dayofweek(i.created_at) AS created_day_of_week,
    CASE WHEN dayofweek(i.created_at) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
    i.created_at,

    -- Modeling Targets (Excluded from feature matrices via leakage_guard)
    i.state,
    i.state_reason,
    i.days_to_close_or_censor,
    i.is_closed_event,
    i.is_closed_without_fix,
    i.closed_by_bot
FROM issues i
LEFT JOIN forms f ON i.ticket_number = f.ticket_number
