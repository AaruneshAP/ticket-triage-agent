{{ config(materialized='table') }}

WITH tickets AS (
    SELECT * FROM {{ ref('fct_tickets') }}
)

SELECT
    date_trunc('day', created_at) AS date_day,
    count(*) AS ticket_volume,
    round(avg(body_len), 1) AS avg_body_len,
    round(avg(n_code_blocks), 2) AS avg_code_blocks,
    round(avg(CASE WHEN has_traceback THEN 1.0 ELSE 0.0 END), 3) AS traceback_rate,
    round(avg(CASE WHEN integration = 'unlabeled' THEN 1.0 ELSE 0.0 END), 3) AS unlabeled_rate,
    round(avg(n_missing_form_fields), 2) AS avg_missing_fields,
    round(avg(CASE WHEN is_closed_without_fix = 1 THEN 1.0 ELSE 0.0 END), 3) AS not_planned_rate
FROM tickets
GROUP BY 1
ORDER BY 1 DESC
