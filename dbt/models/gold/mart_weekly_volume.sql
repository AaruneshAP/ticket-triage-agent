{{ config(materialized='table') }}

WITH tickets AS (
    SELECT * FROM {{ ref('fct_tickets') }}
)

SELECT
    date_trunc('week', created_at) AS week_start,
    count(*) AS total_tickets,
    count(CASE WHEN integration != 'unlabeled' THEN 1 END) AS labeled_tickets,
    count(CASE WHEN state = 'closed' THEN 1 END) AS closed_tickets,
    count(CASE WHEN is_closed_without_fix = 1 THEN 1 END) AS not_planned_tickets,
    count(CASE WHEN closed_by_bot = TRUE THEN 1 END) AS bot_closed_tickets,
    round(percentile_approx(CASE WHEN is_closed_event = 1 THEN days_to_close_or_censor END, 0.5), 2) AS median_days_to_close
FROM tickets
GROUP BY 1
ORDER BY 1 DESC
