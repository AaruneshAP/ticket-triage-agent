{{ config(
    materialized='incremental',
    unique_key='ticket_number',
    incremental_strategy='merge'
) }}

WITH raw AS (
    SELECT * FROM {{ ref('raw_issues') }}
    {% if is_incremental() %}
    WHERE updated_at >= (SELECT date_sub(coalesce(max(updated_at), to_timestamp('1970-01-01')), 7) FROM {{ this }})
       OR ticket_number NOT IN (SELECT ticket_number FROM {{ this }})
    {% endif %}
)

SELECT
    ticket_number,
    title,
    body,
    state,
    coalesce(state_reason, 'unspecified') AS state_reason,
    integration,
    author,
    author_type,
    author_association,
    created_at,
    closed_at,
    updated_at,
    closed_by,
    closed_by_bot,
    -- Survival analysis duration calculation (in fractional days)
    CASE
        WHEN state = 'closed' AND closed_at IS NOT NULL THEN
            (unix_timestamp(closed_at) - unix_timestamp(created_at)) / 86400.0
        ELSE
            (unix_timestamp(current_timestamp()) - unix_timestamp(created_at)) / 86400.0
    END AS days_to_close_or_censor,
    -- Event indicator for survival analysis: 1 if closed, 0 if censored (still open)
    CASE WHEN state = 'closed' THEN 1 ELSE 0 END AS is_closed_event,
    -- Resolution quality target
    CASE
        WHEN state = 'closed' AND lower(state_reason) IN ('not_planned', 'duplicate') THEN 1
        WHEN state = 'closed' THEN 0
        ELSE NULL
    END AS is_closed_without_fix
FROM raw
