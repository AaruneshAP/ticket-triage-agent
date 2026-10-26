{{ config(materialized='view') }}

WITH raw AS (
    SELECT ticket_number, labels
    FROM {{ ref('raw_issues') }}
    WHERE labels IS NOT NULL AND labels != ''
),
exploded AS (
    SELECT
        ticket_number,
        explode(split(labels, '\\|')) AS label_name
    FROM raw
)

SELECT
    ticket_number,
    trim(label_name) AS label_name,
    CASE WHEN lower(trim(label_name)) LIKE 'integration:%' THEN 1 ELSE 0 END AS is_integration_label
FROM exploded
WHERE trim(label_name) != ''
