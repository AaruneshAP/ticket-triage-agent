-- Test that all distinct ticket numbers in raw.raw_issues are present in silver.stg_issues
-- Fails if any ticket_number in raw is missing from silver
SELECT DISTINCT r.ticket_number
FROM {{ source('databricks_raw', 'raw_issues') }} r
LEFT JOIN {{ ref('stg_issues') }} s ON r.ticket_number = s.ticket_number
WHERE s.ticket_number IS NULL
