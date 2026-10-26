{{ config(materialized='view') }}

WITH raw AS (
    SELECT * FROM {{ ref('raw_comments') }}
)

SELECT
    comment_id,
    issue_number,
    author,
    author_association,
    is_bot,
    body,
    created_at,
    updated_at,
    duplicate_of_issue,
    CASE WHEN duplicate_of_issue IS NOT NULL THEN 1 ELSE 0 END AS has_duplicate_reference
FROM raw
