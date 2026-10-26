{{ config(materialized='view') }}

WITH source AS (
    SELECT * FROM {{ source('databricks_raw', 'raw_comments') }}
),
deduped AS (
    SELECT *,
           row_number() OVER (PARTITION BY comment_id ORDER BY coalesce(updated_at, created_at) DESC) AS rn
    FROM source
)

SELECT
    comment_id,
    issue_number,
    author,
    author_association,
    is_bot,
    body,
    to_timestamp(created_at) AS created_at,
    to_timestamp(updated_at) AS updated_at,
    duplicate_of_issue
FROM deduped
WHERE rn = 1

