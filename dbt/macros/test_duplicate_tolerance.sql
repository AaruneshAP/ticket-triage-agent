{% test duplicate_tolerance(model, column_name, max_duplicates=0) %}

WITH validation AS (
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT {{ column_name }}) AS distinct_rows
    FROM {{ model }}
)
SELECT *
FROM validation
WHERE (total_rows - distinct_rows) > {{ max_duplicates }}

{% endtest %}
