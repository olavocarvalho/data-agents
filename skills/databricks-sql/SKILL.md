---
name: databricks-sql
description: Write advanced Databricks SQL queries using Spark-specific features not available in standard SQL. Use when working with arrays, maps, nested data, JSON/variant, window functions with QUALIFY, Delta Lake time travel, higher-order functions (TRANSFORM, FILTER), EXPLAIN query plans, or when the user asks about Databricks-specific SQL capabilities.
last-updated: 2026-01-25
---

# Databricks SQL - Spark-Specific Features

When writing Databricks SQL queries, leverage these powerful features that go beyond standard SQL.

> **This skill is optimized for SQL Warehouses** (Serverless, Pro, or Classic). Some features have version or tier requirements noted inline.

## Quick Reference

| Feature | Use Case | Standard SQL Alternative |
|---------|----------|-------------------------|
| `EXPLAIN [MODE]` | Analyze query execution plan | Limited EXPLAIN |
| `TRANSFORM(array, x -> ...)` | Transform each element | Explode + collect (slow) |
| `FILTER(array, x -> ...)` | Filter array elements | Explode + where + collect |
| `QUALIFY` | Filter window results | Subquery with window |
| `VARIANT` | Semi-structured JSON | String + JSON functions |
| Time Travel | Query historical data | Not possible |
| `explode()` / `inline()` | Unnest arrays/structs | LATERAL JOIN (limited) |

## Guidelines

1. **Prefer higher-order functions over explode/collect patterns** - They're faster and preserve ordering
2. **Use QUALIFY instead of subqueries** for filtering window function results
3. **Use VARIANT for flexible JSON** - Better performance than string parsing
4. **Leverage Delta Lake time travel** for historical analysis and debugging
5. **Use EXPLAIN to understand and optimize queries** - Essential for performance tuning

## EXPLAIN - Query Plan Analysis

Understand how the query engine executes your query:

```sql
-- Basic: show physical plan only
EXPLAIN SELECT * FROM sales WHERE amount > 100;

-- Extended: full pipeline (parsed → analyzed → optimized → physical)
EXPLAIN EXTENDED SELECT * FROM sales WHERE amount > 100;

-- Cost: logical plan with statistics (row counts, sizes)
EXPLAIN COST SELECT * FROM sales WHERE amount > 100;

-- Formatted: readable hierarchical structure with node details
EXPLAIN FORMATTED SELECT * FROM sales WHERE amount > 100;
```

| Mode | Shows | Use When |
|------|-------|----------|
| (default) | Physical plan | Quick check of execution strategy |
| `EXTENDED` | All plan stages | Debugging optimization issues |
| `COST` | Plan + statistics | Understanding data volume estimates |
| `FORMATTED` | Structured output | Detailed node-by-node analysis |

> **Note:** `EXPLAIN CODEGEN` exists but is less useful on SQL Warehouses since they use the Photon engine rather than Spark's code generation.

**Key things to look for:**
- `Exchange` = shuffle (expensive network operation)
- `BroadcastHashJoin` = good for small tables
- `SortMergeJoin` = used for large-large joins
- `Filter` pushed down = good (predicate pushdown)
- `PartitionFilters` = partition pruning working

## Higher-Order Functions

Transform arrays without exploding:

```sql
-- Transform: apply function to each element
SELECT TRANSFORM(values, x -> x * 2) AS doubled
FROM my_table;

-- Filter: keep elements matching predicate
SELECT FILTER(scores, x -> x >= 70) AS passing_scores
FROM students;

-- Aggregate: reduce array to single value
SELECT AGGREGATE(numbers, 0, (acc, x) -> acc + x) AS total
FROM my_table;

-- Combine multiple arrays element-wise
SELECT ZIP_WITH(arr1, arr2, (x, y) -> x + y) AS combined
FROM my_table;

-- Check conditions
SELECT EXISTS(items, x -> x.status = 'FAILED') AS has_failures,
       FORALL(items, x -> x.validated) AS all_validated
FROM orders;
```

## Array & Map Operations

```sql
-- Flatten nested arrays
SELECT FLATTEN(nested_array) FROM my_table;

-- Array creation and manipulation
SELECT ARRAY(1, 2, 3) AS arr,
       ARRAY_CONTAINS(arr, 2) AS has_two,
       ARRAY_DISTINCT(arr) AS unique_vals,
       ARRAY_UNION(arr1, arr2) AS merged,
       ARRAY_EXCEPT(arr1, arr2) AS diff,
       ARRAY_INTERSECT(arr1, arr2) AS common,
       ARRAY_SIZE(arr) AS len;

-- Explode to rows (use in FROM clause, not SELECT)
SELECT t.id, e.col AS value
FROM my_table t, LATERAL explode(t.values) AS e;

-- Explode with position
SELECT t.id, e.pos, e.col AS value
FROM my_table t, LATERAL posexplode(t.values) AS e;

-- Inline: explode array of structs
SELECT t.id, e.*
FROM my_table t, LATERAL inline(t.struct_array) AS e;

-- Collect back to array (use with GROUP BY)
SELECT id,
       collect_list(value) AS all_values,
       collect_set(value) AS unique_values
FROM exploded_table
GROUP BY id;

-- Map operations
SELECT MAP('a', 1, 'b', 2) AS my_map,
       MAP_KEYS(my_map) AS keys,
       MAP_VALUES(my_map) AS values,
       MAP_FILTER(my_map, (k, v) -> v > 1) AS filtered;
```

## QUALIFY - Filter Window Functions

Filter window function results without subqueries:

```sql
-- Get latest record per customer (cleaner than subquery)
SELECT *
FROM orders
QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1;

-- Top 3 products per category
SELECT category, product_name, revenue
FROM products
QUALIFY RANK() OVER (PARTITION BY category ORDER BY revenue DESC) <= 3;

-- Records where running total exceeds threshold
SELECT *
FROM transactions
QUALIFY SUM(amount) OVER (ORDER BY txn_date) > 10000;

-- Deduplicate keeping latest
SELECT * FROM raw_events
QUALIFY ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ingestion_time DESC) = 1;
```

## VARIANT - Semi-Structured Data

Query flexible JSON without predefined schema:

```sql
-- Parse JSON to VARIANT
SELECT PARSE_JSON('{"name": "Alice", "scores": [85, 92, 78]}') AS data;

-- Safe parsing (returns NULL on error)
SELECT TRY_PARSE_JSON(json_column) AS data FROM raw_events;

-- Query variant fields (use : operator)
SELECT data:name::STRING AS name,
       data:scores[0]::INT AS first_score,
       data:address:city::STRING AS city
FROM events;

-- Get schema of variant
SELECT SCHEMA_OF_VARIANT(data) FROM events LIMIT 1;

-- Aggregate schemas across rows
SELECT SCHEMA_OF_VARIANT_AGG(data) FROM events;

-- Explode variant arrays
SELECT t.id, e.col:field::STRING AS field_value
FROM my_table t, LATERAL explode(t.variant_array) AS e;
```

## Delta Lake Time Travel

Query historical versions for analysis and debugging:

```sql
-- Query specific version
SELECT * FROM my_table VERSION AS OF 5;
SELECT * FROM my_table@v5;  -- Shorthand

-- Query by timestamp
SELECT * FROM my_table TIMESTAMP AS OF '2024-01-15 10:00:00';

-- View table history
DESCRIBE HISTORY my_table;

-- Compare versions (what changed?)
SELECT * FROM my_table VERSION AS OF 10
EXCEPT
SELECT * FROM my_table VERSION AS OF 5;

-- Count changes between versions
SELECT
  (SELECT COUNT(*) FROM my_table VERSION AS OF 10) AS current_count,
  (SELECT COUNT(*) FROM my_table VERSION AS OF 5) AS previous_count;
```

## AI Functions

> **SQL Warehouse requirement:** Serverless or Pro only (NOT available on Classic). Pro requires PrivateLink enabled.

```sql
-- Summarize text
SELECT ai_query('summarize', feedback_text) AS summary
FROM customer_feedback;

-- Classify sentiment
SELECT ai_query('classify', review, ARRAY('positive', 'negative', 'neutral')) AS sentiment
FROM reviews;

-- Extract structured data
SELECT ai_query('extract',
  document_text,
  'Extract: customer_name, order_id, total_amount'
) AS extracted
FROM documents;
```

## Recursive CTEs

> **Requires:** DBSQL 2025.20+ (recently added feature)

Navigate hierarchical data:

```sql
-- Org chart traversal
WITH RECURSIVE org_tree AS (
  -- Base: top-level managers
  SELECT employee_id, name, manager_id, 1 AS level
  FROM employees
  WHERE manager_id IS NULL

  UNION ALL

  -- Recursive: employees under each manager
  SELECT e.employee_id, e.name, e.manager_id, t.level + 1
  FROM employees e
  JOIN org_tree t ON e.manager_id = t.employee_id
)
SELECT * FROM org_tree;

-- Bill of materials explosion
WITH RECURSIVE bom AS (
  SELECT part_id, component_id, quantity, 1 AS depth
  FROM parts WHERE part_id = 'FINAL_PRODUCT'

  UNION ALL

  SELECT p.part_id, p.component_id, p.quantity * b.quantity, b.depth + 1
  FROM parts p
  JOIN bom b ON p.part_id = b.component_id
)
SELECT * FROM bom;
```

## Pipe Syntax (|>)

Chain operations for readability:

```sql
-- Traditional
SELECT category, SUM(amount) AS total
FROM (SELECT * FROM sales WHERE year = 2024)
GROUP BY category
HAVING SUM(amount) > 1000
ORDER BY total DESC;

-- Pipe syntax (more readable)
FROM sales
|> WHERE year = 2024
|> SELECT category, SUM(amount) AS total
|> GROUP BY category
|> HAVING total > 1000
|> ORDER BY total DESC;
```

## Common Analytical Patterns

### Sessionization
```sql
SELECT *,
  SUM(new_session) OVER (PARTITION BY user_id ORDER BY event_time) AS session_id
FROM (
  SELECT *,
    CASE WHEN event_time - LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time)
         > INTERVAL 30 MINUTES THEN 1 ELSE 0 END AS new_session
  FROM events
);
```

### Running totals with reset
```sql
SELECT *,
  SUM(amount) OVER (
    PARTITION BY customer_id, grp
    ORDER BY txn_date
  ) AS running_total
FROM (
  SELECT *,
    SUM(CASE WHEN reset_flag THEN 1 ELSE 0 END)
      OVER (PARTITION BY customer_id ORDER BY txn_date) AS grp
  FROM transactions
);
```

### Pivot array to columns
```sql
SELECT id,
       arr[0] AS first,
       arr[1] AS second,
       arr[2] AS third
FROM my_table;
```

For complete function reference, see [reference.md](reference.md).
