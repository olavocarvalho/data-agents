# Databricks SQL Function Reference

## EXPLAIN Modes

| Mode | Syntax | Description |
|------|--------|-------------|
| Default | `EXPLAIN stmt` | Physical plan only |
| Extended | `EXPLAIN EXTENDED stmt` | Parsed → Analyzed → Optimized → Physical |
| Cost | `EXPLAIN COST stmt` | Logical plan with statistics |
| Formatted | `EXPLAIN FORMATTED stmt` | Hierarchical structure with node details |

> **Note:** `EXPLAIN CODEGEN` is less useful on SQL Warehouses (Photon engine doesn't use Spark codegen).

### Common Plan Operators

| Operator | Meaning | Performance Impact |
|----------|---------|-------------------|
| `Exchange` | Shuffle data across nodes | Expensive (network I/O) |
| `BroadcastExchange` | Broadcast small table | Good for small joins |
| `HashAggregate` | Hash-based aggregation | Efficient |
| `SortAggregate` | Sort-based aggregation | More memory needed |
| `BroadcastHashJoin` | Join with broadcasted table | Fast for small tables |
| `SortMergeJoin` | Sort both sides then merge | For large-large joins |
| `ShuffledHashJoin` | Hash join after shuffle | Medium tables |
| `Filter` | Row filtering | Check if pushed down |
| `Project` | Column selection | Minimal cost |
| `LocalTableScan` | Read local data | Fast |
| `FileScan` | Read from files | Check partition pruning |

### What to Look For

- **Predicate pushdown**: `Filter` should appear close to `Scan`
- **Partition pruning**: `PartitionFilters` in `FileScan`
- **Broadcast threshold**: Small tables should use `BroadcastHashJoin`
- **Shuffle reduction**: Minimize `Exchange` operations
- **Skew**: Uneven partition sizes cause slow tasks

## Higher-Order Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `TRANSFORM` | `TRANSFORM(array, x -> expr)` | Apply expression to each element |
| `FILTER` | `FILTER(array, x -> predicate)` | Keep elements where predicate is true |
| `AGGREGATE` | `AGGREGATE(array, init, (acc, x) -> merge)` | Reduce array to single value |
| `ZIP_WITH` | `ZIP_WITH(arr1, arr2, (x, y) -> expr)` | Combine two arrays element-wise |
| `EXISTS` | `EXISTS(array, x -> predicate)` | True if any element matches |
| `FORALL` | `FORALL(array, x -> predicate)` | True if all elements match |
| `REDUCE` | `REDUCE(array, init, (acc, x) -> merge, acc -> finish)` | Reduce with final transform |

## Array Functions

| Function | Description |
|----------|-------------|
| `ARRAY(...)` | Create array from values |
| `ARRAY_APPEND(arr, elem)` | Add element to end |
| `ARRAY_COMPACT(arr)` | Remove NULL values |
| `ARRAY_CONTAINS(arr, elem)` | Check if element exists |
| `ARRAY_DISTINCT(arr)` | Remove duplicates |
| `ARRAY_EXCEPT(arr1, arr2)` | Elements in arr1 not in arr2 |
| `ARRAY_INTERSECT(arr1, arr2)` | Common elements |
| `ARRAY_JOIN(arr, delimiter)` | Join to string |
| `ARRAY_MAX(arr)` | Maximum value |
| `ARRAY_MIN(arr)` | Minimum value |
| `ARRAY_POSITION(arr, elem)` | Position of element (1-indexed) |
| `ARRAY_REMOVE(arr, elem)` | Remove all occurrences |
| `ARRAY_REPEAT(elem, count)` | Create array with repeated element |
| `ARRAY_SIZE(arr)` | Number of elements |
| `ARRAY_SORT(arr)` | Sort ascending |
| `ARRAY_UNION(arr1, arr2)` | Union without duplicates |
| `ARRAYS_OVERLAP(arr1, arr2)` | True if any common elements |
| `ARRAYS_ZIP(arr1, arr2, ...)` | Zip into array of structs |
| `CONCAT(arr1, arr2)` | Concatenate arrays |
| `ELEMENT_AT(arr, index)` | Get element (1-indexed, negative from end) |
| `FLATTEN(nested_arr)` | Flatten one level of nesting |
| `REVERSE(arr)` | Reverse order |
| `SEQUENCE(start, stop, step)` | Generate sequence array |
| `SHUFFLE(arr)` | Random shuffle |
| `SLICE(arr, start, length)` | Extract subarray |
| `SORT_ARRAY(arr, asc)` | Sort with direction |

## Map Functions

| Function | Description |
|----------|-------------|
| `MAP(k1, v1, k2, v2, ...)` | Create map from pairs |
| `MAP_CONCAT(map1, map2)` | Merge maps |
| `MAP_CONTAINS_KEY(map, key)` | Check if key exists |
| `MAP_ENTRIES(map)` | Array of key-value structs |
| `MAP_FILTER(map, (k, v) -> pred)` | Filter by predicate |
| `MAP_FROM_ARRAYS(keys, values)` | Create from two arrays |
| `MAP_FROM_ENTRIES(arr)` | Create from array of structs |
| `MAP_KEYS(map)` | Array of keys |
| `MAP_VALUES(map)` | Array of values |
| `TRANSFORM_KEYS(map, (k, v) -> expr)` | Transform keys |
| `TRANSFORM_VALUES(map, (k, v) -> expr)` | Transform values |

## Table-Valued Functions (Generators)

Use in FROM clause with LATERAL:

```sql
SELECT t.*, g.*
FROM my_table t, LATERAL generator_function(...) AS g
```

| Function | Description |
|----------|-------------|
| `EXPLODE(array)` | One row per array element |
| `EXPLODE(map)` | One row per key-value pair |
| `POSEXPLODE(array)` | Explode with position (pos, col) |
| `POSEXPLODE(map)` | Explode map with position |
| `INLINE(array_of_structs)` | Explode struct fields as columns |
| `STACK(n, v1, v2, ...)` | Create n rows from values |

## JSON/Variant Functions

| Function | Description |
|----------|-------------|
| `PARSE_JSON(str)` | Parse JSON string to VARIANT |
| `TRY_PARSE_JSON(str)` | Safe parse (NULL on error) |
| `TO_JSON(variant)` | Convert to JSON string |
| `SCHEMA_OF_VARIANT(v)` | Get schema DDL string |
| `SCHEMA_OF_VARIANT_AGG(v)` | Aggregate schema across rows |
| `FROM_JSON(str, schema)` | Parse to struct with schema |
| `GET_JSON_OBJECT(str, path)` | Extract using JSONPath |
| `JSON_TUPLE(str, k1, k2, ...)` | Extract multiple keys |

### Variant Field Access

```sql
-- Field access
data:field_name

-- Nested field
data:outer:inner

-- Array index (0-based)
data:array[0]

-- Cast to type
data:field::STRING
data:field::INT
data:field::DOUBLE
data:field::BOOLEAN
data:field::TIMESTAMP
```

## Window Functions

### Ranking Functions

| Function | Description |
|----------|-------------|
| `ROW_NUMBER()` | Unique sequential number |
| `RANK()` | Rank with gaps for ties |
| `DENSE_RANK()` | Rank without gaps |
| `NTILE(n)` | Distribute into n buckets |
| `PERCENT_RANK()` | Relative rank (0 to 1) |
| `CUME_DIST()` | Cumulative distribution |

### Analytic Functions

| Function | Description |
|----------|-------------|
| `LAG(col, n, default)` | Value from n rows before |
| `LEAD(col, n, default)` | Value from n rows after |
| `FIRST_VALUE(col)` | First value in window |
| `LAST_VALUE(col)` | Last value in window |
| `NTH_VALUE(col, n)` | Nth value in window |

### Window Frame Specification

```sql
-- Rows-based frame
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING

-- Range-based frame (for ORDER BY on numeric/date)
RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW
```

## String Functions (Spark-specific)

| Function | Description |
|----------|-------------|
| `REGEXP_EXTRACT(str, pattern, group)` | Extract regex group |
| `REGEXP_EXTRACT_ALL(str, pattern)` | Extract all matches |
| `REGEXP_REPLACE(str, pattern, repl)` | Replace regex matches |
| `SPLIT(str, pattern)` | Split to array |
| `SENTENCES(str)` | Split into sentences and words |
| `TRANSLATE(str, from, to)` | Character-level replace |

## Date/Time Functions (Spark-specific)

| Function | Description |
|----------|-------------|
| `DATE_TRUNC(unit, date)` | Truncate to unit |
| `MAKE_DATE(year, month, day)` | Create date |
| `MAKE_TIMESTAMP(y, m, d, h, mi, s)` | Create timestamp |
| `SEQUENCE(start, end, interval)` | Generate date sequence |
| `DATE_ADD(date, days)` | Add days |
| `DATE_SUB(date, days)` | Subtract days |
| `MONTHS_BETWEEN(end, start)` | Months difference |
| `NEXT_DAY(date, dayOfWeek)` | Next occurrence of day |

## Aggregate Functions (Spark-specific)

| Function | Description |
|----------|-------------|
| `COLLECT_LIST(col)` | Aggregate to array (with duplicates) |
| `COLLECT_SET(col)` | Aggregate to array (unique) |
| `APPROX_COUNT_DISTINCT(col)` | Fast approximate distinct count |
| `PERCENTILE(col, p)` | Exact percentile |
| `PERCENTILE_APPROX(col, p, accuracy)` | Approximate percentile |
| `ARRAY_AGG(col)` | Same as COLLECT_LIST |

## Sources

- [Databricks SQL Language Reference](https://docs.databricks.com/aws/en/sql/language-manual/)
- [Higher-order Functions](https://docs.databricks.com/aws/en/semi-structured/higher-order-functions)
- [Window Functions](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-window-functions)
- [QUALIFY Clause](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-qry-select-qualify)
- [Query Variant Data](https://docs.databricks.com/en/semi-structured/variant.html)
- [Delta Lake Time Travel](https://lakefs.io/blog/databricks-time-travel/)
