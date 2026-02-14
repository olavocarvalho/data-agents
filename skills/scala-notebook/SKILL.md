---
name: scala-notebook
description: use when writing databricks scala notebooks.
---

# Scala - Databricks Scala Notebook Patterns

Best practices for writing robust Spark code on Databricks.

## When to Use This Skill

- Writing Spark ETL pipelines in Scala
- Joining DataFrames with remote table lineage
- Debugging self-join errors on Databricks
- Optimizing notebook performance with caching strategies
- Avoiding ambiguous column references after joins
- Sharing config between Scala and Python cells in mixed notebooks

## Avoid Self-Join Errors on Remote Tables

Databricks blocks self-joins on remote tables. When building pipelines that read from remote tables and then join the results:

**Problem:**
```scala
val filtered = spark.table("remote_catalog.schema.table").filter(...).select(...)
filtered.write.saveAsTable("tmp_filtered")

val result = baseTable
  .join(filtered, ...)  // ERROR: self-join on remote table
```

**Solution:** Read from the saved table to break lineage:
```scala
filtered.write.saveAsTable("tmp_filtered")

// Read from saved table to break lineage
val filteredFromTable = spark.table("tmp_filtered")

val result = baseTable
  .join(filteredFromTable, ...)  // OK: no remote table lineage
```

## Use Temp Tables Instead of Cache

Prefer saving intermediate results to temp tables rather than using `.cache()`. Temp tables:
- Persist between notebook runs (no recomputation)
- Can be queried from outside the notebook for debugging
- Break lineage to remote tables automatically

```scala
val tempTableName = s"${outputPrefix}__tmp_mapping"
val tempTableExists = spark.catalog.tableExists(tempTableName)

if (!tempTableExists) {
  spark.table("remote_catalog.schema.table")
    .select($"id", $"category")
    .write.mode("overwrite").saveAsTable(tempTableName)
}

// Read from temp table - breaks lineage, persists between runs
val mapping = spark.table(tempTableName)

// Safe to use in multiple joins
val subset1 = mapping.filter(...).distinct()
val subset2 = mapping.filter(...).distinct()
```

## Avoid Ambiguous Column References After Joins

When joining a grouped DataFrame back to its source table, avoid creating duplicate column names:

**Problem:**
```scala
val grouped = sourceTable
  .groupBy($"category")
  .agg(
    min($"score").as("min_score"),
    first($"name").as("name")  // Creates duplicate column
  )
  .join(sourceTable, Seq("category"))  // sourceTable also has "name"
  .select($"name")  // ERROR: ambiguous reference
```

**Solution:** Only aggregate what you need, let the join bring in other columns:
```scala
val grouped = sourceTable
  .groupBy($"category")
  .agg(min($"score").as("min_score"))
  .join(sourceTable, Seq("category"))
  .filter($"score" === $"min_score")
  .select($"category", $"name")  // OK: only one "name" column
```

## Use `lit()` for Literal Arithmetic with Columns

When doing arithmetic with Spark Columns, literals on the **left side** of operators cause compilation errors because Scala tries to use the literal's method (e.g., Int's `-` or `/`) which doesn't accept a Column.

**Problem:**
```scala
// Fails: Int's `-` method doesn't accept Column
.withColumn("complement", 1 - $"rate")

// Fails: Int's `/` method doesn't accept Column
.withColumn("inverse", 1 / $"value")
```

**Solution:** Use `lit()` to convert literals to Columns:
```scala
.withColumn("complement", lit(1) - $"rate")
.withColumn("inverse", lit(1) / $"value")

// Example: compound expression
.withColumn("result", sqrt($"p" * (lit(1) - $"p") * (lit(1)/$"n1" + lit(1)/$"n2")))
```

**Note:** When the Column is on the **left** side, no `lit()` is needed:
```scala
.withColumn("adjusted", $"value" - 1)
.withColumn("scaled", $"value" / 100)
```

## Pass Config from Scala to Python Cells

When notebooks mix Scala and Python cells, avoid hardcoding config values in Python. Use `spark.conf` to pass values from Scala:

**Problem:**
```python
# Python cell hardcodes values that should come from Scala config
OUTPUT_PREFIX = "usr.myuser.my_project"  # Duplicates Scala config!
```

**Solution:** Set config in Scala, read in Python:
```scala
// In Scala cell (before Python cells)
spark.conf.set("pipeline.output_prefix", outputPrefix)
spark.conf.set("pipeline.start_date", startDate)
```

```python
# In Python cell
OUTPUT_PREFIX = spark.conf.get("pipeline.output_prefix")
START_DATE = spark.conf.get("pipeline.start_date")
```

This ensures Python cells inherit config from the Scala setup instead of duplicating values.
