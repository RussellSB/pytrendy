# Working with Results

Once your segments are wrapped in `PyTrendyResults`, you can filter, rank, and inspect trends:

!!! tip "PyTrendyResults Access Methods"
    - **`results.print_summary()`** - Print formatted text summary
    - **`results.df`** - Access full DataFrame with all segments
    - **`results.filter_segments()`** - Filter/sort segments by criteria
    - **`results.best`** - Get the top-ranked trend
    - **`results.summary`** - Access aggregate statistics

---

## Print Summary

Display a concise overview of all detected segments:

```python
results.print_summary()
```

**Use case:** Quick inspection during development or debugging.

---

## Access Top Trends

Retrieve the most prominent trends:

```python
# Get top 3 segments ranked by slope
top_segments = results.filter_segments(sort_by='change_rank')[:3]

# Get top 3 upwards trends ranked by slope
top_uptrends = results.filter_segments(direction='Up', sort_by='change_rank')[:3]

# Get first 5 segments in order of time
chronological = results.filter_segments(sort_by='time_index')[:5]

# Get the single best trend
best_trend = results.best
```

**Use case:** Highlighting dominant patterns for reporting, dashboards, or strategic insights.

---

## Filter by Direction

Filter segments by trend direction:

```python
# Filter upward trends as a DataFrame
upward_df = results.filter_segments(direction="Up", format="df")

# Filter downward trends as a list of segment objects
downward_segments = results.filter_segments(direction="Down", format="list")
```

**Use case:** Isolating bullish/bearish runs, rising/falling sensor values, or engagement spikes/drops.

---

## Access Full DataFrame

Get complete segment metadata:

```python
# Access full segment summary as a DataFrame
df = results.df

# Preview the first five rows
print(df.head())
```

**Use case:** Exporting to CSV, integrating with BI tools, or feeding into alerting systems.

---

## Advanced Filtering

Apply custom filters using pandas:

```python
# Filter segments with total_change > 50 and days > 10
strong_trends = results.df[
    (results.df["total_change"] > 50) & (results.df["days"] > 10)
]
```

**Use case:** Fine-grained control for domain-specific thresholds or anomaly detection.

---

## Next Steps

- **[Visualization](visualization.md)** - Generate publication-ready plots
- **[Advanced Usage](advanced-usage.md)** - Build custom pipelines
- **[Configuration Reference](configuration-reference.md)** - Full parameter documentation
