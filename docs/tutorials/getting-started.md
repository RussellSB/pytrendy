# Tutorial 1: Your First Trend Analysis

<span class="badge-beginner">BEGINNER</span> <span class="badge-time">5 minutes</span>

This tutorial walks you through loading data, detecting trends, and exploring results interactively.

---

## Step 1: Load and Inspect Data

Load and explore the synthetic dataset:

```python
import pytrendy as pt

# Load built-in synthetic dataset
df = pt.load_data('series_synthetic')
print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
```

<div class='transparent'>
```
=== Dataset Preview ===
         date     abrupt    gradual  gradual-noisy-20
0  2025-01-01  19.578066  12.500000         27.514106
1  2025-01-02  19.358378  13.421717         -6.620099
2  2025-01-03  19.228408  13.474026         22.122134
3  2025-01-04  19.727130  13.474026         13.863735
4  2025-01-05  20.773716  14.505772          8.884535
5  2025-01-06  20.752303  14.709596         33.067795
6  2025-01-07  19.713066  14.783550          9.225507
7  2025-01-08  20.736886  16.354618         18.465998
8  2025-01-09  19.167441  17.370130         13.817332
9  2025-01-10  19.039469  16.493506         32.737775

Shape: (181, 4)
Columns: ['date', 'abrupt', 'gradual', 'gradual-noisy-20']
```
</div>

!!! example "Dataset Overview"
    The dataset contains **181 days** of synthetic time series with three signal types:
    
    - **`abrupt`**: Sharp transitions (e.g., product launches, market crashes)
    - **`gradual`**: Smooth trends (e.g., seasonal growth, adoption curves)
    - **`gradual-noisy-20`**: Gradual trend with 20% noise added

---

## Step 2: Run Trend Detection

Run trend detection on your data:

```python
# Detect trends on the 'gradual' signal
results = pt.detect_trends(
    df, 
    date_col='date', 
    value_col='gradual', 
    plot=True  # Set to False to skip visualization
)

# Print summary
results.print_summary()
```


<div class='transparent'>
```
Detected: 
- 3 Uptrends. 
- 3 Downtrends.
- 3 Flats.
- 0 Noise.

The best detected trend is Down between dates 2025-05-09 - 2025-06-17

Full Results:
-------------------------------------------------------------------------------
            direction       start         end  days  total_change  change_rank
time_index                                                                   
1                 Up  2025-01-02  2025-01-24    22     14.013348          5.0
2               Down  2025-01-25  2025-02-05    11    -13.564214          6.0
3               Flat  2025-02-06  2025-02-09     3           NaN          NaN
4                 Up  2025-02-10  2025-03-14    32     24.632035          3.0
5               Flat  2025-03-15  2025-03-17     2           NaN          NaN
6               Down  2025-03-18  2025-04-01    14    -22.721861          4.0
7                 Up  2025-04-02  2025-05-08    36     72.611833          2.0
8               Down  2025-05-09  2025-06-17    39    -73.253968          1.0
9               Flat  2025-06-18  2025-06-29    11           NaN          NaN
-------------------------------------------------------------------------------
```
</div>

!!! success "Detection Complete!"
    PyTrendy identified **9 segments** including 3 uptrends, 3 downtrends, and 3 flat regions. The strongest trend (Rank 1) is a **39-day downtrend** with a change of **-73.25 units**.

---

## Step 3: Explore the Results

Access detailed information about the best trend:

```python
# Access the best trend
best_trend = results.best

print("=== Best Trend Details ===")
print(f"Direction: {best_trend['direction']}")
print(f"Duration: {best_trend['days']} days")
print(f"Change: {best_trend['total_change']:.2f}")
print(f"Period: {best_trend['start']} to {best_trend['end']}")
print(f"Classification: {best_trend['trend_class']}")
```


<div class='transparent'>
```
=== Best Trend Details ===
Direction: Down
Duration: 39 days
Change: -73.25
Period: 2025-05-09 to 2025-06-17
Classification: gradual
```
</div>

---

## Step 4: Filter and Sort Trends

Filter segments by various criteria:

```python
# Get top 3 strongest trends
print("=== Top 3 Strongest Trends ===")
top3 = results.filter_segments(sort_by='change_rank', format='df')[:3]
print(top3[['direction', 'start', 'end', 'days', 'total_change', 'change_rank']])

# Get only uptrends, sorted by time
print("\n=== Only Uptrends (by time) ===")
uptrends = results.filter_segments(direction='Up', sort_by='time_index', format='df')
print(uptrends[['direction', 'start', 'end', 'days', 'total_change']])

# Get downtrends as a list of dictionaries
print("\n=== Only Downtrends (as dict) ===")
downtrends = results.filter_segments(direction='Down', format='dict')
for dt in downtrends:
    print(f"  {dt['start']} to {dt['end']}: {dt['total_change']:.2f} (Rank {dt['change_rank']})")
```

<div class='transparent'>
```
=== Top 3 Strongest Trends ===
           direction       start         end  days  total_change  change_rank
time_index                                                                   
8               Down  2025-05-09  2025-06-17    39    -73.253968          1.0
7                 Up  2025-04-02  2025-05-08    36     72.611833          2.0
4                 Up  2025-02-10  2025-03-14    32     24.632035          3.0

=== Only Uptrends (by time) ===
           direction       start         end  days  total_change
time_index                                                      
1                 Up  2025-01-02  2025-01-24    22     14.013348
4                 Up  2025-02-10  2025-03-14    32     24.632035
7                 Up  2025-04-02  2025-05-08    36     72.611833

=== Only Downtrends (as dict) ===
  2025-01-25 to 2025-02-05: -13.56 (Rank 6)
  2025-03-18 to 2025-04-01: -22.72 (Rank 4)
  2025-05-09 to 2025-06-17: -73.25 (Rank 1)
```
</div>

!!! tip "Filtering Tips"
    - Use `sort_by='change_rank'` to prioritize by magnitude
    - Use `sort_by='time_index'` to get chronological order
    - Use `format='df'` for DataFrame or `format='dict'` for list of dictionaries

---

## Step 5: Export Results

Export and analyze results programmatically:

```python
# Export full results to CSV
results.df.to_csv('trend_results.csv')
print("Exported to trend_results.csv")

# Work with DataFrame directly
df_results = results.df

# Filter by custom criteria
strong_trends = df_results[
    (df_results['days'] > 20) & 
    (abs(df_results['total_change']) > 30)
]
print(f"\nFound {len(strong_trends)} strong trends (>20 days, >30 units change)")
print(strong_trends[['direction', 'start', 'end', 'days', 'total_change']])
```

<div class='transparent'>
```
Exported to trend_results.csv

Found 2 strong trends (>20 days, >30 units change)
           direction       start         end  days  total_change
time_index                                                      
7                 Up  2025-04-02  2025-05-08    36     72.611833
8               Down  2025-05-09  2025-06-17    39    -73.253968
```
</div>

---

## Tutorial Complete!

!!! success "You've learned how to:"
    - Load and inspect datasets  
    - Run trend detection with `detect_trends()`  
    - Access the best trend via `results.best`  
    - Filter and sort segments  
    - Export results to CSV

!!! note "See also"
    For advanced options and function signatures, see the [API Reference](../reference/pytrendy/index.md).

!!! tip "Try It Yourself"
    **Challenge**: Run trend detection on the `abrupt` signal column instead of `gradual`. Compare the number of segments detected and their characteristics. What differences do you notice in the trend patterns?

---

## Next Steps

- **[Custom Pipeline Tutorial](custom-pipeline.md)** - Build detection pipelines from scratch
- **[Abrupt vs Gradual Tutorial](abrupt-vs-gradual.md)** - Understand trend classification
- **[Real-World Examples](real-world-examples.md)** - Bitcoin, GitHub, and climate data
- **[User Guide](../user-guide/index.md)** - Complete features and usage reference
- **[API Reference](../reference/pytrendy/index.md)** - Full API documentation
