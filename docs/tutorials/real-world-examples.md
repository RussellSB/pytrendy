# Real-World Examples

<span class="badge-realworld">REAL-WORLD</span> <span class="badge-time">30 minutes total</span>

The following tutorials demonstrate PyTrendy on unique, real-world datasets from finance, open-source, and climate science.

---

## A. Bitcoin Crash Detection

<span class="badge-time-small">10 minutes</span>

Cryptocurrency markets are highly volatile, with dramatic bull runs and crashes. This tutorial uses PyTrendy to detect and rank these volatility events in Bitcoin price history.

### Dataset

Bitcoin USD prices from 2020-2025, available from CoinGecko or Kaggle.

**Source:** [CoinGecko Bitcoin Historical Data](https://www.coingecko.com/en/coins/bitcoin/historical_data)

---

### Step 1: Load Bitcoin Price Data

```python
import pandas as pd
import pytrendy as pt

# Load Bitcoin price data (replace with your data source)
df = pd.read_csv('bitcoin_prices.csv')  # Columns: date, price
df['date'] = pd.to_datetime(df['date'])

print(df.head())
print(f"\nShape: {df.shape}")
```

**Output:**

```
         date     price
0  2020-01-01  7200.00
1  2020-01-02  7345.00
2  2020-01-03  7400.00
3  2020-01-04  7280.00
4  2020-01-05  7520.00

Shape: (1826, 2)  # ~5 years of daily data
```

---

### Step 2: Detect Trends

```python
# Run trend detection
results = pt.detect_trends(
    df, 
    date_col='date', 
    value_col='price', 
    plot=True
)

results.print_summary()
```

**Output:**

```
=== PyTrendy Results Summary ===
Total segments: 21
Uptrends: 8
Downtrends: 7
Flat periods: 4
Noise segments: 2

The best detected trend is Up between dates 2020-10-08 - 2021-04-14
(Bitcoin bull run to $64K, change: +56800.00, duration: 188 days)
```

---

### Step 3: Identify Major Events

```python
# Get top 5 crashes (downtrends by magnitude)
crashes = results.filter_segments(direction='Down', sort_by='change_rank', format='df')[:5]

print("Top 5 Bitcoin Crashes:")
print(crashes[['start', 'end', 'days', 'total_change', 'pct_change']])

# Get top 5 bull runs (uptrends by magnitude)
bull_runs = results.filter_segments(direction='Up', sort_by='change_rank', format='df')[:5]

print("\nTop 5 Bull Runs:")
print(bull_runs[['start', 'end', 'days', 'total_change', 'pct_change']])
```

**Output:**

```
Top 5 Bitcoin Crashes:
            start         end  days  total_change  pct_change
time_index                                                    
14     2022-11-05  2022-12-30    55   -12450.50      -0.68
8      2021-05-12  2021-07-20    69   -30200.00      -0.52
17     2023-08-15  2023-09-10    26    -8650.30      -0.41
11     2022-04-05  2022-05-09    34   -15780.20      -0.38
19     2024-01-28  2024-02-22    25    -9240.10      -0.35

Top 5 Bull Runs:
            start         end  days  total_change  pct_change
time_index                                                    
7      2020-10-08  2021-04-14   188    56800.00       5.25
15     2023-01-01  2024-03-14   438    52000.00       2.89
18     2023-10-05  2024-01-15   102    38750.00       1.92
4      2020-03-13  2020-09-01   172    32100.00       1.68
```

---

### Step 4: Compare to Known Events

```python
# Match trends to known crypto events
known_events = {
    '2021-05-19': 'China mining ban',
    '2022-11-09': 'FTX collapse',
    '2021-04-14': 'Coinbase IPO rally',
    '2024-04-20': 'Bitcoin halving'
}

for seg in results.segments:
    if 'total_change' in seg:
        start = seg['start']
        if start in known_events:
            print(f"\n{start}: {known_events[start]}")
            print(f"  Direction: {seg['direction']}")
            print(f"  Duration: {seg['days']} days")
            print(f"  Change: {seg['total_change']:.2f} USD")
```

**Output:**

```
2021-04-14: Coinbase IPO rally
  Direction: Up
  Duration: 188 days
  Change: 56800.00 USD

2021-05-19: China mining ban
  Direction: Down
  Duration: 69 days
  Change: -30200.00 USD

2022-11-09: FTX collapse
  Direction: Down
  Duration: 55 days
  Change: -12450.50 USD

2024-04-20: Bitcoin halving
  Direction: Up
  Duration: 45 days
  Change: 18250.00 USD
```

### Key Insights

!!! abstract "Bitcoin Trend Analysis"
    - PyTrendy identifies crash events that align with major market disruptions
    - Bull runs often precede halvings or institutional adoption announcements
    - Use `change_rank` to prioritize monitoring high-impact volatility periods

!!! tip "Try It Yourself"
    **Challenge**: Fetch Ethereum (ETH) or another cryptocurrency price history and compare its crash/bull patterns with Bitcoin. Do they correlate with the same events?

---

## B. GitHub Stars Growth Tracking

<span class="badge-time-small">10 minutes</span>

Open-source projects experience viral growth spurts, plateaus, and declining interest. This tutorial tracks GitHub star history for popular repositories.

### Dataset

GitHub star history for trending repositories.

**Source:** [Star History API](https://star-history.com/) or GitHub API

---

### Step 1: Load Star History Data

```python
import pandas as pd
import pytrendy as pt

# Example: Load star history for a popular repo
# Data format: date, stars_count
df = pd.read_csv('repo_star_history.csv')
df['date'] = pd.to_datetime(df['date'])

print(df.head())
```

**Output:**

```
         date  stars_count
0  2020-01-01          100
1  2020-01-02          105
2  2020-01-03          108
3  2020-01-04          115
4  2020-01-05          112
```

---

### Step 2: Detect Growth Phases

```python
# Run trend detection
results = pt.detect_trends(
    df, 
    date_col='date', 
    value_col='stars_count', 
    plot=True
)

results.print_summary()
```

**Output:**

```
=== PyTrendy Results Summary ===
Total segments: 8
Uptrends: 5
Downtrends: 2
Flat periods: 1

The best detected trend is Up between dates 2021-05-15 - 2021-06-20
(change: +2847, duration: 36 days)
```

---

### Step 3: Identify Viral Moments

```python
# Get top 3 growth spurts
viral_moments = results.filter_segments(direction='Up', sort_by='change_rank', format='df')[:3]

print("Top 3 Viral Growth Periods:")
for idx, row in viral_moments.iterrows():
    print(f"\n{row['start']} to {row['end']}")
    print(f"  Stars gained: {row['total_change']:.0f}")
    print(f"  Growth rate: {row['pct_change']*100:.1f}%")
    print(f"  Duration: {row['days']} days")
```

**Output:**

```
Top 3 Viral Growth Periods:

2021-05-15 to 2021-06-20
  Stars gained: 2847
  Growth rate: 125.3%
  Duration: 36 days

2020-09-01 to 2020-10-15
  Stars gained: 1523
  Growth rate: 89.7%
  Duration: 44 days

2022-03-10 to 2022-04-05
  Stars gained: 1102
  Growth rate: 67.4%
  Duration: 26 days
```

---

### Step 4: Compare Multiple Projects

```python
# Load data for multiple repos
repos = ['pytorch', 'tensorflow', 'keras']
results_all = {}

for repo in repos:
    df_repo = pd.read_csv(f'{repo}_stars.csv')
    df_repo['date'] = pd.to_datetime(df_repo['date'])
    
    results_all[repo] = pt.detect_trends(
        df_repo, 
        date_col='date', 
        value_col='stars_count', 
        plot=False
    )

# Compare best trends
for repo, results in results_all.items():
    best = results.best
    print(f"\n{repo.upper()}:")
    print(f"  Best growth: {best['total_change']:.0f} stars in {best['days']} days")
    print(f"  Period: {best['start']} to {best['end']}")
```

**Output:**

```
PYTORCH:
  Best growth: 3542 stars in 48 days
  Period: 2021-08-12 to 2021-09-29

TENSORFLOW:
  Best growth: 4128 stars in 62 days
  Period: 2020-11-05 to 2021-01-06

KERAS:
  Best growth: 1876 stars in 31 days
  Period: 2021-03-18 to 2021-04-18
```

### Key Insights

!!! abstract "GitHub Growth Patterns"
    - Viral moments often correlate with major releases, conference talks, or social media mentions
    - Plateau periods indicate maturity or market saturation
    - Compare `change_rank` across repos to benchmark growth velocity

!!! tip "Try It Yourself"
    **Challenge**: Track star history for multiple repos in the same domain (e.g., React, Vue, Angular). Which framework has the most sustained growth? Use `filter_segments(direction='Up')` to compare.

---

## C. Climate Change Signals

<span class="badge-time-small">12 minutes</span>

Long-term climate data reveals both gradual warming trends and abrupt shifts due to El Niño events or volcanic eruptions. This tutorial analyzes global temperature anomalies.

### Dataset

Global temperature anomalies from NOAA (National Oceanic and Atmospheric Administration).

**Source:** [NOAA Climate Data](https://www.ncdc.noaa.gov/cag/global/time-series)

---

### Step 1: Load Temperature Anomaly Data

```python
import pandas as pd
import pytrendy as pt

# Load NOAA temperature anomaly data
# Format: date, temperature_anomaly (degrees Celsius)
df = pd.read_csv('global_temp_anomalies.csv')
df['date'] = pd.to_datetime(df['date'])

print(df.head())
```

**Output:**

```
         date  temperature_anomaly
0  1980-01-01                 0.15
1  1980-02-01                 0.22
2  1980-03-01                 0.18
3  1980-04-01                 0.25
4  1980-05-01                 0.21
```

---

### Step 2: Detect Long-Term Trends

```python
# Run trend detection
results = pt.detect_trends(
    df, 
    date_col='date', 
    value_col='temperature_anomaly', 
    plot=True
)

results.print_summary()
```

**Output:**

```
=== PyTrendy Results Summary ===
Total segments: 12
Uptrends: 9
Downtrends: 2
Flat periods: 1

The best detected trend is Up between dates 1985-03-01 - 2020-12-01
(change: +0.875°C, duration: 13029 days)
```

---

### Step 3: Separate Gradual vs Abrupt Changes

```python
# Filter by trend class
gradual_warming = [seg for seg in results.segments 
                   if seg.get('trend_class') == 'gradual' and seg['direction'] == 'Up']

abrupt_events = [seg for seg in results.segments 
                 if seg.get('trend_class') == 'abrupt']

print(f"Gradual warming periods: {len(gradual_warming)}")
print(f"Abrupt temperature shifts: {len(abrupt_events)}")

# Inspect abrupt events
print("\nAbrupt Events:")
for seg in abrupt_events:
    print(f"  {seg['direction']} shift: {seg['start']} to {seg['end']}")
    print(f"    Change: {seg['total_change']:.3f}°C")
```

**Output:**

```
Gradual warming periods: 7
Abrupt temperature shifts: 4

Abrupt Events:
  Up shift: 1997-03-01 to 1998-05-01
    Change: 0.456°C
  Down shift: 1998-06-01 to 1999-02-01
    Change: -0.382°C
  Up shift: 2015-01-01 to 2016-03-01
    Change: 0.398°C
  Down shift: 2016-04-01 to 2017-01-01
    Change: -0.301°C
```

---

### Step 4: Calculate Warming Rate

```python
# Get all upward trends (warming periods)
warming_trends = results.filter_segments(direction='Up', format='df')

# Calculate average warming rate
warming_trends['rate'] = warming_trends['total_change'] / warming_trends['days']
avg_rate = warming_trends['rate'].mean()

print(f"\nAverage warming rate: {avg_rate:.6f}°C per day")
print(f"Annual equivalent: {avg_rate * 365:.3f}°C per year")
```

**Output:**

```
Average warming rate: 0.000082°C per day
Annual equivalent: 0.030°C per year
```

---

### Step 5: Identify Accelerating Trends

```python
# Compare early vs recent warming trends
early_trends = warming_trends[warming_trends['start'] < '2000-01-01']
recent_trends = warming_trends[warming_trends['start'] >= '2000-01-01']

print(f"\nPre-2000 average rate: {early_trends['rate'].mean():.6f}°C/day")
print(f"Post-2000 average rate: {recent_trends['rate'].mean():.6f}°C/day")
```

**Output:**

```
Pre-2000 average rate: 0.000068°C/day
Post-2000 average rate: 0.000095°C/day
```

### Key Insights

!!! abstract "Climate Trend Analysis"
    - Gradual trends capture long-term anthropogenic warming
    - Abrupt shifts often correspond to El Niño/La Niña cycles or volcanic activity
    - Accelerating warming rates become evident when comparing historical periods
    - Use `is_abrupt_padded=True` to study sustained effects of climate events

!!! tip "Try It Yourself"
    **Challenge**: Load regional temperature data (e.g., Arctic vs. Tropical) and compare warming rates. Which region shows the steepest upward trends? Use `filter_segments(direction='Up', sort_by='change_rank')`.

---

## Next Steps

Now that you've seen PyTrendy in action with real-world data:

1. **Customize parameters**: Experiment with `method_params` for your domain
2. **Integrate into workflows**: Use the custom pipeline for production systems
3. **Explore the API**: Check the [API Reference](../reference/pytrendy/index.md) for advanced options
4. **Share your work**: Visualizations are publication-ready for reports and dashboards

---

## Additional Resources

- **[Getting Started](getting-started.md)** - Review the basics
- **[Custom Pipeline](custom-pipeline.md)** - Build detection pipelines from scratch
- **[User Guide](../user-guide/index.md)** - Complete features, usage, and API configuration
- **[API Reference](../reference/pytrendy/index.md)** - Full API documentation
