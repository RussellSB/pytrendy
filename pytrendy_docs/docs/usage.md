# PyTrendy: Usage

<p></p>

This guide walks through the core usage of PyTrendy, from raw signal input to refined and classified trend segments. The pipeline is modular, allowing developers to use the full workflow or integrate individual components into custom analysis routines.


## 1. Input Requirements

PyTrendy expects a `pandas.DataFrame` with two columns. Their reference will be passed through `detect_trends()`:


<table style="border-collapse: collapse; width: 100%; font-size: 14px;">
  <thead>
    <tr style="border-bottom: 1px solid #ccc;">
      <th style="padding: 8px; text-align: left;">Column</th>
      <th style="padding: 8px; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">date_col</td>
      <td style="padding: 8px;">Currently only dates are supported (daily data).</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">value_col</td>
      <td style="padding: 8px;">Primary time series signal.</td>
    </tr>
  </tbody>
</table>

<p></p>

These columns are currently the only two columns required for trend detection.


## 2. Full Pipeline Execution

<p></p>
The simplest way to use PyTrendy is via the high-level API. This approach runs the complete 5-stage pipeline and returns a structured `PyTrendyResults` object.

```python
from pytrendy import detect_trends

# Run full pipeline on your time series DataFrame
results = detect_trends(
    df,
    date_col="date",                # Column containing datetime values
    value_col="signal",             # Column with signal values
    plot=True,                      # Enable visualization
    method_params={                 # Optional method-specific parameters
        "is_abrupt_padded": True,   # Optional, defaulted to False. Pads abrupt for Quasi-experimental use-cases.
        "abrupt_padding": 5         # Default 28. Only used when is abrupt padded True. Controls padding after abrupt.
    }
)

```

#### What This Executes Internally?

<p></p>

This function wraps the following stages:

<table style="border-collapse: collapse; width: 100%; font-size: 14px;">
  <thead style="background-color: #f5f5f5;">
    <tr>
      <th style="border: 1px solid #ccc; padding: 8px; text-align: left;">Stage</th>
      <th style="border: 1px solid #ccc; padding: 8px; text-align: left;">Function Name</th>
      <th style="border: 1px solid #ccc; padding: 8px; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #ccc; padding: 8px;">Signal Preprocessing</td>
      <td style="border: 1px solid #ccc; padding: 8px;"><code>process_signals</code></td>
      <td style="border: 1px solid #ccc; padding: 8px;">Cleans and normalizes the input signal</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ccc; padding: 8px;">Raw Segmentation</td>
      <td style="border: 1px solid #ccc; padding: 8px;"><code>get_segments</code></td>
      <td style="border: 1px solid #ccc; padding: 8px;">Identifies initial trend segments</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ccc; padding: 8px;">Refinement & Classification</td>
      <td style="border: 1px solid #ccc; padding: 8px;"><code>refine_segments</code></td>
      <td style="border: 1px solid #ccc; padding: 8px;">Merges, filters, and labels segments based on heuristics</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ccc; padding: 8px;">Metric Analysis</td>
      <td style="border: 1px solid #ccc; padding: 8px;"><code>analyse_segments</code></td>
      <td style="border: 1px solid #ccc; padding: 8px;">Computes metrics like slope, duration, and strength</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ccc; padding: 8px;">Optional Visualization</td>
      <td style="border: 1px solid #ccc; padding: 8px;"><code>plot_pytrendy</code></td>
      <td style="border: 1px solid #ccc; padding: 8px;">Generates annotated trend plots</td>
    </tr>
  </tbody>
</table>

<p></p>

The output is a `PyTrendyResults` object, which provides structured access to trends, metrics, and visual summaries.


## 3. Accessing Results

<p></p>

Once your segments are wrapped in PyTrendyResults, you can filter, rank, and inspect trends. These methods are designed to support both exploratory analysis and downstream integration.

<br>

##### **3.1 Print Segment Summary**


Use `print_summary()` to display a concise overview of all detected segments, including direction, slope, duration, and start/end indices.

```python
# Print a readable summary of all segments
results.print_summary()

```
**Use case:** Quick inspection during development or debugging.

<br>


##### **3.2 Access Top Trends**


Retrieve the most prominent trends based on steepness and duration using `results.best`. You can specify how many top segments to return and which metric to rank by.

```python
# Get top 3 segments ranked by slope
top_segments = results.best(n=3, metric="slope")

# Get top 5 segments ranked by duration
longest_segments = results.best(n=5, metric="duration")

```
**Use case:** Highlighting dominant patterns for reporting, dashboards, or strategic insights.

<br>

##### **3.3 Filter by Direction**


Filter segments by trend direction (`"up"` or `"down"`) and choose the output format (`"df"` for DataFrame or `"list"` for raw segment objects).

```python
# Filter upward trends as a DataFrame
upward_df = results.filter_segments(direction="up", format="df")

# Filter downward trends as a list of segment objects
downward_segments = results.filter_segments(direction="down", format="list")

```
**Use case:** Isolating bullish/bearish runs, rising/falling sensor values, or engagement spikes/drops.

<br>

##### **3.4 Access Full Summary DataFrame**


The `summary` attribute provides a dictionary of precomputed summaries. The `"df"` key returns a full DataFrame with all segment metadata.

```python
# Access full segment summary as a DataFrame
summary_df = results.summary["df"]

# Preview the first five rows
print(summary_df.head())

```
**Use case:** Exporting to CSV, integrating with BI tools, or feeding into alerting systems.

<br>

##### **3.5 Advanced Filtering (Optional)**


You can also apply custom filters using pandas directly on `summary_df`:

```python
# Filter segments with total_change > 50 and days > 10
strong_trends = summary_df[
    (summary_df["total_change"] > 50) & (summary_df["days"] > 10)
]

```
**Use case:** Fine-grained control for domain-specific thresholds or anomaly detection.


## 4. Custom Pipeline (Advanced)

<p></p>


For more control, you can run each stage manually:

```python
from pytrendy import (
    process_signals,
    get_segments,
    refine_segments,
    analyse_segments,
    PyTrendyResults
)

# Step 1: Preprocess signal
df = process_signals(df, value_col="signal")

# Step 2: Segment detection
segments_raw = get_segments(df)

# Step 3: Refinement and classification
method_params = {"is_abrupt_padded": True, "abrupt_padding": 5}
segments_refined = refine_segments(df, time_col="date", value_col="signal", segments=segments_raw, method_params=method_params)

# Step 4: Metric analysis and ranking
segments_final = analyse_segments(df, time_col="date", value_col="signal", segments=segments_refined)

# Step 5: Wrap results
results = PyTrendyResults(segments_final)

```

This approach is ideal for experimentation, debugging, or integrating PyTrendy into larger workflows.



## 5. Plotting Trends

<p></p>

To visualize the detected segments:

```python
from pytrendy import plot_pytrendy

plot_pytrendy(df, time_col="date", value_col="signal", segments_enhanced=segments_final)

```

Plots include shaded regions for each segment type:

<table style="border-collapse: collapse; width: 100%; max-width: 600px; font-size: 14px;">
  <thead style="background-color: #f5f5f5;">
    <tr style="border-bottom: 1px solid #ccc;">
      <th style="padding: 8px; text-align: left;">Segment Type</th>
      <th style="padding: 8px; text-align: left;">Color</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">Uptrend</td>
      <td style="padding: 8px; background-color: LightGreen;">Light Green</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">Downtrend</td>
      <td style="padding: 8px; background-color: LightCoral;">Light Coral</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">Flat</td>
      <td style="padding: 8px; background-color: LightBlue;">Light Blue</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">Noise</td>
      <td style="padding: 8px; background-color: LightGray;">Light Gray</td>
    </tr>
  </tbody>
</table>

<p></p>

Top-ranked trends are annotated directly on the chart.



## 6. Use Cases
<p></p>

PyTrendy is adaptable across domains where time series trend detection is critical. It integrates seamlessly with `matplotlib` and `seaborn` for trend visualization. Below is a reusable snippet to plot segmented trends:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_trendy_segments(df, segments, value_col="close", title="Trend Segments"):
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 6))
    plt.plot(df["timestamp"], df[value_col], label="Signal", color="gray", linewidth=1)

    for seg in segments:
        start, end = seg["start_idx"], seg["end_idx"]
        direction = seg["direction"]
        color = "green" if direction == "Up" else "red"
        plt.plot(df["timestamp"].iloc[start:end+1], df[value_col].iloc[start:end+1], color=color, linewidth=2)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(value_col.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.show()
```

Now, let's look at a few domain specific examples.

These datasets are ideal for testing PyTrendy across financial, sensor, and behavioral domains. Each link points to a reliable source with downloadable CSVs.

<table style="border-collapse: collapse; width: 100%; font-size: 14px;">
  <thead style="background-color: #f5f5f5;">
    <tr>
      <th style="border: 1px solid #ccc; padding: 8px; text-align: left;">Dataset Type</th>
      <th style="border: 1px solid #ccc; padding: 8px; text-align: left;">Description</th>
      <th style="border: 1px solid #ccc; padding: 8px; text-align: left;">Format</th>
      <th style="border: 1px solid #ccc; padding: 8px; text-align: left;">Source</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">Stock Prices</td>
      <td style="padding: 8px;">Daily OHLC data for AAPL</td>
      <td style="padding: 8px;">CSV</td>
      <td style="padding: 8px;">
        <a href="https://finance.yahoo.com/quote/AAPL/history" target="_blank">Yahoo Finance</a>
      </td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">IoT Sensor Logs</td>
      <td style="padding: 8px;">Temperature readings over time</td>
      <td style="padding: 8px;">CSV</td>
      <td style="padding: 8px;">
        <a href="https://archive.ics.uci.edu/ml/datasets/Air+Quality" target="_blank">UCI Air Quality Dataset</a>
      </td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">Engagement Metrics</td>
      <td style="padding: 8px;">Daily active users from a web app</td>
      <td style="padding: 8px;">CSV</td>
      <td style="padding: 8px;">
        <a href="https://www.kaggle.com/code/anshikakashyap12/website-traffic-eda" target="_blank">Kaggle Web Traffic</a>
      </td>
    </tr>
  </tbody>
</table>


<br>

#####   a) **Financial Time Series: Detecting Bullish/Bearish Runs**

Detect trends in stock prices to isolate bullish or bearish runs. Useful for quant strategies, backtesting, or market phase detection.

<details><summary><strong>Example: Stock Price Trend Detection</strong></summary>

```python
import pandas as pd
from pytrendy import (
    process_signals,
    get_segments,
    refine_segments,
    analyse_segments,
    PyTrendyResults
)

# Load historical stock price data
df = pd.read_csv("stock_prices.csv")

# Preprocess the signal (e.g., smoothing, normalization)
df = process_signals(df, value_col="close")

# Identify raw trend segments
segments = get_segments(df)

# Refine segments using financial heuristics
segments = refine_segments(
    df,
    value_col="close",
    segments=segments,
    method_params={
        "min_length": 5,
        "slope_threshold": 0.02
    }
)

# Analyze segments for slope, volatility, etc.
segments = analyse_segments(df, value_col="close", segments=segments)

# Wrap results for filtering and visualization
results = PyTrendyResults(segments)

# Filter bullish and bearish runs
bullish_runs = results.filter_segments(direction="up")
bearish_runs = results.filter_segments(direction="down")


```
</details>

<br>

#####   b) **Sensor Data (IoT): Detecting Temperature Shifts**

Track environmental shifts like temperature or humidity changes. Ideal for anomaly detection, predictive maintenance, or alerting systems.

<details> <summary><strong>Example: Temperature Shift Detection</strong></summary>

```python
import pandas as pd
from pytrendy import (
    process_signals,
    get_segments,
    refine_segments,
    analyse_segments,
    PyTrendyResults
)

# Load IoT sensor logs
df = pd.read_csv("AirQualityUCI.csv")

# Preprocess the temperature signal
df = process_signals(df, value_col="temperature")

# Segment the signal into meaningful shifts
segments = get_segments(df)

# Refine segments based on expected environmental behavior
segments = refine_segments(
    df,
    value_col="temperature",
    segments=segments,
    method_params={
        "padding": 3,
        "noise_floor": 0.5
    }
)

# Analyze segments for magnitude and duration
segments = analyse_segments(df, value_col="temperature", segments=segments)

# Package results for downstream systems
results = PyTrendyResults(segments)

# Export structured summaries for alerting systems
summary_df = results.segments_df


```
</details>

<br>

#####   c) **Behavioral Analytics: Identifying Peak Engagement**

Analyze user engagement metrics to identify peak activity periods. Great for product usage insights, campaign timing, or retention analysis.

<details> <summary><strong>Example: Engagement Peak Detection</strong></summary>


```python
import pandas as pd
from pytrendy import (
    process_signals,
    get_segments,
    refine_segments,
    analyse_segments,
    PyTrendyResults
)

# Load user engagement metrics
df = pd.read_csv("website_wata.csv")

# Preprocess the engagement signal
df = process_signals(df, value_col="active_users")

# Detect engagement phases
segments = get_segments(df)

# Refine segments to capture behavioral shifts
segments = refine_segments(
    df,
    value_col="active_users",
    segments=segments,
    method_params={
        "min_length": 7,
        "derivative_window": 3
    }
)

# Analyze segments for engagement quality
segments = analyse_segments(df, value_col="active_users", segments=segments)

# Wrap results for strategic insights
results = PyTrendyResults(segments)

# Identify peak engagement periods
top_segments = results.best(n=3, metric="slope")


```
</details>



## 7. Configuration Parameters



<table style="border-collapse: collapse; width: 100%; font-size: 14px;">
  <thead>
    <tr style="border-bottom: 1px solid #ccc;">
      <th style="padding: 8px; text-align: left;">Parameter</th>
      <th style="padding: 8px; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">is_abrupt_padded</td>
      <td style="padding: 8px;">Whether to extend abrupt trends with padding</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">abrupt_padding</td>
      <td style="padding: 8px;">Number of days to pad abrupt trends</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">plot</td>
      <td style="padding: 8px;">Whether to generate visual output</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">value_col</td>
      <td style="padding: 8px;">Column name of the signal to analyze</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">date_col</td>
      <td style="padding: 8px;">Column name of the datetime field</td>
    </tr>
  </tbody>
</table>


These can be passed via `method_params` in `detect_trends()`.


## 8. Version Compatibility and Changelog



<table style="border-collapse: collapse; width: 100%; font-size: 14px;">
  <thead>
    <tr style="border-bottom: 1px solid #ccc;">
      <th style="padding: 8px; text-align: left;">PyTrendy Version</th>
      <th style="padding: 8px; text-align: left;">Python Support</th>
      <th style="padding: 8px; text-align: left;">Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">&ge;1.0.0</td>
      <td style="padding: 8px;">&ge;3.8</td>
      <td style="padding: 8px;">Initial stable release</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">&ge;1.1.0</td>
      <td style="padding: 8px;">&ge;3.8</td>
      <td style="padding: 8px;">Added DTW classification and abrupt trend handling</td>
    </tr>
  </tbody>
</table>

<br>


See [GitHub](https://github.com/RussellSB/pytrendy/) Releases for detailed version history.
