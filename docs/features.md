# Key Features

PyTrendy is a Python library for automated trend detection and analysis in time series data.
It transforms raw, noisy signals into meaningful segments (Up, Down, Flat, Noise), enriched with statistics, classifications, and visualizations.

Below is a breakdown of its features.

## 1. Signal Preprocessing

PyTrendy performs signal conditioning prior to trend analysis through the following steps:

- **Savitzky–Golay smoothing**: Applies a polynomial filter to suppress high-frequency noise while retaining signal morphology.

- **Flat region identification**: Detects low-variance segments using a rolling standard deviation threshold.

- **Noise quantification**: Calculates the Signal-to-Noise Ratio (SNR) to isolate unreliable data points.

- **Gradient analysis**: Computes the first derivative of the smoothed signal to capture directional changes.

**Result**: The processed signal is annotated with trend labels: `Up`, `Down`, `Flat`, or `Noise`.


## 2. Trend Segmentation

Following preprocessing, the signal is segmented into contiguous regions based on directional characteristics:

- **Up**: Monotonic or sustained positive slope
- **Down**: Monotonic or sustained negative slope
- **Flat**: Minimal variance within threshold bounds
- **Noise**: High-frequency volatility exceeding SNR limits

**Segment metadata includes:**

- Temporal bounds (start and end timestamps)
- Duration in days
- Trend classification (`Up`, `Down`, `Flat`, `Noise`)
- Positional index within the time series

**Result**: The time series is decomposed into a structured sequence of labeled segments for downstream analysis.


## 3. Segment Refinement

Post-segmentation, PyTrendy applies refinement procedures to eliminate artifacts and enhance segment fidelity:

- **Boundary optimization**: Dynamically adjusts segment edges to better align with signal transitions.

- **Trend classification**: Differentiates gradual vs. abrupt transitions using Dynamic Time Warping (DTW) similarity metrics.

- **Abrupt trend decomposition**: Detects latent change points within sharp directional shifts.

- **Directional merging**: Consolidates adjacent short segments exhibiting consistent trend direction.

- **Artifact filtering**: Removes segments below duration or significance thresholds.

**Result**: Refined segments exhibit improved continuity, reduced noise, and enhanced interpretability for downstream tasks.


## 4. Segment Analysis

Each refined segment is enriched with quantitative metrics to evaluate its behavior and significance:

- **Absolute Change**: Net difference in signal value between segment boundaries.

- **Relative Change**: Percentage change relative to the segment's starting value.

- **Duration**: Total length of the segment, measured in calendar days.

- **Cumulative Change**: Aggregated directional movement across the segment timeline.

- **Signal-to-Noise Ratio (SNR)**: Quantifies trend clarity by comparing signal strength to background variability.

- **Gradient Ranking**: Orders segments by magnitude of directional slope, from steepest to shallowest.

**Result**: These metrics enable comparative analysis, prioritization, and deeper inspection of segment dynamics.


## 5. Trend Ranking & Identification

PyTrendy applies quantitative evaluation to rank trend segments by statistical significance:

- **Magnitude-based sorting (`change_rank`)**: Segments are ordered by the absolute value of their relative (%) change, from steepest to shallowest.

- **Order of time (`time_index`)**: A simple ranking of consequitive order in time. Later segments are assigned higher time index.

- **Directional classification (`direction`)**: Labels each ranked segment as either `Up` or `Down` based on its slope sign.

**Result**: Enables rapid identification of high-impact trends, facilitating prioritization and focused analysis within the time series.


## 6. Visualization

PyTrendy includes a built-in visualization engine for generating interactive, data-rich plots:

- **Segment highlighting**: Time series plots display shaded regions to delineate segment types using a default color map:
  
    - `Uptrend`: Green 🟩

    - `Downtrend`: Red 🟥

    - `Flat`: Blue 🟦

    - `Noise`: Gray ⬜

- **Trend rank annotation**: Top-ranked segments are automatically labeled with ordinal markers for rapid identification.

- **Static rendering**: Visual outputs are fully static in environments such as Jupyter Notebooks and Google Colab.

**Result**: Generates publication-ready visualizations that facilitate intuitive interpretation and seamless integration into analytical workflows.


## 7. High-Level API

PyTrendy exposes a simplified interface for end-to-end trend detection:

```python
from pytrendy import detect_trends

results = detect_trends(df, date_col="date", value_col="gradual", plot=True)
```
This single function executes the full 5-stage pipeline and returns a `PyTrendyResults` object with the following attributes:

- **best**: Most statistically significant trend segment

- **summary**: Aggregate counts of Up, Down, Flat, and Noise segments

- **print_summary()**: Utility for printing summary as a body of text

- **segments**: List of segment dictionaries with metadata

- **df**: Pandas DataFrame for tabular analysis

- **filter_segments()**: Utility for querying segments based on custom criteria

**Result**: Streamlined execution for rapid deployment and minimal setup.

