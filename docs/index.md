<p align="center"><img src="https://raw.githubusercontent.com/RussellSB/pytrendy/3bea91f34bfa8d5452332e5f59f3e2bdf1e3806c/plots/logo.svg" width="250"></p>

<h1 align="center">PyTrendy</h1>

<p align="center"><a href="https://pypi.org/project/pytrendy/"><img src="https://img.shields.io/pypi/v/pytrendy.svg"></a> <a href="https://pypi.org/project/pytrendy/"><img src="https://img.shields.io/badge/python-%3E%3D%203.10-blue.svg"></a> <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a></p>

<p align="center"><a href="https://github.com/RussellSB/pytrendy/actions/workflows/test.yaml"><img src="https://github.com/RussellSB/pytrendy/actions/workflows/test.yaml/badge.svg"></a> <a href="https://github.com/RussellSB/pytrendy/actions/workflows/release.yaml"><img src="https://github.com/RussellSB/pytrendy/actions/workflows/release.yaml/badge.svg"></a></p>

<p align="center"><a href="https://codecov.io/gh/RussellSB/pytrendy"><img src="https://codecov.io/gh/RussellSB/pytrendy/branch/main/graph/badge.svg"></a> <a href="https://pepy.tech/project/pytrendy"><img src="https://static.pepy.tech/badge/pytrendy"></a></p>

---

## A Toolkit for Time Series Trend Analysis

**PyTrendy** is a modular Python toolkit for detecting, refining, and analyzing trend segments in time series data. Designed for developers and analysts working with noisy signals, PyTrendy offers a robust pipeline that combines statistical preprocessing, dynamic segmentation, and DTW-based classification to extract meaningful patterns from complex datasets.

Whether you're analyzing financial indicators, sensor outputs, or behavioral metrics, PyTrendy helps you surface directional trends, classify their nature (gradual vs abrupt), and quantify their steepness and duration - all with developer-friendly access and extensibility.

---

## Visual Demo

![Gradual Trends](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Gradual-Cropped.gif)
![Abrupt Trends](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Abrupt-Cropped.gif)
![Noise Spikes](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Noise-Spikes-Cropped.gif)
![Random Noise](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Noise-Random-Cropped.gif)

---

## Quickstart

Get started with PyTrendy in just a few steps:

```python
import pytrendy as pt

# Load your time series data
# Your DataFrame should have a date column and a value column
results = pt.detect_trends(
    df, 
    date_col='date',      # Column with datetime values
    value_col='signal',   # Column with your time series data
    plot=True             # Visualize the results
)

# Access the most significant trend
best_trend = results.best
print(f"Best trend: {best_trend['direction']} from {best_trend['start']} to {best_trend['end']}")

# View summary of all detected trends
results.print_summary()

# Filter specific trend types
uptrends = results.filter_segments(direction='Up', format='df')
print(f"\nDetected {len(uptrends)} upward trends")
```

!!! tip "New to PyTrendy?"
    See the [Getting Started Tutorial](tutorials/getting-started.md) for a complete walkthrough with examples and output.

---

## Key Features

PyTrendy is built on a modular architecture that exposes each stage of the trend detection pipeline as a standalone, extensible function. See the [User Guide](user-guide/index.md) for complete details.

* **Signal Segmentation**: Automatically detects directional segments (Up, Down, Flat, Noise) using statistical flags from preprocessed signals.
* **Boundary Refinement**: Refines segment boundaries using local extrema and change-point heuristics to ensure accuracy.
* **DTW-based Classification**: Classifies trends as gradual or abrupt by comparing detected segments against synthetic reference signals using Dynamic Time Warping.
* **Trend Ranking**: Prioritizes and ranks segments based on key metrics like steepness and total cumulative change to help identify the most significant patterns.
* **Structured Output**: Provides a clean and structured `PyTrendyResults` object, providing filtering, tabular views, and a dedicated method for selecting the best-ranked trends.

---

## Next Steps

For a complete guide on using PyTrendy, refer to the [User Guide](user-guide/index.md).

For detailed function documentation, see the [API Reference](reference/pytrendy/index.md).