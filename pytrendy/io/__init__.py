"""**Data Loading, Visualization, and Results Access**

This module provides essential tools for interacting with input data, visualizing trend detection results,
and accessing structured outputs. These utilities support both exploratory analysis and integration into
larger workflows.

---

# Included Modules

## 1. [data_loader](data_loader)
Loads built-in datasets packaged with PyTrendy. These include:

- `'series_synthetic'`: A synthetic time series with embedded uptrends, downtrends, and flat regions.
- `'classes_trends'`: Reference signals used internally for classifying segments as gradual or abrupt.

Useful for testing, demos, and validating detection logic.

## 2. [plot_pytrendy](plot_pytrendy)

- Generates annotated matplotlib plots of detected trend segments over the original signal.
- Highlights Up, Down, Flat, and Noise regions with shaded overlays and ranks significant trends.
- Supports visual debugging and presentation-ready output.

## 3. [results_pytrendy](results_pytrendy)
Wraps the output of `detect_trends` into a structured `PyTrendyResults` object. It provides:

- Summary statistics (counts, rankings, best segment)
- Filtering by direction and ranking
- Tabular access to segment metadata

---

Use this module to prepare input data, visualize detection output, and interact with results in a clean, modular way.
"""
