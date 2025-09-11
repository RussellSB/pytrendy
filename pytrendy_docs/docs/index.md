
# PyTrendy
**PyTrendy** is a modular Python toolkit for detecting, refining, and analyzing trend segments in time series data. Designed for developers and analysts working with noisy signals, PyTrendy offers a robust pipeline that combines statistical preprocessing, dynamic segmentation, and DTW-based classification to extract meaningful patterns from complex datasets.

Whether you're analyzing financial indicators, sensor outputs, or behavioral metrics, PyTrendy helps you surface directional trends, classify their nature (gradual vs abrupt), and quantify their steepness and duration - all with developer-friendly access and extensibility.

## Key Features 

- Detects directional segments (Up, Down, Flat, Noise) from preprocessed signal flags

- Refines segment boundaries using local extrema and changepoint heuristics

- Classifies trends using DTW against synthetic gradual/abrupt profiles

- Ranks segments by steepness and cumulative change

- Summarizes results with filtering, tabular views, and best-trend selection

## Quickstart

```python
from pytrendy import detect_trends, PyTrendyResults

# Load your time series DataFrame
df = load_your_data()  # Must include columns: ['signal', 'noise', 'trend_flag']

# Run detection pipeline
segments = detect_trends(df, value_col='signal')

# Wrap results for filtering and summary
results = PyTrendyResults(segments)

# Print summary
results.print_summary()

# Access best trend
best_trend = results.best
```

For a full walkthrough, see Usage. 

For detailed module references, see API Reference.

## Modular Architecture
PyTrendy is built with modularity in mind. Each stage of the pipeline — from raw segmentation to final classification — is exposed as a standalone function, allowing developers to customize, extend, or integrate PyTrendy into larger analytical workflows.

[![PyPI version](https://badge.fury.io/py/pytrendy.svg)](https://pypi.org/project/pytrendy/)
[![Build Status](https://github.com/RussellSB/pytrendy/actions/workflows/tests.yml/badge.svg)](https://github.com/RussellSB/pytrendy/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/RussellSB/pytrendy?tab=MIT-1-ov-file)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://russellsb.github.io/pytrendy/)








