# Pipeline Architecture

PyTrendy uses a **5-stage modular pipeline** where each stage can be used independently or as part of the complete workflow:

---

## Stage 1: Signal Preprocessing

PyTrendy performs signal conditioning prior to trend analysis:

- **Savitzky–Golay smoothing**: Applies a polynomial filter to suppress high-frequency noise while retaining signal morphology
- **Flat region identification**: Detects low-variance segments using a rolling standard deviation threshold
- **Noise quantification**: Calculates the Signal-to-Noise Ratio (SNR) to isolate unreliable data points
- **Gradient analysis**: Computes the first derivative of the smoothed signal to capture directional changes

**Result**: The processed signal is annotated with trend labels: `Up`, `Down`, `Flat`, or `Noise`.

---

## Stage 2: Trend Segmentation

The signal is segmented into contiguous regions based on directional characteristics:

- **Up**: Monotonic or sustained positive slope
- **Down**: Monotonic or sustained negative slope
- **Flat**: Minimal variance within threshold bounds
- **Noise**: High-frequency volatility exceeding SNR limits

**Segment metadata includes:**

- Temporal bounds (start and end timestamps)
- Duration in days
- Trend classification (`Up`, `Down`, `Flat`, `Noise`)
- Positional index within the time series

**Result**: The time series is decomposed into a structured sequence of labeled segments.

---

## Stage 3: Segment Refinement

Post-segmentation refinement eliminates artifacts and enhances segment fidelity:

- **Boundary optimization**: Dynamically adjusts segment edges to better align with signal transitions
- **Trend classification**: Differentiates gradual vs. abrupt transitions using Dynamic Time Warping (DTW) similarity metrics
- **Abrupt trend decomposition**: Detects latent change points within sharp directional shifts
- **Directional merging**: Consolidates adjacent short segments exhibiting consistent trend direction
- **Artifact filtering**: Removes segments below duration or significance thresholds

**Result**: Refined segments exhibit improved continuity, reduced noise, and enhanced interpretability.

---

## Stage 4: Segment Analysis

Each refined segment is enriched with quantitative metrics:

- **Absolute Change**: Net difference in signal value between segment boundaries
- **Relative Change**: Percentage change relative to the segment's starting value
- **Duration**: Total length of the segment, measured in calendar days
- **Cumulative Change**: Aggregated directional movement across the segment timeline
- **Signal-to-Noise Ratio (SNR)**: Quantifies trend clarity by comparing signal strength to background variability
- **Gradient Ranking**: Orders segments by magnitude of directional slope, from steepest to shallowest

**Result**: These metrics enable comparative analysis, prioritization, and deeper inspection of segment dynamics.

---

## Stage 5: Visualization & Results

PyTrendy provides structured output and visualization:

- **Segment highlighting**: Time series plots with color-coded regions for each segment type
- **Trend rank annotation**: Top-ranked segments automatically labeled with ordinal markers
- **Structured `PyTrendyResults` object**: Provides filtering, tabular views, and selection methods

**Result**: Publication-ready visualizations and programmatic access to all results.

---

## Next Steps

- **[Quick Start Guide](quick-start.md)** - Start using the pipeline
- **[Advanced Usage](advanced-usage.md)** - Use individual pipeline stages
- **[Configuration Reference](configuration-reference.md)** - Customize pipeline behavior
