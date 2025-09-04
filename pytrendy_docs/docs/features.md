# Key Features

- **Signal Processing**: Robust smoothing and noise detection using Savitzky Golay filters and SNR analysis.
- **Trend Segmentation**: Automatically detects and partitions time series into up, down, flat, or noisy segments.
- **Refinement**: Adjusts boundaries, merges segments, and removes artifacts for more accurate results.
- **Trend Classification**: Distinguishes between gradual and abrupt trends using Dynamic Time Warping (DTW).
- **Analysis**: Provides metrics such as absolute/relative change, percentage change, duration, and signal-to-noise ratio (SNR).
- **Ranking**: Ranks trends from steepest to shallowest based on total change.
- **Visualization**: Easy-to-read plots of detected trends with shaded regions and rankings.
- **Convenient API**: `detect_trends()` provides a 5-step pipeline and returns results in a simple wrapper object (`PyTrendyResults`).