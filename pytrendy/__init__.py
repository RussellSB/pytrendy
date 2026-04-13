"""*Trend Detection Pipeline for Time Series Signals*


At its core is the `detect_trends` pipeline, which carries out signal preprocessing, segment extraction, refinement, classification, 
and analysis. The package is structured into cohesive submodules - `io`, `post_processing`, `process_signals`, and `simpledtw` to 
support a flexible and interpretable workflow. These components work together to load sample datasets, visualize annotated trends, 
apply classification heuristics, and access structured results for downstream analysis or integration.

---

This package is organized into the following modules:

# 1. [detect_trends](detect_trends)

- Defines the primary pipeline function for executing PyTrendy's trend detection workflow.
- This function coordinates signal preprocessing, segment extraction, boundary refinement, metric analysis, and optional visualization in a single call.
- It returns a structured `PyTrendyResults` object containing enriched, ranked, and classified trend segments suitable for filtering, plotting, or downstream integration.


# 2. [io](io)

The `io` module provides essential interfaces for interacting with the input and output layers of PyTrendy. 
It streamlines access to curated datasets, supports detailed visualization of trend segments, and offers structured result handling for downstream analysis.  
Designed for both exploratory workflows and programmatic integration, this module enables users to efficiently load data, interpret results, and present findings.

## 2.1 [data_loader](io/data_loader)

- Provides access to built-in datasets packaged with PyTrendy.
- Enables quick loading of synthetic time series and classification references.
- Supports testing, demonstration, and validation workflows.
- Delivers standardized input formats optimized for trend detection and segment analysis.


## 2.2 [plot_pytrendy](io/plot_pytrendy)

- Generates annotated visualizations of trend segments over time series data.
- Highlights matplotlib plots with Up, Down, Flat, and Noise regions using shaded overlays and metadata.
- Facilitates visual inspection, debugging, and interpretation of results.
- Makes the visualization ready for reporting, presentation, and analytical review.


## 2.3 [results_pytrendy](io/results_pytrendy)

- Wraps detection output into a structured results object.
- Implements the `PyTrendyResults` class for segment filtering, ranking, and summarization.
- Provides access to structured segment metadata and computed metrics.
- Supports integration with downstream analysis, reporting pipelines, and export workflows.



# 3. [post_processing](post_processing)

The `post_processing` module provides utilities for refining, classifying, and analyzing trend segments.
It transforms raw detections into interpretable, ranked structures by adjusting boundaries, labeling temporal behavior, and computing signal metrics.  
This module ensures that the output is analytically robust and ready for downstream use.

## 3.1 [segments_get](post_processing/segments_get)

- Extracts continuous segments from the `trend_flag` column.
- Applies minimum length constraints to filter out noise.
- Supports directional trends (Up/Down) and neutral regions (Flat/Noise).
- Serves as the first step in segment-level post-processing.


## 3.2 [segments_refine](post_processing/segments_refine)

- Adjusts segment boundaries based on local extrema and changepoint detection.
- Classifies segments as 'gradual' or 'abrupt' using DTW alignment.
- Shaves abrupt segments using z-score outlier detection.
- Groups short consecutive segments and removes artifacts.


## 3.3 [segments_analyse](post_processing/segments_analyse)

- Computes metrics for each segment, comparing pretreatment vs post-treatment behavior.
- Includes absolute and percent change, duration, and cumulative movement.
- Calculates signal-to-noise ratio (SNR) and assigns a change rank.
- Enables filtering and prioritization of significant trends.



# 4. [process_signals](process_signals)

- Implements core signal processing logic to identify meaningful regions within a time series.
- By applying Savitzky-Golay smoothing and rolling statistical measures, this module flags flat, noisy, and directional trends.
- These flags serve as the foundation for segment extraction and subsequent analysis within the PyTrendy pipeline.


# 5. [simpledtw](simpledtw)

- Provides an efficient implementation of Dynamic Time Warping (DTW) for comparing time series segments.
- This module is used internally to classify trends by aligning detected segments with reference signals and evaluating similarity based on alignment cost.
- It supports robust temporal comparison for distinguishing gradual versus abrupt patterns.

---

Use PyTrendy to run end-to-end trend detection, visualize results, and interact with the output through a modular API.
"""


from .detect_trends import detect_trends
from .io.data_loader import load_data
from .io.plot_pytrendy import plot_pytrendy
from .simpledtw import dtw