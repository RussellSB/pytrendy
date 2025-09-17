"""*Trend Detection Pipeline for Time Series Signals*


At its core is the `detect_trends` pipeline, which carries out signal preprocessing, segment extraction, refinement, classification, 
and analysis. The package is structured into cohesive submodules— `io`, `post_processing`, `process_signals`, and `simpledtw` to 
support a flexible and interpretable workflow. These components work together to load sample datasets, visualize annotated trends, 
apply classification heuristics, and access structured results for downstream analysis or integration.

---

**This package is organized into the following modules:**

**1. `detect_trends`**
Main pipeline function that orchestrates the full trend detection workflow. It processes the input signal, extracts segments, refines boundaries, analyzes metrics, and optionally visualizes the results. Returns a structured `PyTrendyResults` object for downstream use.

**2. `io`**
Input/output utilities for working with datasets, plots, and results:

 - `data_loader`: Loads built-in datasets such as synthetic signals and classification references.
 - `plot_pytrendy`: Generates annotated matplotlib plots showing detected trend segments.
 - `results_pytrendy`: Wraps the output segments into a structured object with filtering, ranking, and summary tools.

**3. `post_processing`**
Functions for refining and analyzing detected segments:

- `segments_get`: Extracts contiguous segments based on signal flags (e.g., uptrend, flat, noise).
- `segments_refine`: Adjusts segment boundaries, classifies trends as gradual or abrupt, and removes artifacts.
- `segments_analyse`: Computes metrics like total change, percent change, duration, and signal-to-noise ratio.

**4. `process_signals`**
Core signal processing logic that applies Savitzky-Golay smoothing and rolling statistics to classify regions of the signal. Flags flat, noisy, and trending areas for segmentation.

**5. `simpledtw`**
Implementation of Dynamic Time Warping (DTW) used to compare segments against reference signals. This enables classification of trends as 'gradual' or 'abrupt' based on alignment cost.

---

Use PyTrendy to run end-to-end trend detection, visualize results, and interact with the output through a modular API.
"""


from .detect_trends import detect_trends
from .io.data_loader import load_data
from .io.plot_pytrendy import plot_pytrendy
from .simpledtw import dtw