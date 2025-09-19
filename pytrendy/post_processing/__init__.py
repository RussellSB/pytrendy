"""**Segment Refinement, Classification and Analysis**


This module contains utilities for enhancing the precision, interpretability, and usability of trend segments
detected by the PyTrendy pipeline. It operates on raw segments extracted from signal flags and applies
boundary adjustments, classification heuristics, and quantitative analysis.

---

# Included Modules

## 1. [segments_get](segments_get)
Extracts contiguous segments from the `trend_flag` column produced by signal processing.

Applies minimum length constraints to ensure meaningful segments are retained:

- Up/Down trends: ≥ 7 days
- Flat/Noise regions: ≥ 3 days


## 2. [segments_refine](segments_refine)
Refines segment boundaries and improves classification accuracy through multiple steps:

- `expand_contract_segments`: Adjusts boundaries based on local extrema.
- `classify_trends`: Uses Dynamic Time Warping (DTW) to label segments as 'gradual' or 'abrupt'.
- `shave_abrupt_trends`: Detects changepoints in abrupt segments using z-score outliers.
- `group_segments`: Merges short, consecutive segments with the same direction.
- `clean_artifacts`: Removes segments that are too short to be meaningful.


## 3. [segments_analyse](segments_analyse)
Adds quantitative descriptors to each segment, comparing pretreatment vs post-treatment behavior.

Metrics include:

- Absolute and percent change
- Duration in days
- Cumulative total change
- Signal-to-noise ratio (SNR)
- Change rank based on steepness and length

---

Use this module to transform raw segment flags into interpretable, ranked, and visually meaningful trend segments.
"""
