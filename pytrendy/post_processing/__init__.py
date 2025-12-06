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


## 2. Segment Refinement Package

The segment refinement functionality is organized under the `segments_refine` package:

### [segments_refine](segments_refine)
Main orchestration module with `refine_segments()` function that coordinates the full post-processing pipeline.

The `segments_refine` package contains focused sub-modules:

### [segments_refine.update_neighbours](segments_refine/update_neighbours)
Helper functions for adjusting segment boundaries when neighboring segments are updated:
- `update_prev_segment`: Adjusts the end of the previous segment
- `update_next_segment`: Adjusts the start of the next segment

### [segments_refine.gradual_expand_contract](segments_refine/gradual_expand_contract)
- `expand_contract_segments`: Adjusts boundaries based on local extrema (±7 days window)

### [segments_refine.trend_classify](segments_refine/trend_classify)
- `classify_trends`: Uses Dynamic Time Warping (DTW) to label segments as 'gradual' or 'abrupt'

### [segments_refine.abrupt_shaving](segments_refine/abrupt_shaving)
- `shave_abrupt_trends`: Detects changepoints in abrupt segments using z-score outliers

### [segments_refine.segment_grouping](segments_refine/segment_grouping)
- `group_segments`: Merges short, consecutive segments with the same direction

### [segments_refine.artifact_cleanup](segments_refine/artifact_cleanup)
- `clean_artifacts`: Removes invalid segments (inversions, overlaps)
- `fill_in_flats`: Fills gaps between segments with flat regions


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
