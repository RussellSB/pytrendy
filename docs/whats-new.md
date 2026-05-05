# What's New

Stay up to date with every PyTrendy release — from user-facing improvements to bug fixes that directly affect your results.

!!! tip "Pre-release docs"
    You are viewing the **develop** (pre-release) build.  
    The section at the top reflects changes that are staged for the next stable release.  
    Stable-only users can always switch to the **main** docs via the badge in the header.

---

<!-- WHATS_NEW_CONTENT_START -->

## 🔬 Coming in v1.1.11 <span class="badge-prerelease">pre-release</span>

*These changes are merged on the `develop` branch and will land in the next stable release.*

### 🐛 Bug Fixes

**Trend detection on normalised time series**

Previously, `detect_trends()` could fail to detect any trends when the input signal was scaled to the `[0, 1]` range (e.g., after min-max normalisation). The root cause was an absolute threshold that was too large relative to the signal amplitude.

=== "Before (≤ 1.1.10)"

    ```python
    import numpy as np, pandas as pd
    from pytrendy import detect_trends

    # Values all within [0, 1]
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    values = np.linspace(0.1, 0.9, 60)          # clear uptrend, tiny scale
    df = pd.DataFrame({"date": dates, "value": values})

    result = detect_trends(df)
    print(result.df)  # ← could return empty DataFrame
    ```

=== "After (v1.1.11+)"

    ```python
    import numpy as np, pandas as pd
    from pytrendy import detect_trends

    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    values = np.linspace(0.1, 0.9, 60)          # same data, now correctly detected
    df = pd.DataFrame({"date": dates, "value": values})

    result = detect_trends(df)
    print(result.df)  # ← returns the detected gradual uptrend
    ```

**Metrics for all segment types**

Segment metrics (e.g., `change_rate`, `change_rank`) were not being computed for every trend type. This patch ensures all output rows carry complete metric columns regardless of classification.

---

## ✅ Released in v1.1.10

> Released **2026-03-21**

### 🧪 Testing

- Comprehensive automated tests were added for noise-related edge cases and crash scenarios, reaching full code coverage for the noise detection module.
- Artefact-cleaning helpers were refactored to be more testable and deterministic.
- Several `pytest-mpl` baseline images were corrected.

---

## ✅ Released in v1.1.9

> Released **2026-02-07**

### 🐛 Bug Fixes

- Improved spike precision in noise detection for signals that contain a single dominant outlier surrounded by otherwise stable values.

---

## ✅ Released in v1.1.8

> Released **2025-11-15**

### 🐛 Bug Fixes — Noise & Flat Detection

A focused round of improvements to two of the trickiest parts of the pipeline:

| Area | What changed |
|---|---|
| **Flat fill-in** | Now covers regions that fall outside any detected segment range, preventing visual white gaps |
| **Flat fill-in** | Correctly skips zero-day leading / trailing regions and handles grouped segments |
| **Noise detection** | Better precision when a spike sits on an otherwise flat-zero signal |
| **Noise detection** | Improved sensitivity when flat conversions emerge from noisy gradual trends |
| **Noise detection** | Handles the "semi-flat gradual in noise" edge case; `trend_too_flat` is now treated as flat rather than noise |
| **Noise detection** | Relies on the actual signal (not the smoothed derivative) for up/down classification, reducing the need for artefact cleaning downstream |

---

## ✅ Released in v1.1.7

> Released **2025-11-01**

### 🐛 Bug Fixes

- **Expand-contract:** gradual trends can now be retroactively updated when a newer gradual changes the reference baseline.
- **Noise detection:** several edge cases around abrupt-noise boundaries, opposite-direction overlaps, and post-grouping checks were resolved.
- **Plot:** visual displacement is now only applied when it does not break the up/down direction contract.

---

## 🚀 Released in v1.1.0

> Released **2025-10-15** — *Minor version: new features & major robustness overhaul*

This release represents the most significant update to PyTrendy's core engine since the initial launch.

### ✨ New Feature — Flat Fill-In

Gaps between classified segments are now automatically filled with a **flat** segment, so the output completely covers the input time range.

```python
result = detect_trends(df)
print(result.df)
# Every day is now classified — no more unexplained gaps in the output
```

### ✨ New Feature — Improved Results Interface

The `TrendyResults` object gained two cleaner access patterns:

=== "Before (≤ 1.0.x)"

    ```python
    segments_df = result.segments_df          # detailed rows
    summary_df  = result.summary["df"]        # summarised view
    ```

=== "After (v1.1.0+)"

    ```python
    segments_df = result.df                   # detailed rows (shorter alias)
    summary_df  = result.df_summary           # summarised view (direct attribute)
    ```

Both the old and new names continue to work in v1.1.x.

### 🔧 Core Signal Processing Revamp

The signal processing and post-processing pipeline underwent a **major revamp** ([#8](https://github.com/RussellSB/pytrendy/issues/8)):

- Much more robust handling of edge cases across all trend types.
- Abrupt trend detection improved with better shaving, sub-segmentation, and direction-sensitivity.
- Gradual swallowing logic now stretches flexibly across neighbouring segment adjustments.
- Grouping logic updated: abrupt segments that are exactly touching are now correctly grouped.

### 🐛 Highlights from the Bug Fix Backlog

- `has_inverse()` now also validates total-change consistency, not just direction.
- Windows path separators now handled correctly in the data loader.
- `detect_trends()` is now robust to wide DataFrames that contain non-numeric string columns.
- Fixed a crash when no segments are detected (rare, but possible in real-world data).

---

## 📦 Released in v1.0.x

> Initial releases — **August–September 2025**

### 🎉 Initial Release (v1.0.0)

PyTrendy launched with:

- **Gradual trend detection** — identify slow, sustained up or down movements.
- **Abrupt trend detection** — pinpoint sharp, step-like changes in a signal.
- **Flat detection** — recognise periods of no meaningful movement.
- **One-line API:**

  ```python
  from pytrendy import detect_trends
  result = detect_trends(df, value_col="sales", date_col="date")
  result.plot()
  ```

- **Segment summary** with change magnitude, direction, and duration.
- **Matplotlib integration** — `result.plot()` renders an annotated time-series chart out of the box.

<!-- WHATS_NEW_CONTENT_END -->
