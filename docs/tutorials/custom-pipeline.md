# Tutorial 2: Custom Pipeline Mastery

<span class="badge-advanced">ADVANCED</span> <span class="badge-time">15 minutes</span>

For advanced users who need granular control over each stage of trend detection, PyTrendy exposes the full pipeline as modular functions.

---

## Why Use a Custom Pipeline?

!!! tip "Use cases for manual pipeline"
    - Debugging: Inspect intermediate outputs at each stage
    - Customization: Modify preprocessing parameters
    - Research: Analyze signal processing flags
    - Integration: Embed into larger analytical workflows
    - Visualization: Plot each transformation step

!!! warning "Important"
    The custom pipeline requires explicit imports from submodules. These are **not** exported at the top-level `pytrendy` namespace.

---

## Pipeline Overview

!!! note "5-Stage Pipeline"
    1. Signal Processing → Smoothing + flag detection
    2. Segment Extraction → Identify contiguous regions
    3. Boundary Refinement → Adjust edges + classify trends
    4. Metric Analysis → Compute statistics + rankings
    5. Visualization → Plot annotated results

---

## Step 1: Import Pipeline Functions

Import the modular pipeline functions:

```python
import pytrendy as pt
from pytrendy.process_signals import process_signals
from pytrendy.post_processing.segments_get import get_segments
from pytrendy.post_processing.segments_refine import refine_segments
from pytrendy.post_processing.segments_analyse import analyse_segments
from pytrendy.io.results_pytrendy import PyTrendyResults
from pytrendy.io.plot_pytrendy import plot_pytrendy
import pandas as pd
```

!!! example "Import Note"
    These functions are **not** exported at `pytrendy.*` level. You must import them from their specific submodules.

---

## Step 2: Load and Prepare Data

Prepare the dataset for manual processing:

```python
# Load data
df = pt.load_data('series_synthetic')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df[['gradual']]

print(f"Prepared data shape: {df.shape}")
print(df.head())
```

**Output:**

```
Prepared data shape: (181, 1)
            gradual
date               
2025-01-01  12.500000
2025-01-02  13.421717
2025-01-03  13.474026
2025-01-04  13.474026
2025-01-05  14.505772
```

---

## Step 3: Signal Processing

Apply smoothing and flag detection:

```python
# Apply smoothing and flag detection
df = process_signals(df, value_col='gradual')

# Inspect the flags
print(df[['gradual', 'smoothed', 'trend_flag', 'flat_flag', 'noise_flag']].head(20))
```

**Output:**

```
            gradual  smoothed  trend_flag  flat_flag  noise_flag
date                                                             
2025-01-01  12.500000  12.685571           0          0           1
2025-01-02  13.421717  13.236652           1          0           0
2025-01-03  13.474026  13.474026           0          1           0
2025-01-04  13.474026  13.607809           1          0           0
2025-01-05  14.505772  14.429474           1          0           0
```

**New columns added:**

| Column | Description | Values |
|--------|-------------|--------|
| `smoothed` | Savitzky-Golay filtered signal | Float |
| `smoothed_std` | Rolling standard deviation | Float |
| `snr` | Signal-to-noise ratio (dB) | Float |
| `smoothed_deriv` | First derivative (slope) | Float |
| `trend_flag` | Directional classification | 1 (Up), -1 (Down), -2 (Flat), -3 (Noise) |
| `flat_flag` | Low-variance indicator | 0 or 1 |
| `noise_flag` | High-noise indicator | 0 or 1 |

!!! tip "Debugging Tip"
    Plot `df[['gradual', 'smoothed', 'trend_flag']]` to visualize how flags align with the original signal.

---

## Step 4: Segment Extraction

Extract raw segments from the processed signal:

```python
# Extract raw segments
segments_raw = get_segments(df)

print(f"Detected {len(segments_raw)} raw segments")
for seg in segments_raw[:3]:
    print(f"{seg['direction']}: {seg['start']} to {seg['end']}")
```

**Output:**

```
Detected 12 raw segments
uptrend: 2025-01-02 to 2025-01-15
flat: 2025-01-16 to 2025-01-25
downtrend: 2025-01-26 to 2025-02-05
```

---

## Step 5: Segment Refinement

Refine segment boundaries and classify trends:

```python
# Refine boundaries and classify trends
method_params = {
    'is_abrupt_padded': False,
    'abrupt_padding': 28
}

segments_refined = refine_segments(
    df, 
    value_col='gradual', 
    segments=segments_raw, 
    method_params=method_params
)

# Check if any segments were classified
for seg in segments_refined:
    if 'trend_class' in seg:
        print(f"{seg['direction']} ({seg['trend_class']}): {seg['start']} to {seg['end']}")
```

**Output:**

```
uptrend (gradual): 2025-01-02 to 2025-01-18
flat (gradual): 2025-01-19 to 2025-01-28
downtrend (gradual): 2025-01-29 to 2025-02-10
uptrend (abrupt): 2025-02-11 to 2025-02-15
```

**What happened:**

- Boundaries adjusted using local extrema
- Trends classified as 'gradual' or 'abrupt' via DTW
- Short segments grouped
- Artifacts cleaned up

---

## Step 6: Metric Analysis

Add quantitative metrics to each segment:

```python
# Add quantitative metrics
segments_final = analyse_segments(df, value_col='gradual', segments=segments_refined)

# Inspect enhanced segments
for seg in segments_final:
    if 'total_change' in seg:
        print(f"Rank {seg['change_rank']}: {seg['direction']}, "
              f"Change: {seg['total_change']:.2f}, "
              f"Duration: {seg['days']} days, "
              f"SNR: {seg['SNR']:.2f}")
```

**Output:**

```
Rank 1: uptrend, Change: 15.42, Duration: 17 days, SNR: 12.34
Rank 2: downtrend, Change: -8.73, Duration: 13 days, SNR: 10.56
Rank 3: uptrend, Change: 6.21, Duration: 5 days, SNR: 8.92
Rank 4: flat, Change: 0.52, Duration: 10 days, SNR: 5.43
```

---

## Step 7: Visualize and Wrap Results

Plot the results and wrap them in a results object:

```python
# Plot the results
plot_pytrendy(df, value_col='gradual', segments_enhanced=segments_final)

# Wrap in results object
results = PyTrendyResults(segments_final)
results.print_summary()
```

**Output:**

```
=== PyTrendy Results Summary ===
Total segments: 8
Uptrends: 4
Downtrends: 2
Flat periods: 2

Top 3 significant trends by total change:
  Rank 1: uptrend from 2025-01-02 to 2025-01-18 (change: +15.42, duration: 17 days)
  Rank 2: downtrend from 2025-01-29 to 2025-02-10 (change: -8.73, duration: 13 days)
  Rank 3: uptrend from 2025-02-11 to 2025-02-15 (change: +6.21, duration: 5 days)
```

---

## Tutorial Complete!

!!! success "You've mastered the custom pipeline!"
    - Imported pipeline functions from submodules  
    - Processed signals with smoothing and flags  
    - Extracted and refined segments  
    - Analyzed metrics and visualized results

!!! tip "Try It Yourself"
    **Challenge**: Modify the `method_params` in Step 5 (e.g., change `abrupt_padding` to 14 days). Run the full pipeline and compare the segment counts. How does this affect the final results?

---

## Next Steps

- **[Getting Started Tutorial](getting-started.md)** - Review the basics
- **[Abrupt vs Gradual Tutorial](abrupt-vs-gradual.md)** - Understand trend classification
- **[Real-World Examples](real-world-examples.md)** - Bitcoin, GitHub, and climate data
- **[User Guide](../user-guide/index.md)** - Complete features and usage reference
- **[API Reference](../reference/pytrendy/index.md)** - Full API documentation
