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


<div class='transparent'>
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
</div>

---

## Step 3: Signal Processing

Apply smoothing and flag detection:

```python
# Apply smoothing and flag detection
df = process_signals(df, value_col='gradual')

# Inspect the flags
print(df[['gradual', 'smoothed', 'trend_flag', 'flat_flag', 'noise_flag']].head(20))
```

<div class='transparent'>
```
-------------------------------------------------------------------------------
            date        gradual     smoothed    trend_flag   flat_flag   noise_flag
index                                                                               
1       2025-01-01    12.500000    11.890242        1            0            0
2       2025-01-02    13.421717    12.509611        1            0            0
3       2025-01-03    13.474026    13.128981        1            0            0
4       2025-01-04    13.474026    13.748351        1            0            0
5       2025-01-05    14.505772    14.367721        1            0            0
6       2025-01-06    14.709596    14.987090        1            0            0
7       2025-01-07    14.783550    15.606460        1            0            0
8       2025-01-08    16.354618    16.225830        1            0            0
9       2025-01-09    17.370130    16.819264        1            0            0
10      2025-01-10    16.493506    17.403800        1            0            0
11      2025-01-11    15.620491    18.120911        1            0            0
12      2025-01-12    16.868687    18.878487        1            0            0
13      2025-01-13    19.686147    19.440055        1            0            0
14      2025-01-14    21.560245    19.903620        1            0            0
15      2025-01-15    22.564935    20.543951        1            0            0
16      2025-01-16    21.401515    21.238717        1            0            0
17      2025-01-17    22.189755    21.909712        1            0            0
18      2025-01-18    24.230700    22.439534        1            0            0
19      2025-01-19    24.837662    22.817420       -2            1            0
20      2025-01-20    22.929293    22.915785       -2            1            0
-------------------------------------------------------------------------------

```
</div>

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


<div class='transparent'>
```
Detected 12 raw segments
Up: 2025-01-01 to 2025-01-18
Flat: 2025-01-19 to 2025-01-24
Down: 2025-01-25 to 2025-02-04
```
</div>

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

<div class='transparent'>
```
Up (gradual): 2025-01-02 to 2025-01-24
Down (gradual): 2025-01-25 to 2025-02-05
Up (gradual): 2025-02-10 to 2025-03-14
Down (gradual): 2025-03-18 to 2025-04-01
Up (gradual): 2025-04-02 to 2025-05-08
Down (gradual): 2025-05-09 to 2025-06-17
```
</div>

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

<div class='transparent'>
```
Rank 5: Up, Change: 14.01, Duration: 22 days, SNR: 22.21
Rank 6: Down, Change: -13.56, Duration: 11 days, SNR: 17.36
Rank 3: Up, Change: 24.63, Duration: 32 days, SNR: 18.87
Rank 4: Down, Change: -22.72, Duration: 14 days, SNR: 16.76
Rank 2: Up, Change: 72.61, Duration: 36 days, SNR: 21.70
Rank 1: Down, Change: -73.25, Duration: 39 days, SNR: 21.12
```
</div>

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


<div class='transparent'> 
    <img src="../assets/images/custom-pipeline-plot.png" alt="Plot">
```
Detected: 
- 3 Uptrends. 
- 3 Downtrends.
- 3 Flats.
- 0 Noise.

The best detected trend is Down between dates 2025-05-09 - 2025-06-17

Full Results:
-------------------------------------------------------------------------------
            direction       start         end  days  total_change  change_rank trend_class
time_index                                                                               
1                 Up  2025-01-02  2025-01-24    22     14.013348          5.0     gradual
2               Down  2025-01-25  2025-02-05    11    -13.564214          6.0     gradual
3               Flat  2025-02-06  2025-02-09     3           NaN          NaN         NaN
4                 Up  2025-02-10  2025-03-14    32     24.632035          3.0     gradual
5               Flat  2025-03-15  2025-03-17     2           NaN          NaN         NaN
6               Down  2025-03-18  2025-04-01    14    -22.721861          4.0     gradual
7                 Up  2025-04-02  2025-05-08    36     72.611833          2.0     gradual
8               Down  2025-05-09  2025-06-17    39    -73.253968          1.0     gradual
9               Flat  2025-06-18  2025-06-30    12           NaN          NaN         NaN 
-------------------------------------------------------------------------------
```
</div>

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
