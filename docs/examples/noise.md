# Understanding Noise Segments

This tutorial explains how PyTrendy identifies *noise* within a time series.  
Noise segments represent regions where the signal does not exhibit meaningful directional movement
and instead fluctuates randomly or erratically.

PyTrendy includes specialised handling for several noise patterns, including spike‑based noise and
Gaussian/random noise. These behaviours are validated through the test suite, particularly
`test.py` and `test_noise_random.py`.

![Noise Spikes](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Noise-Spikes-Cropped.gif)

---

## What Defines a Noise Segment

A segment is classified as *noise* when:

- The signal shows no sustained upward or downward direction  
- Fluctuations are short‑lived and inconsistent  
- Local extrema do not form a coherent trend  
- The segment fails both gradual and abrupt classification criteria  
- The movement resembles random or stochastic variation  

Noise segments help isolate regions that should not be interpreted as meaningful trends.

---

## Running Trend Detection

This produces a `PyTrendyResults` object containing all segments, including those classified as noise.

```python
import pytrendy as pt

results = pt.detect_trends(
    df,
    value_col="value",
    date_col="date",
    window=7,
    polyorder=2
)
```

---

## Gaussian / Random Noise (Edge Case)

Gaussian noise refers to random fluctuations drawn from a normal distribution.
These patterns:

- Oscillate around a mean
- Show no directional persistence
- Contain many small, alternating movements
- Fail all trend‑shape criteria

Example from `test_noise_random.py`:

```python
results = pt.detect_trends(df_gaussian)
noise_segments = results.filter_segments(kind="noise")
```

PyTrendy correctly groups these regions as noise, ensuring that random variation does not appear as a trend.

---

## Filtering for Noise Segments

After detection, isolate noise segments:

```python
noise_segments = results.filter_segments(kind="noise")
noise_segments.head()
```

You may also filter by duration or other metrics:

```python
short_noise = noise_segments[noise_segments["duration"] < 10]
```

Noise segments are often useful for diagnostics, smoothing decisions, and understanding baseline volatility.

---

## Selecting the Most Relevant Noise Segments

Although noise is not ranked as a “trend,” PyTrendy still computes metrics such as:

- Duration
- Amplitude range
- Local volatility
- Signal‑to‑noise ratio (SNR)

To retrieve the largest or most volatile noise segments:

```python
top_noise = results.best(n=5, kind="noise")
top_noise
```

This is helpful when analysing instability or identifying regions that may require preprocessing.

---

## Visualising Noise Segments

Generate annotated plots to inspect noise behaviour:

```python
pt.plot_pytrendy(
    df,
    results,
    highlight="noise"
)
```

This highlights noise regions on the time series, showing their boundaries and metadata.


## Next steps