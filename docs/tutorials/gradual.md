# Understanding Gradual Trends

This tutorial explains how PyTrendy identifies and classifies *gradual* trends within a time series.  

Gradual trends represent sustained directional movement where the signal changes smoothly over time.
They differ from abrupt trends, which exhibit sharp, short‑lived deviations.

![Gradual Trends](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Gradual-Cropped.gif)

---

## What Defines a Gradual Trend

PyTrendy classifies a segment as *gradual* when:

- The signal shows consistent directional movement (upward or downward).
- The change occurs over a longer duration rather than a sudden spike.
- Local extrema and DTW alignment indicate smooth temporal progression.
- The segment passes minimum‑length and noise‑filtering thresholds.

Gradual trends are typically associated with structural shifts, long‑term behaviour, or meaningful
business changes rather than short‑term volatility.



## Running Trend Detection

Let's start by detecting trends. This produces a PyTrendyResults object containing all detected segments, 
including their classification as gradual, abrupt or noise.

```py
import pytrendy as pt
```
```py
results = pt.detect_trends(
    df, value_col="value", date_col="date", window=7, polyorder=2
)
```

---

## Filtering for Gradual Segments

Once detection is complete, you can isolate gradual trends. This returns only the segments where PyTrendy’s
refinement step has assigned the label "gradual".

```py
gradual_segments = results.filter_segments(kind="gradual")
gradual_segments.head()
```

You may also filter by direction.

```py
gradual_up = results.filter_segments(kind="gradual", direction="up")
gradual_down = results.filter_segments(kind="gradual", direction="down")
```

---

## Selecting the Most Significant Gradual Trends 

PyTrendy computes several metrics for each segment, including:

- Absolute change
- Percent change
- Duration
- Cumulative movement
- Signal‑to‑noise ratio (SNR)
- Change rank

To retrieve the highest‑impact gradual trends:

```py
top_gradual = results.best(n=5, kind="gradual")
top_gradual
```

Or restrict by direction:

```py
top_gradual_up = results.best(n=3, kind="gradual", direction="up")
```

---

## Visualising Gradual Trends

You can now generate highlighted plots to inspect gradual behaviour. This highlights all gradual segments
on the time series, showing their boundaries and metadata.

```py
pt.plot_pytrendy(
    df,
    results,
    highlight="gradual"
)
```

## Next Steps

Let's explore abrupt trends and noise next.

- **[Abrupt](abrupt.md)** 
- **[Noise](noise.md)**
