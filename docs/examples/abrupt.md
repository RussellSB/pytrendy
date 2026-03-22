# Understanding Abrupt Trends

This tutorial explains how PyTrendy identifies and classifies *abrupt* trends within a time series.  

Abrupt trends represent short‑lived, sharp deviations in the signal.  
They typically correspond to shocks, anomalies, sudden events, or rapid structural changes.

![Abrupt Trends](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Abrupt-Cropped.gif)

---

## What Defines an Abrupt Trend

PyTrendy classifies a segment as *abrupt* when:

- The signal exhibits a rapid, high‑magnitude deviation over a short duration  
- The change is concentrated around a local extremum  
- DTW alignment identifies a sharp temporal pattern  
- The segment fails the smoothness criteria required for gradual classification  
- The duration is below the minimum threshold for a sustained trend  

Abrupt trends are often associated with anomalies, interventions, or sudden behavioural shifts.

---

## Running Trend Detection

This produces a PyTrendyResults object containing both gradual and abrupt segments.
Begin by applying the standard PyTrendy pipeline:

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

## Detecting Trends with Abrupt Padding (Default: 28 Days)

PyTrendy automatically pads abrupt segments to provide context around the spike.
By default, each abrupt segment is expanded by 28 days on both sides.

```python
results = pt.detect_trends(
    df,
    value_col="value",
    date_col="date",
    abrupt_padding=28   # default
)
```

Padding ensures that the abrupt change is interpreted within its surrounding behaviour,
which is useful for anomaly analysis, root‑cause investigation, and before/after comparisons.

---

## Detecting Trends with Custom Abrupt Padding (Example: 60 Days)

You can customise the default padding to widen the contextual window. 

```python
results = pt.detect_trends(
    df,
    value_col="value",
    date_col="date",
    abrupt_padding=60
)
```

A larger padding window is helpful when:

- The signal has slow recovery after a shock
- You want to compare long pre‑ and post‑periods
- The abrupt event has extended influence

---

## Filtering for Abrupt Segments

Once detection is complete, isolate abrupt segments:

```python
abrupt_segments = results.filter_segments(kind="abrupt")
abrupt_segments.head()
```

You may also filter by direction:

```python
abrupt_up = results.filter_segments(kind="abrupt", direction="up")
abrupt_down = results.filter_segments(kind="abrupt", direction="down")
```

---

## Selecting the Most Significant Abrupt Trends

Just like gradual trends, PyTrendy ranks segments using metrics such as:

- Absolute change
- Percent change
- Duration
- Cumulative movement
- Signal‑to‑noise ratio (SNR)
- Change rank

Retrieve the highest‑impact abrupt segments:

```python
top_abrupt = results.best(n=5, kind="abrupt")
top_abrupt
```

Or restrict by direction:

```python
top_abrupt_up = results.best(n=3, kind="abrupt", direction="up")
```

---

## Visualising Abrupt Trends

Let's visualise highlighted plots to inspect abrupt behaviour. This highlights abrupt segments 
and their padded windows, showing boundaries and metadata.

```python
pt.plot_pytrendy(
    df,
    results,
    highlight="abrupt"
)
```

---

## Next Steps

Let's explore noise next.

- **[Noise](noise.md)**