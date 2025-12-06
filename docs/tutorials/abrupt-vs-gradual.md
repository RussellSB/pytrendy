# Tutorial 3: Abrupt vs Gradual Trends

<span class="badge-advanced">ADVANCED</span> <span class="badge-time">10 minutes</span>

PyTrendy distinguishes between gradual and abrupt transitions using Dynamic Time Warping (DTW). This tutorial explores the classification system and the `is_abrupt_padded` parameter.

---

## Understanding Trend Classification

**Gradual trends** change smoothly over time (e.g., seasonal growth, long-term adoption).

**Abrupt trends** have sharp transitions (e.g., product launches, market crashes, policy interventions).

---

## Example: Detecting Abrupt Changes

```python
import pytrendy as pt

# Load dataset with abrupt transitions
df = pt.load_data('series_synthetic')

# Detect trends on the 'abrupt' signal
results_default = pt.detect_trends(
    df, 
    date_col='date', 
    value_col='abrupt', 
    plot=False
)

# Check classification
for seg in results_default.segments:
    if 'trend_class' in seg:
        print(f"{seg['direction']} ({seg['trend_class']}): "
              f"{seg['start']} to {seg['end']}, "
              f"{seg['days']} days")
```

**Output:**

```
uptrend (gradual): 2025-01-01 to 2025-01-30, 30 days
uptrend (abrupt): 2025-01-31 to 2025-02-05, 6 days
downtrend (abrupt): 2025-02-06 to 2025-02-12, 7 days
flat (gradual): 2025-02-13 to 2025-03-10, 26 days
```

---

## Using Abrupt Padding

For quasi-experimental designs or intervention analysis, you may want to extend abrupt trends to capture post-intervention effects.

```python
# Without padding
results_no_pad = pt.detect_trends(
    df, 
    date_col='date', 
    value_col='abrupt', 
    plot=False,
    method_params={'is_abrupt_padded': False}
)

# With padding (28 days)
results_padded = pt.detect_trends(
    df, 
    date_col='date', 
    value_col='abrupt', 
    plot=False,
    method_params={
        'is_abrupt_padded': True,
        'abrupt_padding': 28
    }
)

# Compare segment lengths
print("Without padding:")
for seg in results_no_pad.segments:
    if 'trend_class' in seg and seg['trend_class'] == 'abrupt':
        print(f"  {seg['direction']}: {seg['days']} days")

print("\nWith 28-day padding:")
for seg in results_padded.segments:
    if 'trend_class' in seg and seg['trend_class'] == 'abrupt':
        print(f"  {seg['direction']}: {seg['days']} days")
```

**Output:**

```
Without padding:
  uptrend: 6 days
  downtrend: 7 days

With 28-day padding:
  uptrend: 34 days
  downtrend: 35 days
```

---

## When to Use Padding

!!! tip "Best use cases for padding"
    - **Intervention studies**: Extend abrupt changes to measure sustained effects
    - **Policy analysis**: Capture post-policy stabilization periods
    - **A/B testing**: Include post-treatment observation windows
    - **Event studies**: Measure aftermath of discrete events

---

## Tutorial Complete!

!!! success "You've learned:"
    - Difference between gradual and abrupt trends  
    - How to configure abrupt padding  
    - When to use each detection strategy  
    - How padding affects segment boundaries

!!! note "See also"
    For detailed parameter options, see the [API Reference](../reference/pytrendy/index.md).

!!! tip "Try It Yourself"
    **Challenge**: Experiment with different padding values (e.g., 14, 21, 35 days). Plot the results side-by-side. How does padding affect the identification of post-event stabilization periods?

---

## Next Steps

- **[Getting Started Tutorial](getting-started.md)** - Review the basics
- **[Custom Pipeline Tutorial](custom-pipeline.md)** - Build detection pipelines from scratch
- **[Real-World Examples](real-world-examples.md)** - Bitcoin, GitHub, and climate data
- **[User Guide](../user-guide/index.md)** - Complete features and usage reference
- **[API Reference](../reference/pytrendy/index.md)** - Full API documentation
