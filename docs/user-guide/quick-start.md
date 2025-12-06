# Quick Start

Get up and running with PyTrendy's trend detection in just a few steps.

---

## Input Requirements

PyTrendy expects a `pandas.DataFrame` with two columns:

<table style="border-collapse: collapse; width: 100%; font-size: 14px;">
  <thead>
    <tr style="border-bottom: 1px solid #ccc;">
      <th style="padding: 8px; text-align: left;">Column</th>
      <th style="padding: 8px; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">date_col</td>
      <td style="padding: 8px;">Currently only dates are supported (daily data).</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">value_col</td>
      <td style="padding: 8px;">Primary time series signal.</td>
    </tr>
  </tbody>
</table>

!!! warning "Data Format Requirements"
    - **Date column** must be in `datetime` format. Use `pd.to_datetime(df['date'])` if needed.
    - **Value column** must be numeric (int or float). Remove NaN values before processing.
    - **Daily frequency**: PyTrendy is optimized for daily time series. Higher frequencies may require resampling.

---

## Basic Usage

Run the full pipeline with a single function call:

```python
from pytrendy import detect_trends

# Run full pipeline on your time series DataFrame
results = detect_trends(
    df,
    date_col="date",                # Column containing datetime values
    value_col="signal",             # Column with signal values
    plot=True,                      # Enable visualization
    method_params={                 # Optional method-specific parameters
        "is_abrupt_padded": True,   # Pads abrupt trends for quasi-experimental use-cases
        "abrupt_padding": 28         # Default 28. Controls padding duration after abrupt change
    }
)
```

This executes all 5 stages and returns a `PyTrendyResults` object with structured access to trends and metrics.

---

## Next Steps

- **[Working with Results](working-with-results.md)** - Learn how to access and filter results
- **[Visualization](visualization.md)** - Generate plots of your trends
- **[Getting Started Tutorial](../tutorials/getting-started.md)** - Complete walkthrough with examples
