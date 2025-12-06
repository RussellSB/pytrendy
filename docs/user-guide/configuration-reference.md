# Configuration Reference

Complete parameter reference for all PyTrendy functions and configuration options.

---

## Method Parameters

These parameters can be passed via `method_params` in `detect_trends()`:

<table style="border-collapse: collapse; width: 100%; font-size: 14px;">
  <thead>
    <tr style="border-bottom: 1px solid #ccc;">
      <th style="padding: 8px; text-align: left;">Parameter</th>
      <th style="padding: 8px; text-align: left;">Description</th>
      <th style="padding: 8px; text-align: left;">Default</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;"><code>is_abrupt_padded</code></td>
      <td style="padding: 8px;">Whether to extend abrupt trends with padding</td>
      <td style="padding: 8px;"><code>False</code></td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;"><code>abrupt_padding</code></td>
      <td style="padding: 8px;">Number of days to pad abrupt trends (when enabled)</td>
      <td style="padding: 8px;"><code>28</code></td>
    </tr>
  </tbody>
</table>

---

## Function Parameters

Main function parameters for `detect_trends()`:

<table style="border-collapse: collapse; width: 100%; font-size: 14px;">
  <thead>
    <tr style="border-bottom: 1px solid #ccc;">
      <th style="padding: 8px; text-align: left;">Parameter</th>
      <th style="padding: 8px; text-align: left;">Description</th>
      <th style="padding: 8px; text-align: left;">Required</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;"><code>df</code></td>
      <td style="padding: 8px;">Input DataFrame with time series data</td>
      <td style="padding: 8px;">Yes</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;"><code>date_col</code></td>
      <td style="padding: 8px;">Column name of the datetime field</td>
      <td style="padding: 8px;">Yes</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;"><code>value_col</code></td>
      <td style="padding: 8px;">Column name of the signal to analyze</td>
      <td style="padding: 8px;">Yes</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;"><code>plot</code></td>
      <td style="padding: 8px;">Whether to generate visual output</td>
      <td style="padding: 8px;">No (default: <code>False</code>)</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;"><code>method_params</code></td>
      <td style="padding: 8px;">Dictionary of method-specific parameters</td>
      <td style="padding: 8px;">No (default: <code>{}</code>)</td>
    </tr>
  </tbody>
</table>

---

## Parameter Usage Examples

### Basic Detection (No Configuration)

```python
results = detect_trends(df, date_col='date', value_col='price')
```

### With Visualization

```python
results = detect_trends(df, date_col='date', value_col='price', plot=True)
```

### With Abrupt Padding

```python
results = detect_trends(
    df, 
    date_col='date', 
    value_col='price',
    method_params={
        'is_abrupt_padded': True,
        'abrupt_padding': 28  # Extend abrupt trends by 28 days
    }
)
```

### Custom Padding Duration

```python
results = detect_trends(
    df, 
    date_col='date', 
    value_col='price',
    method_params={
        'is_abrupt_padded': True,
        'abrupt_padding': 14  # Shorter padding for rapid changes
    }
)
```

---

## Abrupt Padding Explained

**`is_abrupt_padded`**: When `True`, extends abrupt trend segments to capture post-event stabilization periods.

**`abrupt_padding`**: Controls how many days to extend the trend after the abrupt change point.

**Use cases:**
- Intervention studies (measure sustained effects)
- Policy analysis (capture post-policy stabilization)
- Event studies (measure aftermath periods)
- A/B testing (include post-treatment windows)

**Example:**
- Without padding: Abrupt downtrend detected for 3 days
- With 28-day padding: Same downtrend extended to 31 days total

---

## Next Steps

- **[Quick Start](quick-start.md)** - Start using PyTrendy
- **[Advanced Usage](advanced-usage.md)** - Custom pipeline configuration
- **[API Reference](../reference/pytrendy/index.md)** - Complete function documentation
