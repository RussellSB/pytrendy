# Visualization

Generate publication-ready plots with automatic segment highlighting and trend annotations.

---

## Plotting Trends

Visualize detected segments with color-coded regions:

```python
from pytrendy import plot_pytrendy

plot_pytrendy(df, time_col="date", value_col="signal", segments_enhanced=segments_final)
```

This generates a time series plot with:
- Original signal plotted as a line
- Shaded regions for each segment type
- Annotations for top-ranked trends
- Legend showing segment types

---

## Segment Colors

Plots include shaded regions for each segment type:

<table style="border-collapse: collapse; width: 100%; max-width: 600px; font-size: 14px;">
  <thead style="background-color: #f5f5f5;">
    <tr style="border-bottom: 1px solid #ccc;">
      <th style="padding: 8px; text-align: left;">Segment Type</th>
      <th style="padding: 8px; text-align: left;">Color</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">Uptrend</td>
      <td style="padding: 8px; background-color: LightGreen;">Light Green</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">Downtrend</td>
      <td style="padding: 8px; background-color: LightCoral;">Light Coral</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">Flat</td>
      <td style="padding: 8px; background-color: LightBlue;">Light Blue</td>
    </tr>
    <tr style="border-bottom: 1px solid #eee;">
      <td style="padding: 8px;">Noise</td>
      <td style="padding: 8px; background-color: LightGray;">Light Gray</td>
    </tr>
  </tbody>
</table>

Top-ranked trends are annotated directly on the chart with their rank number and metadata.

---

## Plot Customization

The plotting function uses matplotlib under the hood, so you can customize the output:

```python
import matplotlib.pyplot as plt

# Create custom figure
fig, ax = plt.subplots(figsize=(14, 6))

# Generate plot
plot_pytrendy(df, time_col="date", value_col="signal", segments_enhanced=segments_final)

# Additional customization
plt.title("My Custom Title", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Signal Value", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save or display
plt.savefig("trend_analysis.png", dpi=300)
plt.show()
```

---

## Next Steps

- **[Working with Results](working-with-results.md)** - Filter and analyze detected trends
- **[Configuration Reference](configuration-reference.md)** - Customize detection parameters
- **[Real-World Examples](../tutorials/real-world-examples.md)** - See visualization in action
