# Advanced Usage

For more control over the trend detection pipeline, you can run each stage manually using lower-level functions.

---

## Custom Pipeline

Run each stage step-by-step:

```python
from pytrendy import (
    process_signals,
    get_segments,
    refine_segments,
    analyse_segments,
    PyTrendyResults
)

# Step 1: Preprocess signal
df = process_signals(df, value_col="signal")

# Step 2: Segment detection
segments_raw = get_segments(df)

# Step 3: Refinement and classification
method_params = {"is_abrupt_padded": True, "abrupt_padding": 28}
segments_refined = refine_segments(
    df, 
    time_col="date", 
    value_col="signal", 
    segments=segments_raw, 
    method_params=method_params
)

# Step 4: Metric analysis and ranking
segments_final = analyse_segments(
    df, 
    time_col="date", 
    value_col="signal", 
    segments=segments_refined
)

# Step 5: Wrap results
results = PyTrendyResults(segments_final)
```

This approach is ideal for:
- Experimentation with individual stages
- Debugging specific pipeline steps
- Integrating PyTrendy into larger workflows
- Custom preprocessing or postprocessing

---

## When to Use Custom Pipelines

**Use the high-level `detect_trends()` API when:**
- You want quick results with sensible defaults
- Your data follows standard patterns
- You need minimal configuration

**Use the custom pipeline when:**
- You need to inspect intermediate results
- You want to modify signal processing parameters
- You're integrating with existing data pipelines
- You need fine-grained control over each stage

---

## Next Steps

- **[Configuration Reference](configuration-reference.md)** - Full parameter documentation
- **[Custom Pipeline Tutorial](../tutorials/custom-pipeline.md)** - Step-by-step walkthrough
- **[API Reference](../reference/pytrendy/index.md)** - Complete function documentation
