---
name: pytrendy
description: Use when modifying the trend detection pipeline, adding/renaming modules under pytrendy/, or working with datasets and results. Covers the 5-stage pipeline, module boundaries, built-in data, and the PyTrendyResults API.
---

# pytrendy package architecture

## 5-stage pipeline

`pytrendy/detect_trends.py` — the only entrypoint that matters. `detect_trends(df, date_col, value_col, plot=True, method_params=None, debug=False)` runs:

```
df → process_signals → get_segments → refine_segments → analyse_segments → [plot_pytrendy] → PyTrendyResults
```

Stages mutate a copy of the input DataFrame and a list of segment dicts. The final `PyTrendyResults` wraps the segment list.

## Module map

```
pytrendy/
├── __init__.py            # public exports: detect_trends, load_data, plot_pytrendy, dtw
├── detect_trends.py       # pipeline orchestrator (85 lines — read it first)
├── process_signals.py     # Savitzky-Golay smoothing + rolling stats → flat/noise/trend flags
├── simpledtw.py           # DTW for gradual-vs-abrupt classification (standalone, no deps on rest)
├── io/
│   ├── data_loader.py     # load_data('series_synthetic' | 'classes_signals')
│   ├── plot_pytrendy.py   # annotated matplotlib viz; called by detect_trends when plot=True
│   ├── results_pytrendy.py # PyTrendyResults: .print_summary(), filtering, ranking, tabular access
│   └── data/              # series_synthetic.csv, classes_signals.csv (packaged CSVs)
└── post_processing/
    ├── segments_get.py        # extract contiguous segments from trend_flag; min-length filter
    ├── segments_refine/       # package — the heaviest stage
    │   ├── __init__.py        # refine_segments() orchestrator
    │   ├── gradual_expand_contract.py  # boundary adjust via local extrema (±7d window)
    │   ├── trend_classify.py          # DTW-based gradual/abrupt labeling
    │   ├── abrupt_shaving.py          # z-score outlier changepoint detection on abrupt segs
    │   ├── segment_grouping.py        # merge short consecutive same-direction segments
    │   ├── artifact_cleanup.py        # remove invalid segments, fill gaps with flats
    │   └── update_neighbours.py       # boundary cascade when a neighbor changes
    └── segments_analyse.py    # per-segment metrics: change, duration, SNR, change_rank
```

## Datasets

`load_data(name)` reads CSVs shipped in `pytrendy/io/data/`:

- `'series_synthetic'` — synthetic daily time series with embedded up/down/flat regions. The README quickstart and core plot tests use this.
- `'classes_signals'` — reference signals used internally by `trend_classify` for DTW alignment (gradual vs abrupt templates).

Adding a dataset = drop a CSV in `pytrendy/io/data/` + extend `data_loader.py`. It's a plain `pd.read_csv` wrapper, not a registry.

## PyTrendyResults (`io/results_pytrendy.py`)

Returned by `detect_trends()`. Constructed from the segment list and auto-populates `best`, `df`, `df_summary`, `summary` in `__init__`.

### Attributes

| Attr | Type | What it is |
|---|---|---|
| `.segments` | `list[dict]` | Raw segment dicts (all directions including Flat/Noise). |
| `.trend_segments` | `list[dict]` | Subset with a `trend_class` key (Up/Down only — Flats/Noise excluded). |
| `.best` | `dict \| None` | Segment with lowest `change_rank` among `trend_segments`; `None` if no trends. Picked on `total_change` magnitude (steepness × length), not raw `change`. |
| `.df` | `pd.DataFrame` | Full segment table, indexed by `time_index`. All columns: `direction, start, end, trend_class, change, pct_change, days, total_change, SNR, change_rank` (+ `padded` when `abrupt_padding>0`). |
| `.df_summary` | `pd.DataFrame` | Slimmer view: `direction, start, end, days, total_change, change_rank, trend_class`. What `.print_summary()` prints. |
| `.summary` | `dict` | `direction_counts` (Counter→dict over Up/Down/Flat/Noise), `trend_class_counts` (gradual/abrupt), `highest_total_change`. |

### Methods

- `.print_summary()` — prints direction counts, best trend (direction + date range), and the `df_summary` table. **Use for high-level overview**; cheap to paste into an LLM context when developing.
- `.filter_segments(direction='Any', sort_by='time_index', format='df')` — the workhorse for narrowing focus.
  - `direction`: `'Any'` | `'Up/Down'` | `'Up'` | `'Down'` | `'Flat'` | `'Noise'`.
  - `sort_by`: `'time_index'` (ascending) or `'change_rank'` (descending by `abs(total_change)`).
  - `format`: `'df'` (default) or `'dict'`.
  - Returns DataFrame/list; slice with `[:3]` for top-N. Invalid `direction`/`sort_by`/`format` print a hint and fall through.

### Usage patterns (from the example notebooks)

- **High-level overview** → `results.print_summary()`. One-glance counts + best trend.
- **Debugging one direction** → `results.filter_segments(direction='Up')` to isolate uptrends only (the abrupt notebook does this to compare uptrends against each other).
- **Rank the strongest trends** → `results.filter_segments(direction='Up/Down', sort_by='change_rank')`. Rank 1 = steepest+longest by `total_change` magnitude.
- **Top-N** → `results.filter_segments(direction='Up', sort_by='change_rank')[:3]`.
- **Full per-segment metrics** → `results.df` (includes `change`, `pct_change`, `SNR` that `df_summary` omits).
- **Raw dicts for downstream code** → `results.filter_segments(format='dict')` or `results.segments` / `results.trend_segments`.
- **Best single segment** → `results.best` (dict).

### Metric columns (what they mean)

- `change` / `total_change` — cumulative sum of day-over-day differences across the segment.
- `pct_change` — relative change; e.g. `3.67` = +367%. Useful for comparing trends of different baselines.
- `days` — segment length.
- `SNR` — signal-to-noise ratio. **Lower SNR ≈ noisier segment.** Noise segments typically have SNR < ~7; clean gradual trends sit around 17-22. Borderline trends (e.g. SNR ~5.9) show up with high `change_rank` numbers — still detected but flagged as weak.
- `change_rank` — 1 = strongest trend by `abs(total_change)`. Lower is stronger. Artifact trends (short, low-magnitude) get high rank numbers and are effectively de-prioritized.
- `padded` — `True` only when `abrupt_padding>0` extended an abrupt segment's end date; `NaN` for gradual/flat/noise.
- `trend_class` — `gradual` / `abrupt` / `NaN` (NaN for Flat/Noise).

### Use-case notes (from the notebooks)

- **Gradual trends** — the default case; `trend_class='gradual'` for all Up/Down. Compare via `filter_segments(direction='Up', sort_by='change_rank')` to rank competing uptrends.
- **Abrupt trends** — short, sharp shifts (changepoint-style). Pass `method_params=dict(abrupt_padding=N)` to extend each abrupt segment N days forward (for Interrupted Time Series / Pre-Post analysis). Padding stops early if the signal reverses; segments gain a `padded=True` column. Combine with gradual in one run — DTW classifies each segment independently, so a single `detect_trends()` call returns both classes.
- **Noise** — `direction='Noise'` segments are flagged by a rolling SNR threshold, not by magnitude. Use `filter_segments(direction='Noise')` to inspect what got rejected. As noise std rises, trends shorten and more Noise/Flat segments appear; borderline trends keep high `change_rank` so they're easy to filter out.
- **Noise control via `avoid_noise`** — `method_params=dict(avoid_noise=False)` opts out of noise detection entirely. Spikes and noisy regions are ignored and trend detection proceeds straight through them. **Use case**: new-market launches or quasi-experiments where the signal is zero before and after an activation window — with `avoid_noise=True` (default), the step-change boundaries produce Noise artifacts; with `avoid_noise=False`, you get clean Up/Down segments. See `whats-new.md` v1.2.0 entry / PR #110.

Tests exercise this heavily in `tests/test_io_results.py` (all `@pytest.mark.core`) — any API change there breaks the core suite immediately. Read `results_pytrendy.py` before extending the class.

## Segment classification

- **direction** ∈ {`Up`, `Down`, `Flat`, `Noise`} — assigned during `process_signals`/`segments_get`.
- **trend_class** ∈ {`gradual`, `abrupt`, `NaN`} — assigned by `segments_refine/trend_classify.py` via DTW against `classes_signals` references. Only Up/Down segments get a class; Flat/Noise stay NaN.
- **Min segment length**: Up/Down ≥3 days, Flat/Noise ≥1 day (enforced in `segments_get.py`).

## `method_params` (the only tuning surface)

```python
{'abrupt_padding': 0, 'avoid_noise': True}  # defaults; see maintenance skill for deprecation note
```

`detect_trends` reconstructs this dict from an allowlist — unknown keys are dropped. Adding a tunable = update the allowlist in `detect_trends.py:72` + the docstring + add a test.
