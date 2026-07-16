---
name: maintenance
description: Use when modifying pytrendy's public API, deprecating or removing parameters, renaming internal modules, or touching core pipeline behavior. Covers the API surface, deprecation policy, and diff conventions that preserve the package's design principles (issue #141).
---

# Maintenance & API evolution

This is the consolidated context issue #141 asked for — the design principles agents keep re-litigating on PRs.

## Public API surface

The public API is **what `pytrendy/__init__.py` exports** plus the `detect_trends()` signature:

```python
from pytrendy import detect_trends, load_data, plot_pytrendy, dtw
```

- `detect_trends(df, date_col, value_col, plot=True, method_params=None, debug=False)` → `PyTrendyResults`
- `load_data(name)` — `name` ∈ `{'series_synthetic', 'classes_signals'}` (CSVs in `pytrendy/io/data/`)
- `plot_pytrendy(...)` — annotated visualization
- `dtw` — re-exported from `pytrendy.simpledtw`

`PyTrendyResults` (from `pytrendy.io.results_pytrendy`) is the return type: `.print_summary()`, filtering by direction/rank, tabular segment access.

**Changing any of these is user-observable.** Internal helpers in `post_processing/`, `process_signals.py`, `simpledtw.py` are not public API — restructure freely with `refactor:`.

## Deprecation policy (the rule agents get wrong)

| Situation | Commit type | Version bump | Mechanism |
|---|---|---|---|
| Parameter still works but deprecated (soft) | `feat:` | minor | `warnings.warn(..., DeprecationWarning)` + keep accepting the old name |
| Parameter/behavior removed entirely (hard) | `feat!:` or `BREAKING CHANGE:` | major | Remove the old code path |
| Internal restructure, zero public API impact | `refactor:` | none | No warning needed |

**Deprecating a public param is `feat:`, NOT `refactor:`** — even if the code change is a rename. Deprecation is a user-facing signal that triggers a minor bump and documents intent to remove in a future major. See `docs/contributing.md` "Deprecations" and `.github/copilot-instructions.md` (now migrated here).

### Current live deprecation

`detect_trends(..., method_params=...)`: `is_abrupt_padded` is **deprecated** (raises `DeprecationWarning`, see `pytrendy/detect_trends.py:63`). Use `abrupt_padding` instead.

### Current `method_params` keys (the only ones honored)

```python
method_params = {
    'abrupt_padding': 0,      # int: days to pad around abrupt transitions
    'avoid_noise': True,      # bool: skip noisy segments
}
```

Any other keys passed are silently dropped — `detect_trends` reconstructs the dict from these two defaults (`pytrendy/detect_trends.py:72`). Don't add a third key without also updating this allowlist and the docstring.

## Code conventions

- **Docstrings**: Google style (mkdocstrings is configured for it in `mkdocs.yml`). Required on public functions.
- **Type hints** on all public function signatures.
- **No debug prints or commented-out code** in committed code. Use `debug=True` in `detect_trends` for dev plots/prints — that's what it's for.
- **Keep diffs minimal**: only touch lines directly relevant to the change. No stray reformatting, whitespace, blank-line, or indentation changes. These wreck reviews and cause merge conflicts. If something nearby is ugly, open a separate `refactor:` PR.
- Follow existing patterns in the module you're editing — don't import a new library when the module already uses something equivalent.

## Pipeline (don't break the order)

`detect_trends()` runs five stages in `pytrendy/detect_trends.py:77`:

1. `process_signals(df, value_col, method_params, debug)` — Savitzky-Golay smoothing, flat/noise flags.
2. `get_segments(df)` — contiguous segment extraction with min-length: Up/Down ≥3d, Flat/Noise ≥1d.
3. `refine_segments(df, value_col, segments, method_params)` — boundary adjust, DTW gradual/abrupt classification, abrupt shaving, grouping, artifact cleanup.
4. `analyse_segments(df, value_col, segments)` — metrics: total/percent change, duration, SNR, change rank.
5. `plot_pytrendy(df, value_col, segments)` — only if `plot=True`.

Each stage's output is the next stage's input. See the `pytrendy` skill for the module map.

## Segment schema (what tests assume)

Segment dicts have at minimum: `direction` ('Up'/'Down'/'Flat'/'Noise'), `start`, `end`. After `analyse_segments`, also: `days`, `total_change`, `change_rank`, `trend_class` ('gradual'/'abrupt'/NaN). Tests in `tests/conftest.py` assert on `direction`/`start`/`end` — changing the keys or values breaks the whole suite.
