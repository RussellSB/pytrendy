---
name: test
description: Use when running, writing, or verifying pytrendy's test suite. Also covers pytest configuration and install/verify commands.
---

# Testing in pytrendy

## pytest is pre-configured

`pyproject.toml` `[tool.pytest.ini_options]` sets `addopts = "--mpl --cov=pytrendy --cov-report=term --cov-report=xml:coverage.xml"` and `timeout = 15`. You don't pass these flags; `pytest` alone runs the full suite with mpl comparison + coverage.

## Markers

| Marker | Meaning |
|---|---|
| `@pytest.mark.core` | Essential — must pass for core functionality. CI runs these first. |
| `@pytest.mark.plot` | Visualization/plotting tests (always paired with `@pytest.mark.mpl_image_compare`). |

CI ordering (`test.yaml`): `pytest tests/ -m core` then `pytest tests/ -m "not core" --cov-append`. The `--cov-append` is load-bearing — the second run appends to the `.coverage` from the first; running both with fresh coverage resets state. Mirror this locally if you care about combined coverage.

Run a single marker group: `pytest tests/ -m core`. Run one file: `pytest tests/test_io_results.py`. Run one test: `pytest tests/test_io_results.py::TestClassName::test_method`.

## pytest-mpl baselines (the big gotcha)

Plot tests use `@pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_*.png', style='default')`. Baselines live **next to the test file**, not in a shared `baseline/` dir:

- `tests/tests_plotting/core/*.png`
- `tests/tests_plotting/edgecases/*.png`

`baseline_dir='./'` resolves relative to the test file, so the PNG must be in the same directory as the `.py`.

**matplotlib is pinned to 3.10.8** in `pyproject.toml`. Baselines are only valid for that version — different matplotlib renders differently and tests fail on pixel diff. Don't regenerate on another version.

Regenerate baselines after an *intentional* plot change:

```bash
pytest tests/tests_plotting/core/test_plot_pytrendy_core.py --mpl-generate-path=tests/tests_plotting/core/
```

Commit the new PNGs. Verify the diff is what you intended — pytest-mpl can't tell a bug from a deliberate restyle.

## conftest helpers (`tests/conftest.py`)

- `assert_segments_match(detected, expected)` — strict: same count, direction, start, end for every segment.
- `assert_segments_in_a_haystack(detected, expected)` — subset: every expected segment must appear in detected, extras allowed. Use when the pipeline legitimately detects more than the test cares about.

Both expect segment dicts with keys `direction` ('Up'/'Down'/'Flat'/'Noise'), `start`, `end`. Expected dates as `'YYYY-MM-DD'` strings.

## Adding a regression test from a debug reproduction

When a bug is fixed and reproduced in `tests/test.py` (see the `debug` skill), migrate it into a formal regression test:

1. **Pick the fixture**: if the `tests/test.py` cell loads from `tests/tests_crashes_edgecases/data/*.csv`, reuse that CSV. If it builds synth data inline, consider persisting it as a CSV in the same `data/` dir for reuse.
2. **Add a test class** to the appropriate file in `tests/tests_crashes_edgecases/test_*.py` (or `tests/tests_noise/` for noise-specific bugs). Mark `@pytest.mark.core` — regression tests for fixed bugs are essential.
3. **Call `detect_trends(...)`** with the same params that reproduced the bug. Assert on the segment list with `assert_segments_match` (strict) or `assert_segments_in_a_haystack` (subset — use when the pipeline legitimately detects more than the test cares about). **Always** use these helpers for segment assertions — do not use raw list comprehensions, set operations, or manual counting, as those patterns miss segment boundaries and drift out of sync with the rest of the test suite.
4. **Leave the `tests/test.py` reproduction cell in place** — it's not deleted when the formal test is added.

## Test layout

- `tests/test_*.py` — core API, I/O, data loader, debug, method params.
- `tests/test.py` (no underscore suffix) is **not** pytest-collected — it's the interactive debug sandbox; see the `debug` skill.
- `tests/tests_noise/` — noise-handling (spikes abrupt/gradual, random, avoid-false).
- `tests/tests_crashes_edgecases/` — regression cases; fixtures in `tests/tests_crashes_edgecases/data/`.
- `tests/tests_plotting/{core,edgecases}/` — mpl baseline tests + their PNGs.

Edge-case fixtures (e.g. `zero_baseline_edgecases.csv`) back real bug reports; reuse them when reproducing, don't invent synthetic data.

## Test CSV data files

Two distinct locations:

- **`pytrendy/io/data/`** — packaged with the library, accessed via `load_data('series_synthetic' | 'classes_signals')`. Ships to users; used by README quickstart, example notebooks, and core plot tests.
- **`tests/tests_crashes_edgecases/data/`** — test-only fixtures, accessed via `pd.read_csv('tests/tests_crashes_edgecases/data/<file>.csv')` with **relative paths from the repo root** (run pytest from repo root or paths break). `TESTDATA.md` in that dir documents each file's provenance:
  - `noisy_crashes.csv` — edge cases that crashed pytrendy (hangups, execution errors).
  - `noisy_edgecases.csv` — weird detection / visualization bugs.
  - `low_value_series.csv` — hand-drawn [0,1] series that broke the algorithm.
  - `zero_baseline_edgecases.csv` — zero-baseline market entry (new-market launch scenario; broke abrupt padding, fixed in v1.2.4 / PR #142).

**Convention**: one CSV holds many scenario columns (e.g. `noisy_crash`, `noisy_crash_2`, ..., `noisy_edgecase_1`, ..., `noisy_edgecase_11`). Tests pick a scenario via `value_col='noisy_crash_7'`. When adding a regression test, reuse an existing CSV/column or add a new column to the appropriate file + update `TESTDATA.md`. Don't duplicate data into a new CSV if an existing one fits.
