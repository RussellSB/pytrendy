---
name: debug
description: Use when debugging a pytrendy bug, reproducing a regression, or inspecting intermediate pipeline stage output. Covers the tests/test.py sandbox convention, phase 1 (process_signals debug=True plots), phase 2 (segments_refine stage-by-stage bisect), and handoff to the test skill for formal regression tests.
---

# Debugging in pytrendy

## Scratch sandbox: tests/test.py

`tests/test.py` is the committed reproduction sandbox. Key facts:

- **Not pytest-collected** — filename `test.py` doesn't match the `test_*.py` glob, so `pytest` ignores it. Its module docstring states this explicitly.
- **`# %%` cell markers throughout** → VSCode treats it as an interactive notebook. Run cells inline with Shift+Enter; plots render inline via the interactive backend.
- **Magic setup at top**: `%load_ext autoreload` / `%autoreload 2` — edits to `pytrendy/` reload without restarting the kernel.
- **`# TODONE:` / `# TODO:` inline notes** track fix status per cell — the file's distinctive convention (see line 30, 53, 89, etc. for examples).
- **Existing sections are specific bug reproductions** (synth data + `detect_trends()` call + inspect plot), not generic templates. Agent-added sections must match: one `# %%` cell per bug, specific scenario/data/params.

### When adding a section

- `# ---------- <bug summary>` header (matches existing convention — see line 325 "New Reproduction: abrupt padding all-flat fallback on zero-baseline market entry").
- `# %%` cell loading the specific fixture / building the specific synth scenario.
- `# %%` cell calling `detect_trends(...)` — with `debug=True` for phase-1 bugs, plain for phase-2 bugs.
- `# TODONE:` note on the cell once the bug is fixed (matches the file's tracking convention).
- Cell stays in place after migration to formal tests — not deleted.

### After the fix

Mark the reproduction cell's note `# TODONE:`. Leave the cell in place; it stays useful for re-running after future changes. Then migrate the reproduction into a formal regression test — see the `test` skill ("Adding a regression test from a debug reproduction").

## Phase 1 — process_signals (rolling window flags)

Bugs in flagging propagate everywhere downstream. Diagnose here first.

Pass `debug=True` to `detect_trends()`. `process_signals.py:196` emits 7 diagnostic plots in sequence, each showing the value column on the left y-axis and the metric/flag on the right:

1. **SNR** + `THRESHOLD_NOISE` line (2.5) — noise detection input.
2. **noise_flag** — where SNR fell below threshold.
3. **smoothed** — Savitzky-Golay output (window=15).
4. **smoothed_std** — rolling std (window=7) for flat detection.
5. **flat_flag** — where smoothed_std ≤ min non-zero rolling std.
6. **smoothed_deriv** + `±derivative_limit` lines — trend direction input.
7. **trend_flag** — final per-day classification (1=Up, -1=Down, -2=Flat, -3=Noise).

If a flag is wrong here (e.g. `noise_flag` fires on a legitimate abrupt shift, or `trend_flag` misses an obvious uptrend), the bug is in `process_signals.py`. The knobs: `THRESHOLD_NOISE=2.5`, `THRESHOLD_SMOOTH=0.001`, `WINDOW_SMOOTH=15`, `WINDOW_FLAT=7`. Reproduce with a single series + `debug=True` in a `# %%` cell of `tests/test.py` and iterate.

## Phase 2 — segments_refine (post-processing bisect)

If `process_signals` flags look correct but the final segments are wrong, the bug is in the post-processing chain. `pytrendy/post_processing/segments_refine/__init__.py:refine_segments()` runs ~14 stage calls in a specific order, and bugs compound — a misclassification at `classify_trends` propagates through grouping, shaving, cleanup, and re-classification.

**Stage order from `__init__.py`:**

1. `classify_trends` — DTW gradual/abrupt labels.
2. `group_segments` (1st pass — sporadic flats/noises).
3. `expand_contract_segments` — gradual boundary adjust (±7d extrema).
4. `shave_abrupt_trends` — abrupt changepoint z-score shaving.
5. `clean_artifacts` — remove overlaps from expand/contract.
6. `group_segments` (2nd pass).
7. `clean_artifacts` (again).
8. `classify_trends` (reclassify — some graduals → abrupts).
9. If changed: `shave_abrupt_trends` (2nd pass, `second_pass=True, init_segments=...`).
10. `group_segments` + `clean_artifacts`.
11. `fill_in_flats` — fill gaps with flats.
12. `group_segments` (3rd pass, final) + `clean_artifacts` (`inverse_only=True`).

### Primary bisect — comment in source

Open `refine_segments()` in `pytrendy/post_processing/segments_refine/__init__.py`. Comment out stages one at a time (working from the bottom of the chain upward, or from `classify_trends` onward). After each edit, re-run the `tests/test.py` reproduction cell (Shift+Enter in VSCode) to see the effect on the plot/segments. The first uncomment that reintroduces the wrong behavior isolates the culprit stage.

### Alternative bisect — snapshot helper (exploration only, not persisted)

For faster iteration without touching source, run this loop outside `tests/test.py` (REPL, separate scratch notebook, inline eval). Prints segment state after each stage; the divergence point is the culprit. **Throwaway — never persist this in `tests/test.py`.**

```python
import pytrendy as pt
from pytrendy.post_processing.segments_get import get_segments
from pytrendy.post_processing.segments_refine import (
    classify_trends, group_segments, expand_contract_segments,
    shave_abrupt_trends, clean_artifacts, fill_in_flats,
)

df = pt.load_data('series_synthetic')[['date', 'gradual']].set_index('date')
value_col = 'gradual'
method_params = dict(abrupt_padding=0)

segs = get_segments(df)
print('pre-refine', len(segs), [(s['direction'], str(s['start']), str(s['end'])) for s in segs])

for name, fn in [
    ('classify_trends', lambda s: classify_trends(df, value_col, s)),
    ('group_segments#1', lambda s: group_segments(s)),
    ('expand_contract', lambda s: expand_contract_segments(df, value_col, s)),
    ('shave_abrupt', lambda s: shave_abrupt_trends(df, value_col, s, method_params)),
    ('clean_artifacts#1', lambda s: clean_artifacts(df, value_col, s, method_params)),
    ('group_segments#2', lambda s: group_segments(s)),
    ('clean_artifacts#2', lambda s: clean_artifacts(df, value_col, s, method_params)),
    ('classify_trends#2', lambda s: classify_trends(df, value_col, s)),
    ('fill_in_flats', lambda s: fill_in_flats(df, s)),
    ('group_segments#3', lambda s: group_segments(s)),
    ('clean_artifacts#3', lambda s: clean_artifacts(df, value_col, s, method_params, inverse_only=True)),
]:
    segs = fn(segs)
    print(name, len(segs), [(s['direction'], str(s['start']), str(s['end'])) for s in segs])
```

### Re-classification loop gotcha

Steps 8-10 only fire if `classify_trends` (reclassify) changes anything (`if segments_refined != init_segments` at `__init__.py:52`). If your bug only appears on series with mixed gradual/abrupt, the 2nd-pass `shave_abrupt_trends` with `second_pass=True, init_segments=...` is a likely culprit — it has different behavior than the 1st pass.

## Handoff to formal tests

Once fixed, the `test` skill takes over — migrate the reproduction into `tests/tests_crashes_edgecases/test_*.py` using `assert_segments_match` / `assert_segments_in_a_haystack` from `tests/conftest.py`. The `tests/test.py` cell stays in place.
