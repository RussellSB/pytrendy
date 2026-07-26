This folder contains test data that was found to break the logic of the algorithm in new and interesting ways.
Some of this came from random noise generation for quick manual testing, some from synthetic 'hand drawn' data.
As such, they were backed up here and immortalised for a never ending cycle of automated testing.

- noisy_crashes.csv: edge cases that were found to crash pytrendy (either by hanging up, or throwing execution errors)
- noisy_edgecases.csv: would make trends get detected in weird unexpected way, or plot visualisation bugs, etc...
- low_value_series.csv: a hand drawn series that is in the domain [0, 1], which was found to break the algorithm, leaving valid trends undetected as flats instead.
- zero_baseline_edgecases_1.csv: zero-baseline market entry series (new-market launch scenario) that broke abrupt padding — the entire series collapsed to Flat when `abrupt_padding` was set (fixed in v1.2.4, PR #142). Column: `zero_baseline_market_entry_1`.
- zero_baseline_edgecases_2.csv: zero-baseline market entry series with two ramp sizes for issue #163 (false Noise segments at leading edge of abrupt transitions). Columns: `zero_baseline_market_entry_2` (20→195 ramp), `zero_baseline_market_entry_3` (10→125 ramp), `zero_baseline_spikes` (scattered spikes on zero baseline: 400, 500, 200, 700, 700, 50).
- gradual_ramp_edgecases.csv: synthetic 90-day gradual ramp from 1000 to 1900, then flat at 1900, then drop to 500 (issue #195). Column: `gradual_ramp_90d`. Tests that long gradual uptrends are detected as a single Up segment rather than being truncated by false flat detection.
