This folder contains test data that was found to break the logic of the algorithm in new and interesting ways.
Some of this came from random noise generation for quick manual testing, some from synthetic 'hand drawn' data.
As such, they were backed up here and immortalised for a never ending cycle of automated testing.

- noisy_crashes.csv: edge cases that were found to crash pytrendy (either by hanging up, or throwing execution errors)
- noisy_edgecases.csv: would make trends get detected in weird unexpected way, or plot visualisation bugs, etc...
- low_value_series.csv: a hand drawn series that is in the domain [0, 1], which was found to break the algorithm, leaving valid trends undetected as flats instead.
- zero_baseline_edgecases.csv: a zero-baseline market entry series (new-market launch scenario) that broke abrupt padding — the entire series collapsed to Flat when `abrupt_padding` was set (fixed in v1.2.4, PR #142).
