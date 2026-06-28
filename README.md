<div align="center">
  <img src="https://raw.githubusercontent.com/RussellSB/pytrendy/3bea91f34bfa8d5452332e5f59f3e2bdf1e3806c/plots/logo.svg" alt="PyTrendy Logo" width="250" />
  <br>
  <h1>PyTrendy</h1>

  [![PyPI version](https://img.shields.io/pypi/v/pytrendy.svg)](https://pypi.org/project/pytrendy/)
  [![Python](https://img.shields.io/badge/python-%3E%3D%203.10-blue.svg)](https://pypi.org/project/pytrendy/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  <br>
  [![Tests](https://github.com/RussellSB/pytrendy/actions/workflows/test.yaml/badge.svg)](https://github.com/RussellSB/pytrendy/actions/workflows/test.yaml)
  [![Release](https://github.com/RussellSB/pytrendy/actions/workflows/release.yaml/badge.svg)](https://github.com/RussellSB/pytrendy/actions/workflows/release.yaml)
  <br>
  [![codecov](https://codecov.io/gh/RussellSB/pytrendy/branch/main/graph/badge.svg)](https://codecov.io/gh/RussellSB/pytrendy)
  [![Downloads](https://static.pepy.tech/badge/pytrendy)](https://pepy.tech/project/pytrendy)
</div>

PyTrendy is a robust solution for identifying and analysing trends in time series. Unlike other trend detection packages, it is robust to noisy and flat segments, and handles gradual and abrupt trend cases with high precision. It aims to be the best package for trend detection in Python.

**Read more in the documentation:** [russellsb.github.io/pytrendy/main](https://russellsb.github.io/pytrendy/main/)

## Why PyTrendy?

Most time series tools give you either a "trend component" (via decomposition) or "changepoints" (the moments of shift). PyTrendy is built for **labelled segment analysis**, answering *what trends existed, how strong were they, and when did they start and end?*

- **Beyond step changes** - `ruptures` is the gold standard for abrupt shifts, but it doesn't handle gradual slope changes (digital marketing, stocks, energy). PyTrendy detects both in a single run.
- **The flat/noise problem** - closest peers (`pytrendseries`, `trendet`, `tstrends`) over-fit trends on flat or noisy periods. PyTrendy's signal-processing and post-processing logic ensures trends are only detected when they are precise and valid.
- **Strategic value** - where dozens of time series interact, knowing how they align or confound at specific times is invaluable for experiment design.

## Features

![](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Gradual-Cropped.gif)
![](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Abrupt-Cropped.gif)
![](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Noise-Spikes-Cropped.gif)
![](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Noise-Random-Cropped.gif)

## Quickstart

Install the package from PyPi.
```
pip install pytrendy
```

Import pytrendy, and apply trend detection on daily time series data.
```py
import pytrendy as pt
df = pt.load_data('series_synthetic')
results = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True)
results.print_summary()
```
![](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/pytrendy-gradual.png)
```md
Detected: 
- 3 Uptrends. 
- 3 Downtrends.
- 3 Flats.
- 0 Noise.

The best detected trend is Down between dates 2025-05-09 - 2025-06-17

Full Results:
-------------------------------------------------------------------------------
            direction       start         end  days  total_change  change_rank trend_class
time_index                                                                               
1                 Up  2025-01-02  2025-01-24    22     14.013348            5     gradual
2               Down  2025-01-25  2025-02-05    11    -13.564214            6     gradual
3               Flat  2025-02-06  2025-02-09     3     -1.168831            9         NaN
4                 Up  2025-02-10  2025-03-14    32     24.632035            3     gradual
5               Flat  2025-03-15  2025-03-17     2      5.660173            7         NaN
6               Down  2025-03-18  2025-04-01    14    -22.721861            4     gradual
7                 Up  2025-04-02  2025-05-08    36     72.611833            2     gradual
8               Down  2025-05-09  2025-06-17    39    -73.253968            1     gradual
9               Flat  2025-06-18  2025-06-30    12      3.910534            8     NaN 
-------------------------------------------------------------------------------
```

Explore the strongest uptrends:
```py
results.filter_segments(direction='Up', sort_by='change_rank')[:3]
```

| time_index | direction | start | end | trend_class | change | pct_change | days | total_change | SNR | change_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | Up | 2025-04-02 | 2025-05-08 | gradual | 72.61 | 367.50% | 36 | 72.61 | 21.70 | 2 |
| 4 | Up | 2025-02-10 | 2025-03-14 | gradual | 24.63 | 169.22% | 32 | 24.63 | 18.87 | 3 |
| 1 | Up | 2025-01-02 | 2025-01-24 | gradual | 14.01 | 104.41% | 22 | 14.01 | 22.21 | 5 |

`filter_segments` ranks segments by magnitude (`change_rank`). See the [API reference](https://russellsb.github.io/pytrendy/main/reference/pytrendy/io/results_pytrendy/#pytrendy.io.results_pytrendy.PyTrendyResults.filter_segments) for all filter and sort options.

For the full per-segment metrics table, use `results.df`.

For more examples on interpreting the results, see [Detect Gradual Trends](https://russellsb.github.io/pytrendy/main/examples/fundamentals/gradual/).

