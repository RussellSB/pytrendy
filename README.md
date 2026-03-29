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

PyTrendy is a robust solution for identifying and analyzing trends in time series. Unlike other trend detection packages, it is robust to noisy & flat segments, and handles for gradual & abrupt trend cases with a high precision. It aims to be the best package for trend detection in python.

## Features

![](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Gradual-Cropped.gif)
![](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Abrupt-Cropped.gif)
![](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Noise-Spikes-Cropped.gif)
![](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Noise-Random-Cropped.gif)

## Quickstart

Import pytrendy, and apply trend detection on daily time series data.
```py
import pytrendy as pt
df = pt.load_data('series_synthetic')
results = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True)
results.df
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
            direction       start         end  days  total_change  change_rank
time_index                                                                   
1                 Up  2025-01-02  2025-01-24    22     14.013348            5
2               Down  2025-01-25  2025-02-05    11    -13.564214            6
3               Flat  2025-02-06  2025-02-09     3           NaN            7
4                 Up  2025-02-10  2025-03-14    32     24.632035            3
5               Flat  2025-03-15  2025-03-17     2           NaN            8
6               Down  2025-03-18  2025-04-01    14    -22.721861            4
7                 Up  2025-04-02  2025-05-08    36     72.611833            2
8               Down  2025-05-09  2025-06-17    39    -73.253968            1
9               Flat  2025-06-18  2025-06-30    12           NaN            9 
-------------------------------------------------------------------------------
```