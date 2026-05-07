<p align="center"><img src="https://raw.githubusercontent.com/RussellSB/pytrendy/3bea91f34bfa8d5452332e5f59f3e2bdf1e3806c/plots/logo.svg" width="250"></p>
<h1 align="center">PyTrendy</h1>

[![PyPI version](https://img.shields.io/pypi/v/pytrendy.svg)](https://pypi.org/project/pytrendy/)
[![Python](https://img.shields.io/badge/python-%3E%3D%203.10-blue.svg)](https://pypi.org/project/pytrendy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/RussellSB/pytrendy/actions/workflows/test.yaml/badge.svg)](https://github.com/RussellSB/pytrendy/actions/workflows/test.yaml)
[![Release](https://github.com/RussellSB/pytrendy/actions/workflows/release.yaml/badge.svg)](https://github.com/RussellSB/pytrendy/actions/workflows/release.yaml)
[![codecov](https://codecov.io/gh/RussellSB/pytrendy/branch/develop/graph/badge.svg)](https://codecov.io/gh/RussellSB/pytrendy)
[![Downloads](https://static.pepy.tech/badge/pytrendy)](https://pepy.tech/project/pytrendy)

---

## Welcome

PyTrendy is a robust solution for identifying and analyzing trends in time series. Unlike other packages, it detects uptrends and downtrends in a way that they are not falsely detected over periods of flat or noise segments. 

It is a thoughtful algorithm with a focus on signal processing and post-processing. It aims to be the best package for trend detection in Python. 

---

## Why PyTrendy?

Trend detection has several use cases, such as analysing stock prices for investing, identifying demand trends in seasonality patterns to optimise inventory management, analysing google trends at scale for emerging movements in industries, and more. 

However, one main use case is for identifying different periods of marketing activity at scale - to help with observationally measuring the effectiveness of digital marketing.
 
- By applying it to digital marketing spend by day (treatment), it can identify valid treatment (uptrends/downtrends) & placebo (flat) periods for observational causal inference. 
- By applying to the response of an experiment design, it can also be used to identify periods of noise (such as sales promotions) to mitigate the risks of misleading indications.

---

## Features

![Gradual Trends](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Gradual-Cropped.gif)
![Abrupt Trends](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Abrupt-Cropped.gif)
![Noise Spikes](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Noise-Spikes-Cropped.gif)
![Random Noise](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/Noise-Random-Cropped.gif)

---

## Installation

Install the package from PyPi.
```bash
pip install pytrendy
```

Alternatively, if you want the latest pre-release
```bash
pip install --pre pytrendy
```

---

## Quickstart

Import pytrendy, and apply trend detection on daily time series data.
```py
import pytrendy as pt
df = pt.load_data('series_synthetic')
results = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True)
results.print_summary()
```
![](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/develop/plots/pytrendy-gradual.png)
<div class='transparent'>
```
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
9               Flat  2025-06-18  2025-06-30    12      3.910534            8         NaN 
-------------------------------------------------------------------------------
```
</div>

More information on how you can interpret the trend detection results are available in the [Example Gallery](examples/index.md).

</br>
</br>
