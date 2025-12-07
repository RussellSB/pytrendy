<p align="center"><img src="https://raw.githubusercontent.com/RussellSB/pytrendy/3bea91f34bfa8d5452332e5f59f3e2bdf1e3806c/plots/logo.svg" width="250"></p>
<h1 align="center">PyTrendy</h1>

[![PyPI version](https://img.shields.io/pypi/v/pytrendy.svg)](https://pypi.org/project/pytrendy/)
[![Python](https://img.shields.io/badge/python-%3E%3D%203.10-blue.svg)](https://pypi.org/project/pytrendy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/RussellSB/pytrendy/actions/workflows/test.yaml/badge.svg)](https://github.com/RussellSB/pytrendy/actions/workflows/test.yaml)
[![Release](https://github.com/RussellSB/pytrendy/actions/workflows/release.yaml/badge.svg)](https://github.com/RussellSB/pytrendy/actions/workflows/release.yaml)
[![codecov](https://codecov.io/gh/RussellSB/pytrendy/branch/main/graph/badge.svg)](https://codecov.io/gh/RussellSB/pytrendy)
[![Downloads](https://static.pepy.tech/badge/pytrendy)](https://pepy.tech/project/pytrendy)

---

## Welcome

PyTrendy is a robust solution for identifying and analyzing trends in time series. Unlike other packages, it  detects uptrends and downtrends in a way that they are not falsely detected over flat and noise segments. 

It is a thoughtful algorithm with a focus on signal processing and a considerable amount of post-processing for high precision at a daily level. It aims to be the best package for trend detection in Python. 

---

## Features

![Gradual Trends](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Gradual-Cropped.gif)
![Abrupt Trends](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Abrupt-Cropped.gif)
![Noise Spikes](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Noise-Spikes-Cropped.gif)
![Random Noise](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Noise-Random-Cropped.gif)

---

## Why PyTrendy?

Though trend detection can be used for several use cases, this package's intended purpose is for identifying different phases of digital marketing at scale.
 
- By applying it to digital marketing spend by day (treatment), it can identify valid treatment (uptrends/downtrends) & placebo (flat) periods for observational causal inference. 
- By applying to the response of an experiment design, it can also be used to identify periods of noise (such as sales promotions) that could greatly mislead indications.

---

## Next Steps

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Set up in 5 minutes__

    ---

    Install [`pytrendy`](#) with [`pip`](#) and get up
    and running in minutes.

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-cog-outline:{ .lg .middle } __Further notes on usage__

    ---

    Refer to a high-level reference on configuration and utilities.

    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

-   :material-notebook:{ .lg .middle } __Learn practically__

    ---

    Learn how to make the most out of PyTrendy through practical tutorials.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Refer to the API, covering information on all functions and parameters.

    [:octicons-arrow-right-24: API Reference](reference/pytrendy/index.md)

</div>