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

**PyTrendy** is a modular Python toolkit for detecting, refining, and analyzing trend segments in time series data. Designed for developers and analysts working with noisy signals, PyTrendy offers a robust pipeline that combines statistical preprocessing, dynamic segmentation, and DTW-based classification to extract meaningful patterns from complex datasets.

Whether you're analyzing financial indicators, sensor outputs, or behavioral metrics, PyTrendy helps you surface directional trends, classify their nature (gradual vs abrupt), and quantify their steepness and duration - all with developer-friendly access and extensibility.

---

## Features

![Gradual Trends](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Gradual-Cropped.gif)
![Abrupt Trends](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Abrupt-Cropped.gif)
![Noise Spikes](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Noise-Spikes-Cropped.gif)
![Random Noise](https://raw.githubusercontent.com/RussellSB/pytrendy/refs/heads/main/plots/Noise-Random-Cropped.gif)

---

## Next Steps

For a complete guide on using PyTrendy, refer to the [User Guide](user-guide/index.md).

For detailed function documentation, see the [API Reference](reference/pytrendy/index.md).