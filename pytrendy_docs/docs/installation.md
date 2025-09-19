# Installation

PyTrendy can be installed via `pip` and used immediately for trend detection and time series analysis.


## Installation Requirements and Dependencies

### Python

- **Python version:** 3.8 or newer


### Package Dependencies

The following packages are installed automatically when you use `pip install pytrendy`.

- `pandas`
- `numpy`
- `matplotlib`
- `scipy`


## Standard Installation

To install PyTrendy and all its dependencies, run the following command:

```bash
pip install pytrendy

```


## Verify Installation

After installation, confirm that PyTrendy is functioning correctly by importing the package and loading a sample dataset:

```python
import pytrendy as pt

# Load a built-in synthetic dataset
df = pt.load_data('series_synthetic')
print(df.head())
```


## Upgrading PyTrendy

To upgrade to the latest version of the package, run the following command:

```bash
pip install --upgrade pytrendy

```

## Installing from Source (For Developers)

For developers who want to install the package directly from the GitHub repository, we recommend creating a virtual environment first.

## Step 1: Create and Activate a Virtual Environment

It is a best practice to install PyTrendy in a virtual environment to avoid conflicts with other projects.

```bash
python -m venv venv
```

### 1. Create a new virtual environment:

```bash
python -m venv venv
```

### 2. Activate the environment:

- On Windows:

```bash
venv\Scripts\activate
```

- On macOS/Linux

```bash
source venv/bin/activate
```

## Step 2: Install from GitHub

Once your virtual environment is active, clone the repository and install the package in editable mode:

```bash
git clone https://github.com/RussellSB/pytrendy.git
cd pytrendy
pip install -e .
```


## Next Steps

Once installed, explore PyTrendy’s core functionality:

- `detect_trends()` — executes the full trend detection pipeline
- `load_data()` — loads built-in datasets for testing and classification
- `plot_pytrendy()` — visualizes detected trend segments
- `PyTrendyResults` — provides structured access to results

For a complete guide on how to use PyTrendy, refer to the [Usage](usage.md) section.




