# Installation

PyTrendy can be installed via `pip` and used immediately for trend detection and time series analysis.


## Installation Requirements and Dependencies

### Python

- **Python version:** 3.10 or newer


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

After installation, confirm that PyTrendy is functioning correctly:

```python
import pytrendy as pt
print(f"PyTrendy version: {pt.__version__}")
```

!!! success "Ready to Go!"
    If the import succeeds, you're ready to start analyzing trends! Check out the [User Guide](user-guide/index.md) or jump into the [Tutorials](tutorials/index.md).


## Upgrading PyTrendy

To upgrade to the latest version of the package, run the following command:

```bash
pip install --upgrade pytrendy

```

## Installing from Source (For Developers)

For developers who want to install the package directly from the GitHub repository, we recommend creating a virtual environment first.

### Step 1: Create and Activate a Virtual Environment

It is a best practice to install PyTrendy in a virtual environment to avoid conflicts with other projects.

Create a new virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

=== "Linux"

    ```bash
    source .venv/bin/activate
    ```

=== "Windows"

    ```powershell
    venv\Scripts\activate
    ```

### Step 2: Install from PyPi

Once your virtual environment is active, install the package from PyPi:

```bash
pip install pytrendy
```


## Next Steps

Once installed, explore PyTrendy's core functionality:

- **[User Guide](user-guide/index.md)** — Complete guide to using PyTrendy's API
- **[Tutorials](tutorials/index.md)** — Hands-on examples with real-world data
- **[API Reference](reference/pytrendy/index.md)** — Detailed function documentation




