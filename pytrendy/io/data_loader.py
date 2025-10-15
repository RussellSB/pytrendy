"""**Load Built-In Datasets for Testing and Classification**"""

from pathlib import Path
import pandas as pd

def load_data(dataset:str='series_synthetic') -> pd.DataFrame:
    """
    Loads sample datasets bundled with PyTrendy.

    This provides quick access to preloaded datasets for testing and demonstration.
    Available datasets include synthetic time series and trend classification examples.

    Args:
        dataset (str, optional):
            Name of the dataset to load. Options include:

            - `'series_synthetic'`: A synthetic time series with embedded trends.
            - `'classes_trends'`: Reference signals for classifying trends as gradual or abrupt.

    Returns:
        pd.DataFrame:
            A pandas DataFrame containing the requested dataset.
    """

    options = ['classes_signals', 'series_synthetic']
    if dataset not in options:
        print(f'{dataset} is not a valid dataset to load from PyTrendy. Please try either of {options}')

    dir_path = Path(__file__).resolve().parent
    file_path = dir_path / "data" / f"{dataset}.csv"
    df = pd.read_csv(file_path)
    return df