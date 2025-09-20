from pathlib import Path
import pandas as pd

def load_data(dataset:str='series_synthetic') -> pd.DataFrame:

    options = ['classes_signals', 'series_synthetic']
    if dataset not in options:
        print(f'{dataset} is not a valid dataset to load from Pytrendy. Please try either of {options}')

    dir_path = Path(__file__).resolve().parent
    file_path = dir_path / "data" / f"{dataset}.csv"
    df = pd.read_csv(file_path)
    return df