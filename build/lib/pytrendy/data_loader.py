from pathlib import Path
import pandas as pd

def load_data(dataset:str='series_synthetic') -> pd.DataFrame:

    options = ['classes_trends', 'series_synthetic']
    if dataset not in options:
        print(f'{dataset} is not a valid dataset to load from Pytrendy. Please try either of {options}')

    dir_path = str(Path(__file__).resolve().parent)
    df = pd.read_csv(dir_path + '/data/' + dataset + '.csv')
    return df