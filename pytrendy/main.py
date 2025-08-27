# %%
%load_ext autoreload
%autoreload 2
%reload_ext autoreload 

#%%
import pandas as pd
from process_signals import process_signals
from segments_get import get_segments
from segments_refine import refine_segments
from segments_analyse import analyse_segments
from plot_pytrendy import plot_pytrendy

def main(df:pd.DataFrame, date_col:str, value_col: str):
    """Main pipeline TODO: talk about it all...!"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = process_signals(df, value_col)
    segments = get_segments(df)
    segments = refine_segments(df, value_col, segments)
    segments = analyse_segments(df, value_col, segments)
    plot_pytrendy(df, value_col, segments)

    return segments

#%%
# Example Runs
df = pd.read_csv('../data/series_synthetic.csv')
segments = main(df, date_col='date', value_col='abrupt')
segments = main(df, date_col='date', value_col='gradual')
segments = main(df, date_col='date', value_col='gradual-noisy-20')



# %%
# noise_std = 20
# df = pd.read_csv('./data/series_synthetic.csv')
# df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
# segments = main(df, date_col='date', value_col='value_noisy')
# segments = main(df, date_col='date', value_col='value_noisy')

# %%
import numpy as np
for noise_std in [0, 10, 20, 50]:
    print(f'Noise value: {noise_std}')
    df = pd.read_csv('../data/series_synthetic.csv')
    df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
    segments = main(df, date_col='date', value_col='value_noisy')
# %%
