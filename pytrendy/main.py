# %%
%load_ext autoreload
%autoreload 2

# %%
%reload_ext autoreload

#%%
import pandas as pd
from pytrendy.process_signals import process_signals
from pytrendy.segments_get import get_segments
from pytrendy.segments_refine import refine_segments
from pytrendy.segments_analyse import analyse_segments
from pytrendy.plot_pytrendy import plot_pytrendy

def main(df:pd.DataFrame, date_col:str, value_col: str):
    """Main pipeline TODO: talk about it all...!"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = process_signals(df, value_col)
    segments = get_segments(df)
    segments = refine_segments(df, value_col, segments) #TODO: abrupt contractor (3), and grouping (1?)
    segments = analyse_segments(df, value_col, segments)
    plot_pytrendy(df, value_col, segments)

    return segments

# Use Case 2: Abrupt
df = pd.read_csv('./data/series_abrupt.csv')
segments = main(df, date_col='date', value_col='value')
# segments

# %%
# Use Case 1: Simple
# df = pd.read_csv('./data/series_gradual.csv')
# segments = main(df, date_col='date', value_col='value')

# %%


# %%
segments

#%%
# import numpy as np
# for noise_std in [0, 2, 5, 10, 20, 50]:
#     print(f'Noise value: {noise_std}')
#     df = pd.read_csv('./data/series_gradual.csv')
#     df['value_noisy'] = df['value'] + np.random.normal(0, noise_std, size=len(df))
#     segments = main(df, date_col='date', value_col='value_noisy')
# %%
