# %%
%load_ext autoreload
%autoreload 2

# %%
%reload_ext autoreload

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
    segments = refine_segments(df, value_col, segments)
    # segments = analyse_segments(df, value_col, segments)
    plot_pytrendy(df, value_col, segments)

    return segments

# Use Case 1: Simple
df = pd.read_csv('./data/series_gradual.csv')
segments = main(df, date_col='date', value_col='value')
segments

# %%
