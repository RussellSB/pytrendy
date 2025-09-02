#%%
import os
os.getcwd()

# %%
%load_ext autoreload
%autoreload 2

# %%
%reload_ext autoreload

#%%
import pandas as pd
import pytrendy as pt

# Example Runs
df = pt.load_data('series_synthetic')
df['str_var'] = 'str_example'
results = pt.detect_trends(df, date_col='date', value_col='abrupt')

#%%
# df = pt.load_data('series_synthetic')
# results = pt.detect_trends(df, date_col='date', value_col='gradual-noisy-20')
# results = pt.detect_trends(df, date_col='date', value_col='abrupt')
# results = pt.detect_trends(df, date_col='date', value_col='gradual', plot=False)

# %%
# noise_std = 20
# df = pt.load_data('series_synthetic')
# df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
# results = pt.detect_trends(df, date_col='date', value_col='value_noisy')
# results = pt.detect_trends(df, date_col='date', value_col='value_noisy')

# # %%
# import numpy as np
# for noise_std in [0, 10, 20, 50]:
#     print(f'Noise value: {noise_std}')
#     df = pt.load_data('series_synthetic')
#     df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
#     results = pt.detect_trends(df, date_col='date', value_col='value_noisy')
# %%
