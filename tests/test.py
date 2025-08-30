
# %%
%load_ext autoreload
%autoreload 2

# %%
%reload_ext autoreload

#%%
import pandas as pd
from detect_trends import detect_trends

# Example Runs
df = pd.read_csv('./data/series_synthetic.csv')
# results = detect_trends(df, date_col='date', value_col='gradual-noisy-20')
# results = detect_trends(df, date_col='date', value_col='abrupt')
results = detect_trends(df, date_col='date', value_col='gradual', plot=False)
results.filter_segments(sort_by='time_index')[:3]

# %%
# noise_std = 20
# df = pd.read_csv('./data/series_synthetic.csv')
# df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
# results = main(df, date_col='date', value_col='value_noisy')
# results = main(df, date_col='date', value_col='value_noisy')

# # %%
# import numpy as np
# for noise_std in [0, 10, 20, 50]:
#     print(f'Noise value: {noise_std}')
#     df = pd.read_csv('./data/series_synthetic.csv')
#     df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
#     results = main(df, date_col='date', value_col='value_noisy')