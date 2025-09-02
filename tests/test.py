
# %%
%load_ext autoreload
%autoreload 2

# %%
%reload_ext autoreload

#%%
import pytrendy as pt
# Example Runs

df = pt.load_data('series_synthetic')
results_gradual = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True)
results_abrupt = pt.detect_trends(df, date_col='date', value_col='abrupt', plot=True)

# %%
# noise_std = 20
# df = pt.load_data('series_synthetic')
# df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
# results = pt.detect_trends(df, date_col='date', value_col='value_noisy')
# results = pt.detect_trends(df, date_col='date', value_col='value_noisy')

# # %%
import numpy as np
for noise_std in [0, 10, 20, 50]:
    print(f'Noise value: {noise_std}')
    df = pt.load_data('series_synthetic')
    df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
    results = pt.detect_trends(df, date_col='date', value_col='value_noisy')
# %%
