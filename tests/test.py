#%%
import os
os.getcwd()

# %%
%load_ext autoreload
%autoreload 2

# %%
%reload_ext autoreload

#%%
import pytrendy as pt
# Example Runs
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125 # added
df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250 # added more recently
df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
# df[['abrupt']].plot(figsize=(20,5))
results = pt.detect_trends(df.reset_index(), date_col='date', value_col='abrupt')

#%%
df = pt.load_data('series_synthetic')
results_gradual = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True)
results_abrupt = pt.detect_trends(df, date_col='date', value_col='abrupt', plot=True)

# %%
import numpy as np
for i in range(50):
    for noise_std in [10]: #[0, 10, 15, 20,50]
        print(f'Noise value: {noise_std}')
        df = pt.load_data('series_synthetic')
        df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
        results = pt.detect_trends(df, date_col='date', value_col='value_noisy')


# %%
df.to_csv('../temp_noisy_crash_3.csv')        

# %%
noise_df = pd.read_csv('../temp_noisy_crash_2.csv')
noise_df['date'] = pd.to_datetime(noise_df['date'])
noise_df = noise_df.set_index('date')
noise_df['value_noisy'].plot(figsize=(20,3))

# %%
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy')

# %% 
import matplotlib.pyplot as plt
import pandas as pd

temp_df = pd.read_csv('../temp_2.csv')
temp_df['date'] = pd.to_datetime(temp_df['date'])
temp_df = temp_df.set_index('date')
temp_df['value_noisy'].plot(figsize=(20,3))
plt.show()

# %%
results = pt.detect_trends(temp_df.reset_index(), date_col='date', value_col='value_noisy')
# %%
