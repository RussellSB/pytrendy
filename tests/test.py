#%%
import os
os.getcwd()

# %%
%load_ext autoreload
%autoreload 2

# %%
%reload_ext autoreload

# %%
import pytrendy as pt
import pandas as pd

#%%
# synth 1
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125 # added
df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250 # added more recently
df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200 
# df[['abrupt']].plot(figsize=(20,5))
results = pt.detect_trends(df.reset_index(), date_col='date', value_col='abrupt', method_params=dict(is_abrupt_padded=False))

#%%
# synth 2 - 1 spike
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125 # added
df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250 # added more recently
df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300 # TODONE: shave noise more precisely
# TODO: make flat stretch out better, and fill in the gaps
results = pt.detect_trends(df.reset_index(), date_col='date', value_col='abrupt')

#%%
# synth 3 - 3 spikes
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125 # added
df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250 # added more recently
df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300 # TODONE: shave noise more precisely

df.loc['2025-03-01':'2025-03-01', 'abrupt'] = 500 # TODONE: detect the noise appropriately
df.loc['2025-02-01':'2025-02-01', 'abrupt'] = 500 # TODONE: detect the noise appropriately
# df[['abrupt']].plot(figsize=(20,5))
results = pt.detect_trends(df.reset_index(), date_col='date', value_col='abrupt')

#%%
# original test case 1: gradual
df = pt.load_data('series_synthetic')
results = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True, method_params=dict(is_abrupt_padded=False))

#%%
# original test case 2: abrupt
df = pt.load_data('series_synthetic')
results = pt.detect_trends(df, date_col='date', value_col='abrupt', plot=True, method_params=dict(is_abrupt_padded=False)) # TODONE: Fix overfitted down from noise

# %%
# noise test 1 - increasing noise 
import numpy as np
for noise_std in [0, 10, 15, 20, 50]:
    print(f'Noise value: {noise_std}')
    df = pt.load_data('series_synthetic')
    df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
    results = pt.detect_trends(df, date_col='date', value_col='value_noisy')

# %%
# noise test 2 - noise noise noise
import numpy as np
for noise_std in [50]*5:
    print(f'Noise value: {noise_std}')
    df = pt.load_data('series_synthetic')
    df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
    results = pt.detect_trends(df, date_col='date', value_col='value_noisy')

# %%
# noise test 3 - add a spike
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-03-25':'2025-03-25', 'gradual'] = 200 
# TODONE: detect the noise appropriately
# TODONE: shave noise more precisely
df = df.reset_index()
results_gradual = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True, method_params=dict(is_abrupt_padded=True))
# results_abrupt = pt.detect_trends(df, date_col='date', value_col='abrupt', plot=True, method_params=dict(is_abrupt_padded=True))

# %%
# noise test 4 - run till crashes
import numpy as np
for i in range(50):
    for noise_std in [10]: #[0, 10, 15, 20,50]
        print(f'Noise value: {noise_std}')
        df = pt.load_data('series_synthetic')
        df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
        results = pt.detect_trends(df, date_col='date', value_col='value_noisy')

### TEMP NOISE CRASH CASES

# %%
# ----- Latest
noise_df = pd.read_csv('../temp_noisy_crash_4.csv')
noise_df['date'] = pd.to_datetime(noise_df['date'])
noise_df = noise_df.set_index('date')
noise_df['value_noisy'].plot(figsize=(20,3))

# %%
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) # TODONE: fixed new edge case crash

# %%
df.to_csv('../temp_noisy_crash_4.csv')        

# %%
noise_df = pd.read_csv('../temp_noisy_crash_2.csv')
noise_df['date'] = pd.to_datetime(noise_df['date'])
noise_df = noise_df.set_index('date')
noise_df['value_noisy'].plot(figsize=(20,3))

# %%
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True))

# %% 
import matplotlib.pyplot as plt
import pandas as pd

temp_df = pd.read_csv('../temp_2.csv')
temp_df['date'] = pd.to_datetime(temp_df['date'])
temp_df = temp_df.set_index('date')
temp_df['value_noisy'].plot(figsize=(20,3))
plt.show()

# %%
results = pt.detect_trends(temp_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True))
# %%
