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

# ---------- Abrupts and Spikes

#%%
# synth 1
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125 #TODONE: cater for this abrupt detected as noise again # TODONE: cater for clean artifacts update, dont clean if too flat
df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200 
results = pt.detect_trends(df.reset_index(), date_col='date', value_col='abrupt', method_params=dict(is_abrupt_padded=False))

#%%
# synth 2 - 1 spike
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300 # TODONE: shave noise more precisely
results = pt.detect_trends(df.reset_index(), date_col='date', value_col='abrupt')

#%%
# synth 3 - 3 spikes
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125 #TODONE: detect noise on this flat more precisely # TODONE: detect noise well while also not detecting abrupt on right as noise
df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300 # TODONE: shave noise more precisely
df.loc['2025-02-01':'2025-02-01', 'abrupt'] = 500  # TODONE: detect the noise appropriately
df.loc['2025-03-01':'2025-03-01', 'abrupt'] = 500  # TODONE: fix that it neglects downtrend abrupt on right
# df[['abrupt']].plot(figsize=(20,5))
results = pt.detect_trends(df.reset_index(), date_col='date', value_col='abrupt')

#%%
# synth 4 - 4 spikes
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300 # TODONE: shave noise more precisely

df.loc['2025-02-01':'2025-02-01', 'abrupt'] = 500  # TODONE: detect the noise appropriately
# df.loc['2025-02-25':'2025-02-25', 'abrupt'] = 500 # TODONE: fix that it affects uptrend abrupt on left #TODONE: fix flat overlap from right # TODONE: flat fill ins
df.loc['2025-03-01':'2025-03-01', 'abrupt'] = 500 # TODONE: fix that it neglects downtrend abrupt on right
df.loc['2025-04-14':'2025-04-14', 'abrupt'] = 500 #TODONE: improve downtrends on right, so it doesnt displace start left # TODONE: fix that it affects downtrend gradual on right
# df[['abrupt']].plot(figsize=(20,5))
results = pt.detect_trends(df.reset_index(), date_col='date', value_col='abrupt')

# ---------- Original Test Cases

#%%
# original test case 1: gradual
df = pt.load_data('series_synthetic')
results = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True, method_params=dict(is_abrupt_padded=False))

#%%
# original test case 2.1: abrupt
df = pt.load_data('series_synthetic')
results = pt.detect_trends(df, date_col='date', value_col='abrupt', plot=True, method_params=dict(is_abrupt_padded=False)) # TODONE: Fix downtrend from 2nd pass shave

#%%
# original test case 2.2: abrupt padded
df = pt.load_data('series_synthetic')
results = pt.detect_trends(df, date_col='date', value_col='abrupt', plot=True, method_params=dict(is_abrupt_padded=True)) 

# ---------- Random Noise

# %%
# noise test 1 - increasing noise 
import numpy as np
for noise_std in [0, 10, 20, 50]:
    print(f'Noise value: {noise_std}')
    df = pt.load_data('series_synthetic')
    df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
    results = pt.detect_trends(df, date_col='date', value_col='value_noisy')

# %%
# noise test 2 - noise noise noise
import numpy as np
for noise_std in [50]*1:                                        #TODONE: improve that it should not detect trends on high noise.
    print(f'Noise value: {noise_std}')
    df = pt.load_data('series_synthetic')
    df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
    results = pt.detect_trends(df, date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True))

# %%
# noise test 3 - run till crashes
import numpy as np
for noise_std in [10]*1: #[0, 10, 15, 20,50]
    print(f'Noise value: {noise_std}')
    df = pt.load_data('series_synthetic')
    df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
    results = pt.detect_trends(df, date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True))

# %%
# noise test 4 - checking more edge cases
import numpy as np
for noise_std in [20]*1: #[0, 10, 15, 20,50]
    print(f'Noise value: {noise_std}')
    df = pt.load_data('series_synthetic')
    df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
    results = pt.detect_trends(df, date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True))

# %%
# Temp
results = pt.detect_trends(df, date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True))

# ---------- Graduals and Spikes

# %%
# spike test 0.1 - add a spike 
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-03-25':'2025-03-25', 'gradual'] = 200 # TODONE: Reopened. fix white gap after noise from final clean artifact # TODONE: still detect precisely after generelisation # TODONE: improve that it doesnt cover full one noise spike. #TODONE: improve that bad red stretches good green change rank 2
# TODO: Reopened. fix that neglects downtrend start, on left of noise
df = df.reset_index()
results_gradual = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True, method_params=dict(is_abrupt_padded=True))

# %%
# spike test 1.1 - add a spike 
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-04-06':'2025-04-06', 'gradual'] = 200 # TODONE: fix noise artifact on right # DONE: fix displaced noise on left
df = df.reset_index()
results_gradual = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True, method_params=dict(is_abrupt_padded=True))

# %%
# spike test 1.2 - add 3 spikes
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-04-08':'2025-04-08', 'gradual'] = 200 
df.loc['2025-05-08':'2025-05-08', 'gradual'] = 200 # TODONE: understand why uncovering this, changes noise at 04-08 to be tighter, and removes flats at beginning
df.loc['2025-06-08':'2025-06-08', 'gradual'] = 200 # TODONE: improved fill in flats to also cover the end # TODONE: still detect precisely after generelisation # TODONE: fix hang up on abrupt shave # TODONE: fix displaced downtrend on right
df = df.reset_index()
results_gradual = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True, method_params=dict(is_abrupt_padded=True))

# %%
# spike test 1.3 - add 3 spikes
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-04-08':'2025-04-08', 'gradual'] = 250 # TODO: see if noise could be any more precise TODONE: fix white gap. TODONE: still detect precisely after generelisation # TODONE: fix hang up on abrupt shave (also messes up for 250)
df = df.reset_index()
results_gradual = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True, method_params=dict(is_abrupt_padded=False))

# %%
# spike test 1.4 - add 3 spikes
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-04-09':'2025-04-09', 'gradual'] = 100
df.loc['2025-05-06':'2025-05-06', 'gradual'] = 200 # TODONE: fix white gap. TODONE: still detect precisely after generelisation #TODONE: make sure still detects after noise changes #TODONE: fix that it doesnt cover noise in middle # TODONE: fix that it kills uptrend on left
# df.loc['2025-04-09':'2025-04-09', 'gradual'] = 100
df = df.reset_index()
results_gradual = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True, method_params=dict(is_abrupt_padded=False))

# %%
# spike test 1.5 - add 3 spikes
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-02-17':'2025-02-17', 'gradual'] = 100
df.loc['2025-04-09':'2025-04-09', 'gradual'] = 150
df.loc['2025-06-03':'2025-06-03', 'gradual'] = 350 # TODONE: fix white gap. TODONE: still detect precisely after generelisation # TODONE: make sure it detects this noise, right now it overcasts with a red (also with 350, 250)
df = df.reset_index()
results_gradual = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True, method_params=dict(is_abrupt_padded=False))

# %%
# spike test 1.6 - add 3 spikes
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-02-17':'2025-02-17', 'gradual'] = 100
df.loc['2025-04-09':'2025-04-09', 'gradual'] = 150
df.loc['2025-06-03':'2025-06-03', 'gradual'] = 320 # TODONE: fix white gap. TODONE: still detect precisely after generelisation # DONE: fix far right wont be exact, group then shave
df = df.reset_index()
results_gradual = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True, method_params=dict(is_abrupt_padded=False))


# %%
# spike test 1.7 - add 4 spikes
df = pt.load_data('series_synthetic')
df.set_index('date', inplace=True)
df.loc['2025-02-28':'2025-02-28', 'gradual'] = 125 # TODONE: fix white gap. TODONE: fix noise detected on right-side
df.loc['2025-04-09':'2025-04-09', 'gradual'] = 150
df.loc['2025-05-08':'2025-05-08', 'gradual'] = 300 
df.loc['2025-06-03':'2025-06-03', 'gradual'] = 320 
df = df.reset_index()
results_gradual = pt.detect_trends(df, date_col='date', value_col='gradual', plot=True, method_params=dict(is_abrupt_padded=False))


# ---------- Previous Edge Case Instances from Noise (dont crash, but not perfect logic)

# %%
noise_df = pd.read_csv('../temp_noisy_edgecase_10.csv') #TODONE: same result with padded False and True  # TODONE: get rid of green 05-23 on true padded # TODONE: make sure detects trends 04-15 - 05-20 (and up on padded true)
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) 
#  df.to_csv('../temp_noisy_edgecase_10.csv')

# %%
noise_df = pd.read_csv('../temp_noisy_edgecase_9.csv') # TODONE: make sensitive to flats, but be sensitive to up from 03-01 and 04-16
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=False)) 
#  df.to_csv('../temp_noisy_edgecase_9.csv')

# %%
noise_df = pd.read_csv('../temp_noisy_edgecase_8.csv') # TODONE: 03-18 upwards should be flat/noise # TODONE 05-08 Upwards end should be one day longer
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=False)) 
# df.to_csv('../temp_noisy_edgecase_8.csv')

# %%
noise_df = pd.read_csv('../temp_noisy_edgecase_7.csv') # TODONE: 05-16 too small a red when padded is False
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=False)) 
# df.to_csv('../temp_noisy_edgecase_7.csv')

# %%
noise_df = pd.read_csv('../temp_noisy_edgecase_6.csv') # TODONE: 03-09 too small a green
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) 
# df.to_csv('../temp_noisy_edgecase_6.csv')

# %%
noise_df = pd.read_csv('../temp_noisy_edgecase_5.csv') #TODONE: 05-01 green overlaps blue
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) 
# df.to_csv('../temp_noisy_edgecase_5.csv')

# %%
noise_df = pd.read_csv('../temp_noisy_edgecase_4.csv') #TODONE: 02-25 should be noise not up # TODONE: Red overlaps green 04-01
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) 
# df.to_csv('../temp_noisy_edgecase_4.csv')   

# %%
noise_df = pd.read_csv('../temp_noisy_edgecase_3.csv') # TODONE: 03-02 could be noise
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) 
# df.to_csv('../temp_noisy_edgecase_3.csv')   

# %%
noise_df = pd.read_csv('../temp_noisy_edgecase_2.csv') # TODONE: fix green at 03-01 start that is should be too tiny for significance
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) 
# df.to_csv('../temp_noisy_edgecase_2.csv')   

# %%
noise_df = pd.read_csv('../temp_noisy_edgecase_1.csv') # TODONE: fix when green overlaps red
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) 
# df.to_csv('../temp_noisy_edgecase_1.csv')   

# ---------- Previous Crash Instances

# ------------ Latest
# %%
noise_df = pd.read_csv('../temp_noisy_crash_7.csv') # TODONE: fix when padded out of bound # TODONE: crash fix
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) # TODONE: doesnt crash now
# df.to_csv('../temp_noisy_crash_7.csv')   

# %%
noise_df = pd.read_csv('../temp_noisy_crash_6.csv')
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) # TODONE: doesnt crash now
# df.to_csv('../temp_noisy_crash_6.csv') 


# %%
noise_df = pd.read_csv('../temp_noisy_crash_5.csv')
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) # TODONE: doesnt crash now
# df.to_csv('../temp_noisy_crash_5.csv') 


# %%
noise_df = pd.read_csv('../temp_noisy_crash_4.csv')
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) # TODONE: doesnt crash now
# df.to_csv('../temp_noisy_crash_4.csv') 

# %%
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True))

# %%
noise_df = pd.read_csv('../temp_noisy_crash_5.csv')
noise_df['date'] = pd.to_datetime(noise_df['date'])
noise_df = noise_df.set_index('date')
noise_df['value_noisy'].plot(figsize=(20,3))

# %%
# df.to_csv('../temp_noisy_crash_5.csv')   

# %%
noise_df = pd.read_csv('../temp_noisy_crash_4.csv')
noise_df['date'] = pd.to_datetime(noise_df['date'])
noise_df = noise_df.set_index('date')
noise_df['value_noisy'].plot(figsize=(20,3))

# %%
results = pt.detect_trends(noise_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) # TODONE: fixed new edge case crash

# %%
# df.to_csv('../temp_noisy_crash_4.csv')        

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
results = pt.detect_trends(temp_df.reset_index(), date_col='date', value_col='value_noisy', method_params=dict(is_abrupt_padded=True)) # TODONE: fix hangup
# %%



# ---------- Documentation Testing 

