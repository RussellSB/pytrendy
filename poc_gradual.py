# %%
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('./data/series_gradual.csv', infer_datetime_format=True, index_col="date")
df.plot(figsize=(20,3))

# %%
# Main trend indicator with savgol filter
# 1. Savgol filter (rolling avg improvement). Caters for seasonality with tightness to day.
# 2. Uses first derivates (like diff). Results in signal that's uptrend > 0, else down.
df['smoothed_deriv'] = savgol_filter(df['value'], window_length=15, polyorder=1, deriv=1)
ax = df[['value', 'smoothed_deriv']].plot(figsize=(20,3), secondary_y='smoothed_deriv')
ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
plt.show()


# %%
# Derive binary flags when up or down based on Savgol derivative
df['flag_temp'] = 0
df['smoothed_deriv_2'] = df['smoothed_deriv'].diff()
df.loc[(df['smoothed_deriv'] >= 0), 'flag_temp'] = 1
df.loc[(df['smoothed_deriv'] < 0), 'flag_temp'] = -1
ax = df[['value', 'flag_temp']].plot(figsize=(20,3), secondary_y='flag_temp')
ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
plt.show()


# %%
# Improvement Attempt 1:
# trying to improve and cater for trailing end flatlined
ax = df[['value', 'smoothed_deriv', 'smoothed_deriv_2']].plot(figsize=(20,3), secondary_y=['smoothed_deriv_2'])
ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
plt.show()

# makes beginning and end more abrupt
# todo: filter short enough periods (eg 7, and also first detect flat periods)
df['flag_temp'] = 0
df['smoothed_deriv_2'] = df['smoothed_deriv'].diff()
df.loc[(df['smoothed_deriv'] >= 0) & (df['smoothed_deriv_2'] != 0), 'flag_temp'] = 1
df.loc[(df['smoothed_deriv'] < 0) & (df['smoothed_deriv_2'] != 0), 'flag_temp'] = -1
ax = df[['value', 'flag_temp']].plot(figsize=(20,3), secondary_y='flag_temp')
ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
plt.show()

# %%
# Improvement Attempt 2
# First detecting flat periods
df['smoothed'] = savgol_filter(df['value'], window_length=15, polyorder=1)
df['smoothed_std'] = df['smoothed'].rolling(14).std().shift(-7)
# ax = df[['value', 'smoothed_std']].plot(figsize=(20,3), secondary_y='smoothed_std')
# ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
# plt.show()

df['flat_flag'] = 0
df.loc[df['smoothed_std'] < 2, 'flat_flag'] = 1
# ax = df[['value', 'flat_flag']].plot(figsize=(20,3), secondary_y='flat_flag')
# ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
# plt.show()

# %%
# Using the flat periods to filter out flatties
df['flag_temp'] = 0
df['smoothed_deriv_2'] = df['smoothed_deriv'].diff()
df.loc[(df['smoothed_deriv'] >= 0) & (df['flat_flag'] == 0), 'flag_temp'] = 1
df.loc[(df['smoothed_deriv'] < 0) & (df['flat_flag'] == 0), 'flag_temp'] = -1
ax = df[['value', 'flag_temp']].plot(figsize=(20,3), secondary_y='flag_temp')
ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
plt.show()

# %%
# Filtering out windows shorter than 7 data points

# initialise
map_direction = {
    0: 'Flat'
    , 1: 'Up'
    , -1: 'Down'
}
segment_length = 0
segment_length_prev = 0
direction_prev = map_direction[0]
segments = []

for index, value in df[['flag_temp']].itertuples():
    direction = map_direction[value]

    if direction == direction_prev:
        segment_length += 1
    elif direction != direction_prev: 
        if (    # Save only when satisfies min window for up/down or flat respectively.
                (direction_prev in ['Up', 'Down'] and (segment_length_prev >= 7)) \
                or (direction_prev == 'Flat'  and (segment_length_prev >= 3)) \
            ):
            segments.append({
                'direction': direction_prev
                , 'segmenth_length': segment_length_prev
                , 'start': (pd.to_datetime(index) - pd.Timedelta(days=segment_length_prev+1)).strftime('%Y-%m-%d')
                , 'end': (pd.to_datetime(index) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            })
            segment_length=0

    direction_prev = direction
    segment_length_prev = segment_length

display(segments)

# %%
