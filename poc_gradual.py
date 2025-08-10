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
# Improvement Attempt 1: (DISCARDED)
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
# Improvement Attempt 2 (KEPT)
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
# Final Display Mock 1
df[['value']].plot(figsize=(20,5), color='black')
plt.show()# Ensure datetime index

df.index = pd.to_datetime(df.index)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# Define colors
color_map = {
    'Up': 'lightgreen',
    'Down': 'lightcoral',  # soft red
    'Flat': 'lightblue'
}

fig, ax = plt.subplots(figsize=(20, 5))

# Plot the value line
ax.plot(df.index, df['value'], color='black', lw=1)

# Add shaded regions with fill_between
ymin, ymax = ax.get_ylim()  # get plot's visible y-range
for seg in segments:
    start = pd.to_datetime(seg['start'])
    end = pd.to_datetime(seg['end'])
    color = color_map.get(seg['direction'], 'gray')

    mask = (df.index >= start) & (df.index <= end)
    ax.fill_between(df.index[mask],
                    ymin, ymax,
                    color=color, alpha=0.4)


# Set limits
first_date = df.index.min()
last_date = df.index.max()
ax.set_xlim(first_date, last_date)
ax.set_ylim(ymin, ymax)

# Major ticks: every 7 days (with labels)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Minor ticks: every day (no labels, just tick marks/grid)
ax.xaxis.set_minor_locator(mdates.DayLocator())

# Rotate major tick labels
plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

# Optional: show grid lines for both
ax.grid(True, which='major', color='gray', alpha=0.3)

ax.set_title("PyTrendy Detection", fontsize=20)
ax.set_xlabel("Date")
ax.set_ylabel("Value")

# Create custom legend handles (colored boxes)
legend_handles = [
    mpatches.Patch(color='lightgreen', alpha=0.4, label='Up'),
    mpatches.Patch(color='lightblue', alpha=0.4, label='Flat'),
    mpatches.Patch(color='lightcoral', alpha=0.4, label='Down'),
]
ax.legend(handles=legend_handles, loc='upper right', 
          bbox_to_anchor=(1, 1.15), ncol=3, frameon=True)


plt.tight_layout()
plt.show()

# %%
