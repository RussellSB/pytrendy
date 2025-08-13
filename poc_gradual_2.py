# %%
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

def plot_pytrendy(df:pd.DataFrame, value_col: str, segments_enhanced:list):
    # Define colors
    color_map = {
        'Up': 'lightgreen',
        'Down': 'lightcoral',  # soft red
        'Flat': 'lightblue'
    }

    fig, ax = plt.subplots(figsize=(20, 5))

    # Plot the value line
    ax.plot(df.index, df[value_col], color='black', lw=1)

    # Add shaded regions with fill_between
    ymin, ymax = ax.get_ylim()  # get plot's visible y-range
    for rank, seg in enumerate(segments_enhanced, start=1):
        start = pd.to_datetime(seg['start'])
        end = pd.to_datetime(seg['end'])
        color = color_map.get(seg['direction'], 'gray')

        mask = (df.index >= start) & (df.index <= end + pd.Timedelta(days=1))
        ax.fill_between(df.index[mask], ymin, ymax, color=color, alpha=0.4)
        
        if seg['direction'] in ['Up', 'Down']:
            mid_date = start + (end - start) / 2
            y_pos = ymax - (ymax - ymin) * 0.05

            ax.text(mid_date, y_pos, str(rank), fontsize=12,
                    fontweight='bold', ha='center', va='top',
                    color=color[5:])

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


def enhance_segments(df:pd.DataFrame, value_col: str, segments: list):
    segments_enhanced = []
    for segment in segments:
        segment_enhanced = segment.copy()
        df_segment = df.loc[segment['start']:segment['end']]
        # Best to use min/max instead of first/last to be more robust to noise.
        if segment['direction'] == 'Up': # max - min
            segment_enhanced['change'] = float(df_segment[value_col].max() - df_segment[value_col].min())
            segment_enhanced['pct_change'] = float(df_segment[value_col].max()/df_segment[value_col].min() -1)
        if segment['direction'] == 'Down': # min - max
            segment_enhanced['change'] = float(df_segment[value_col].min() - df_segment[value_col].max())
            segment_enhanced['pct_change'] = float(df_segment[value_col].min()/df_segment[value_col].max() -1)

        # Calculate days & cumulative total change
        segment_enhanced['days'] = (pd.to_datetime(segment['end']) - pd.to_datetime(segment['start'])).days
        if segment['direction'] in ['Up', 'Down']:
            segment_enhanced['total_change'] = float(df_segment[value_col].diff().sum())

        # Append
        segments_enhanced.append(segment_enhanced)

    # Rank steepest to shallowest change
    segments_enhanced = sorted(segments_enhanced, key=lambda x: abs(x.get('total_change', 0)), reverse=True)
    return segments_enhanced


def get_segments(df: pd.DataFrame):
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
        if index == df.index.max(): direction = 'Done'

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

    return segments # main result


def process_signals(df:pd.DataFrame, value_col: str):
    WINDOW_SMOOTH = 15
    WINDOW_FLAT = int(WINDOW_SMOOTH*0.5)

    # 1. Savgol filter (rolling avg improvement). Caters for seasonality with tightness to day.
    df['smoothed'] = savgol_filter(df[value_col], window_length=WINDOW_SMOOTH, polyorder=1)

    # 2. Flat detection using rolling std of savgol filter.
    # with leading and trailing to cater for periods centered windows doesnt cover
    df['smoothed_std'] = df['smoothed'].rolling(WINDOW_FLAT, center=True).std()
    df['smoothed_std_leading'] = df['smoothed'].iloc[::-1].rolling(window=WINDOW_FLAT).std().iloc[::-1]
    df['smoothed_std_trailing'] = df['smoothed'].rolling(WINDOW_FLAT).std()
    df['smoothed_std'] = df['smoothed_std'].fillna(df['smoothed_std_leading']).fillna(df['smoothed_std_trailing'])
    df['flat_flag'] = 0
    threshold_flat = df['value'].rolling(int(WINDOW_FLAT), center=True).std().min() # initially set at 2 for series_gradual example
    df.loc[df['smoothed_std'] < threshold_flat, 'flat_flag'] = 1 # can comment out to not care about flats. Just take flats with up/down

    # 3. Detect up/down trend. Uses first derivates of savgol filter (like diff). 
    # Results in signal that's uptrend > 0, else down. As long as its not on a flat.
    df['flag_temp'] = 0
    df['smoothed_deriv'] = savgol_filter(df[value_col], window_length=WINDOW_SMOOTH, polyorder=1, deriv=1)
    df.loc[(df['smoothed_deriv'] >= 0) & (df['flat_flag'] == 0), 'flag_temp'] = 1
    df.loc[(df['smoothed_deriv'] < 0) & (df['flat_flag'] == 0), 'flag_temp'] = -1

    # ax = df[[value_col, 'flag_temp']].plot(figsize=(20,3), secondary_y='flag_temp')
    # ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    # plt.show()

    return df

def main(df:pd.DataFrame, date_col:str, value_col: str):
    """Main pipeline TODO: talk about it all...!"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = process_signals(df, value_col)
    segments = get_segments(df)
    segments = enhance_segments(df, value_col, segments)
    plot_pytrendy(df, value_col, segments)

    return segments

# %%
# Use Case 1: Simple
df = pd.read_csv('./data/series_gradual.csv', infer_datetime_format=True)
segments = main(df, date_col='date', value_col='value')
segments

# %%
# Use Case 2; Much higher magnitudes
df = pd.read_csv('./data/series_gradual.csv', infer_datetime_format=True)
df['value'] = df['value'] * 50
segments = main(df, date_col='date', value_col='value')

# %%
# Use Case 3; NOISE NOISE NOISE
import numpy as np
for noise_std in [0, 2, 5, 10, 20, 50]:
    print(f'Noise value: {noise_std}')
    df = pd.read_csv('./data/series_gradual.csv')
    df['value_noisy'] = df['value'] + np.random.normal(0, noise_std, size=len(df))
    segments = main(df, date_col='date', value_col='value_noisy')


# %%
# Trying the noise noise noise fix (I want it to not show trends when its noisy!)
import numpy as np
noise_std = 50
print(f'Noise value: {noise_std}')
df = pd.read_csv('./data/series_gradual.csv')
df['value_noisy'] = df['value'] + np.random.normal(0, noise_std, size=len(df))
segments = main(df, date_col='date', value_col='value_noisy')

# %%
segments = main(df, date_col='date', value_col='value_noisy')

# %%
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.signal import periodogram

def detect_noise_windows_kurtosis(df, value_col, window=15, kurtosis_thresh=0.8, autocorr_thresh=0.3, max_lag=3, spectral_ratio_thresh=3):
    """
    Detect 'pure noise' windows robust to seasonal/trending signals.
    Marks df['test'] as 1 (noise) or 0 (not noise).
    """
    # Rolling kurtosis
    roll_kurt = df[value_col].rolling(window).apply(
        lambda s: kurtosis(s, fisher=False), raw=True
    )

    # Rolling mean autocorrelation over lags
    def mean_abs_autocorr(s):
        return np.mean([abs(s.autocorr(lag=lag)) for lag in range(1, max_lag+1)])
    roll_ac_mean = df[value_col].rolling(window).apply(mean_abs_autocorr, raw=False)

    # Rolling spectral dominance
    def spectral_dominance(s):
        f, Pxx = periodogram(s, window='hann')
        if np.mean(Pxx) == 0:
            return 0
        return np.max(Pxx) / np.mean(Pxx)
    roll_spec_dom = df[value_col].rolling(window).apply(spectral_dominance, raw=True)

    # Noise flag: Gaussian-like, low autocorr, flat spectrum
    df['test'] = (
        ((roll_kurt - 3).abs() < kurtosis_thresh) &
        (roll_ac_mean < autocorr_thresh) &
        (roll_spec_dom < spectral_ratio_thresh)
    ).astype(int)

    
    ax = df[[value_col, 'test']].plot(figsize=(20,3), secondary_y='test')
    ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    plt.show()

    return df

# %%
# Example
np.random.seed(42)
df = pd.DataFrame({'value_col': np.concatenate([
    np.random.randn(100),             # noise
    np.sin(np.linspace(0, 20, 100))   # signal
])})

df = detect_noise_windows_kurtosis(df, 'value_col', window=15)


# %%
df = detect_noise_windows_kurtosis(df, 'value_noisy', window=15)
# df.plot(figsize=(20,3))

# %%
df = detect_noise_windows_kurtosis(df, 'value', window=15)


#%%
# df['value_noisy'] = df['value'] + np.random.normal(0, 50, size=len(df))

#%%
# manual try
# df['diff'] = df['value_noisy'] - df['value_noisy'].shift(-1).diff()
# ax = df[['value_noisy', 'diff']].plot(figsize=(20,3), secondary_y='diff')
# ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
# plt.show()

#%% 
df['signal'] = df['value_noisy'].rolling(window=7, center=True, min_periods=1).mean()
df['noise'] = df['value_noisy'] - df['signal']
signal_power = np.mean(df['signal']**2)
noise_power = np.mean(df['noise']**2)
snr_db = 10 * np.log10(signal_power / noise_power)
print(f"SNR (dB): {snr_db:.2f}")
# %%

df['signal'] = df['value'].rolling(window=7, center=True, min_periods=1).mean()
df['noise'] = df['value'] - df['signal']
signal_power = np.mean(df['signal']**2)
noise_power = np.mean(df['noise']**2)
snr_db = 10 * np.log10(signal_power / noise_power)
print(f"SNR (dB): {snr_db:.2f}")