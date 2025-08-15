# %%
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np

def plot_pytrendy(df:pd.DataFrame, value_col: str, segments_enhanced:list):
    # Define colors
    color_map = {
        'Up': 'lightgreen',
        'Down': 'lightcoral',  # soft red
        'Flat': 'lightblue',
        'Noise': 'lightgray',
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
        
        # Add ranking if up/down trend
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
        mpatches.Patch(color='lightcoral', alpha=0.4, label='Down'),
        mpatches.Patch(color='lightblue', alpha=0.4, label='Flat'),
        mpatches.Patch(color='lightgray', alpha=0.4, label='Noise'), # TODO: Show optionally later, based on up_only, down_only, robustness=False ... etc
    ]
    ax.legend(handles=legend_handles, loc='upper right', 
            bbox_to_anchor=(1, 1.15), ncol=4, frameon=True)

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


def get_segments(df: pd.DataFrame, value_col:str):
    map_direction = {
        0: 'Unknown'
        , 1: 'Up'
        , -1: 'Down'
        , -2: 'Flat'
        , -3: 'Noise'
    }

    segment_length = 0
    segment_length_prev = 0
    direction_prev = map_direction[0]
    segments = []

    for index, value in df[['trend_flag']].itertuples():
        direction = map_direction[value]
        if index == df.index.max(): direction = 'Done'

        if direction == direction_prev:
            segment_length += 1
        elif direction != direction_prev: 
            if (    # Save only when satisfies min window for up/down or flat respectively.
                    (direction_prev in ['Up', 'Down'] and (segment_length_prev >= 7)) \
                    or (direction_prev == 'Flat' and (segment_length_prev >= 3)) \
                    or (direction_prev == 'Noise' and (segment_length_prev >= 3)) \
                ):
                start = (pd.to_datetime(index) - pd.Timedelta(days=segment_length_prev+1))
                end = (pd.to_datetime(index) - pd.Timedelta(days=1))

                # Post process around start & ends to get exact peaks and troughs (TODO: get to work better, no overlaps)
                start_area = df.loc[pd.to_datetime(start) - pd.Timedelta(days=7): pd.to_datetime(start) + pd.Timedelta(days=7)].index
                end_area = df.loc[pd.to_datetime(end) - pd.Timedelta(days=7): pd.to_datetime(end) + pd.Timedelta(days=7)].index
                if direction_prev == 'Up':
                    print(df.loc[start_area])
                    start = df.loc[start_area, value_col].idxmin() # idk dont overwrite flats or noise?
                    end = df.loc[end_area, value_col].idxmax()
                if direction_prev == 'Down':
                    start = df.loc[start_area, value_col].idxmax()
                    end = df.loc[end_area, value_col].idxmin()

                # Save the segment
                segments.append({
                    'direction': direction_prev
                    , 'segmenth_length': segment_length_prev
                    , 'start': start.strftime('%Y-%m-%d')
                    , 'end': end.strftime('%Y-%m-%d')
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

    # 3. Noise detection via SNR. Make sure that up/down trend selection isn't overly sensitive to periods of noise
    df['signal'] = df[value_col].rolling(window=15, center=True, min_periods=1).mean()
    df['noise'] = df[value_col] - df['signal']
    df['snr'] = 10 * np.log10(df['signal']**2 / df['noise']**2)
    df['noise_flag'] = 0
    df.loc[df['snr'] <= 5, 'noise_flag'] = 1

    # signal_power = np.mean(df['signal']**2)
    # noise_power = np.mean(df['noise']**2)
    # snr_db = 10 * np.log10(signal_power / noise_power)
    # print(f"SNR (dB): {snr_db:.2f}")
    # ax = df[[value_col, 'noise_flag']].plot(figsize=(20,3), secondary_y='noise_flag')
    # ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    # plt.show()

    # 4. Detect up/down trend. Uses first derivates of savgol filter (like diff). 
    # Results in signal that's uptrend > 0, else down. As long as its not on a flat.
    df['trend_flag'] = 0
    df.loc[df['flat_flag']==1, 'trend_flag'] = -2
    df.loc[df['noise_flag']==1, 'trend_flag'] = -3
    df['smoothed_deriv'] = savgol_filter(df[value_col], window_length=WINDOW_SMOOTH, polyorder=1, deriv=1)
    df.loc[(df['smoothed_deriv'] >= 0) & (df['flat_flag'] == 0) & (df['noise_flag'] == 0), 'trend_flag'] = 1
    df.loc[(df['smoothed_deriv'] < 0) & (df['flat_flag'] == 0) & (df['noise_flag'] == 0), 'trend_flag'] = -1

    # ax = df[[value_col, 'trend_flag']].plot(figsize=(20,3), secondary_y='trend_flag')
    # ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    # plt.show()

    return df

def main(df:pd.DataFrame, date_col:str, value_col: str):
    """Main pipeline TODO: talk about it all...!"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = process_signals(df, value_col)
    segments = get_segments(df, value_col)
    segments = enhance_segments(df, value_col, segments)
    plot_pytrendy(df, value_col, segments)

    return segments

# %%
# Use Case 1: Simple
df = pd.read_csv('./data/series_gradual.csv', infer_datetime_format=True)
segments = main(df, date_col='date', value_col='value')

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

