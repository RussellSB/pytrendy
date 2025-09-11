"""Module B."""

import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt

from .io.data_loader import load_data
from .simpledtw import dtw

def process_signals(df:pd.DataFrame, value_col: str):
    """Core logic. Uses savgol filter derivative to find uptrend & downtred, robust to flat (std) & noisy (snr) periods
    
    @df (pd.DataFrame): df with daily data...
    @value_col()
    """
    WINDOW_SMOOTH = 15
    WINDOW_FLAT = int(WINDOW_SMOOTH*0.5)
    WINDOW_NOISE = int(WINDOW_SMOOTH*0.5)

    THRESHOLD_ZEROS = 0.50 # Sensitivity to zeros pct, needed for nosie by snr (0-1)
    THRESHOLD_NOISE = 5 # Sensitivity to detecting noise (recommended 0-10)
    THRESHOLD_SMOOTH = 0.25 # Sensetivity to detecting trends (recommended 0-0.5)

    # 1. Savgol filter (rolling avg improvement). Caters for seasonality with tightness to day.
    df['smoothed'] = savgol_filter(df[value_col], window_length=WINDOW_SMOOTH, polyorder=1)

    # 2. Flat detection using rolling std of savgol filter.
    # with leading and trailing to cater for periods centered windows doesnt cover
    df['smoothed_std'] = df['smoothed'].rolling(WINDOW_FLAT, center=True).std()
    df['smoothed_std_leading'] = df['smoothed'].iloc[::-1].rolling(window=WINDOW_FLAT).std().iloc[::-1]
    df['smoothed_std_trailing'] = df['smoothed'].rolling(WINDOW_FLAT).std()
    df['smoothed_std'] = df['smoothed_std'].fillna(df['smoothed_std_leading']).fillna(df['smoothed_std_trailing'])
    df['flat_flag'] = 0
    rolling_std = df[value_col].rolling(WINDOW_FLAT, center=True).std()
    min_nonzero_std = rolling_std[rolling_std > 0].min()
    df.loc[df['smoothed_std'] <= min_nonzero_std, 'flat_flag'] = 1 # can comment out to not care about flats. Just take flats with up/down

    # 3. Noise detection via SNR. 

    # 3.1 Compute zero flat edge cases
    df['zeros_pct_trailing'] = (df[value_col] == 0).rolling(WINDOW_NOISE, min_periods=1).sum() / WINDOW_NOISE
    df['zeros_pct_leading'] = ((df[value_col] == 0).iloc[::-1].rolling(window=WINDOW_NOISE, min_periods=1).sum() / WINDOW_NOISE).iloc[::-1]
    df['zeros_pct'] = df[['zeros_pct_trailing', 'zeros_pct_leading']].max(axis=1)

    # 3.2 Compute the SNR
    df['signal'] = df[value_col].rolling(window=WINDOW_NOISE, center=True, min_periods=1).mean()
    df['noise'] = df[value_col] - df['signal']
    df['snr'] = 10 * np.log10(df['signal']**2 / df['noise']**2)

    # 3.3 Define noise flag when SNR & not all zero
    df['noise_flag'] = 0
    df.loc[(df['snr'] <= THRESHOLD_NOISE) & (df['zeros_pct'] <= THRESHOLD_ZEROS), 'noise_flag'] = 1

    # 3.4 Double check & refresh noise flag. Distinguish noise from abrupt change. TODO
    df['noise_flag_diff'] = df['noise_flag'].diff()
    noise_starts = df.loc[df['noise_flag_diff'] == 1].index
    noise_ends = df.loc[df['noise_flag_diff'] == -1].index
    
    # Construct noise segments list based on flag_diff
    noise_segments = []
    for noise_start in noise_starts: # Loops from first start onwards
        after_ends = [end for end in noise_ends if end > noise_start]
        noise_end = after_ends[0] if len(after_ends) > 0 else df.index[-1]
        noise_segments.append(dict(start=noise_start, end=noise_end))

    if len(noise_ends) > 0: # Adds noise end with no start if at beginning
        noise_end = noise_ends[0]
        early_starts = [start for start in noise_starts if start < noise_end]
        if len(early_starts) == 0:
            noise_start = df.index[0]
            noise_segments.insert(0, dict(start=noise_start, end=noise_end))

    # Loads classes signals
    if len(noise_segments) > 0: 
        df_class = load_data('classes_signals')
        df_class.set_index('date', inplace=True)
        df_class = (df_class - df_class.min()) / (df_class.max() - df_class.min())

    # Distinguishes noise signals from abrupt change trends.
    for segment in noise_segments:

        width = (segment['end'] - segment['start']).days
        start = segment['start'] - pd.Timedelta(days=width)
        end = segment['end'] + pd.Timedelta(days=width)

        df_segment = df.loc[start:end]
        df_segment = (df_segment - df_segment.min()) / (df_segment.max() - df_segment.min())

        _, cost_abrupt_up, _, _, _ = dtw(df_segment[value_col], df_class['abrupt_up'])
        _, cost_abrupt_down, _, _, _ = dtw(df_segment[value_col], df_class['abrupt_down'])
        
        _, cost_noise_up, _, _, _ = dtw(df_segment[value_col], df_class['noise_up'])
        _, cost_noise_down, _, _, _ = dtw(df_segment[value_col], df_class['noise_down'])

        # If signal more like abrupt (e.g. up but stays up), then not noise (e.g. up then down). Set noise flag to 0.
        if np.argmin([cost_abrupt_up, cost_abrupt_down, cost_noise_up, cost_noise_down]) < 2:
            df.loc[segment['start']:segment['end'], 'noise_flag'] = 0


    # 4. Detect up/down trend. Uses first derivates of savgol filter (like diff). 
    # Results in signal that's uptrend > 0, else down. As long as its not on a flat.
    df['trend_flag'] = 0
    df.loc[df['flat_flag']==1, 'trend_flag'] = -2
    df.loc[df['noise_flag']==1, 'trend_flag'] = -3
    df['smoothed_deriv'] = savgol_filter(df[value_col], window_length=WINDOW_SMOOTH, polyorder=1, deriv=1)
    df.loc[(df['smoothed_deriv'] >= THRESHOLD_SMOOTH) & (df['flat_flag'] == 0) & (df['noise_flag'] == 0), 'trend_flag'] = 1
    df.loc[(df['smoothed_deriv'] < -THRESHOLD_SMOOTH) & (df['flat_flag'] == 0) & (df['noise_flag'] == 0), 'trend_flag'] = -1

    return df