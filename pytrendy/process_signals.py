import pandas as pd
import numpy as np

from scipy.signal import savgol_filter

def process_signals(df:pd.DataFrame, value_col: str):
    """Core logic. Uses savgol filter derivative to find uptrend & downtred, robust to flat (std) & noisy (snr) periods"""
    WINDOW_SMOOTH = 15
    WINDOW_FLAT = int(WINDOW_SMOOTH*0.5)
    WINDOW_NOISE = int(WINDOW_SMOOTH*0.5)

    THRESHOLD_ZEROS = 0.50 # Sensitivity to zeros pct, needed for nosie by snr (0-1)asdasdas
    THRESHOLD_NOISE = 5 # Sensitivity to detecting noise (recommended 0-10)
    THRESHOLD_SMOOTH = 0.25 # Sensetivity to detecting trends (recommended 0-0.5)

    # 1. Flat detection using rolling std of savgol filter.
    # with leading and trailing to cater for periods centered windows doesnt cover
    df['smoothed'] = savgol_filter(df[value_col], window_length=WINDOW_SMOOTH, polyorder=1)

    df['smoothed_std'] = df['smoothed'].rolling(WINDOW_FLAT, center=True).std()
    df['smoothed_std_leading'] = df['smoothed'].iloc[::-1].rolling(window=WINDOW_FLAT).std().iloc[::-1]
    df['smoothed_std_trailing'] = df['smoothed'].rolling(WINDOW_FLAT).std()
    df['smoothed_std'] = df['smoothed_std'].fillna(df['smoothed_std_leading']).fillna(df['smoothed_std_trailing'])

    df['flat_flag'] = 0
    rolling_std = df[value_col].rolling(WINDOW_FLAT, center=True).std()
    min_nonzero_std = rolling_std[rolling_std > 0].min()
    df.loc[df['smoothed_std'] <= min_nonzero_std, 'flat_flag'] = 1 # can comment out to not care about flats. Just take flats with up/down

    # 2. Noise detection via SNR. 
    # 2.1 Compute zero flat edge cases
    df['zeros_pct_trailing'] = (df[value_col] == 0).rolling(WINDOW_NOISE, min_periods=1).sum() / WINDOW_NOISE
    df['zeros_pct_leading'] = ((df[value_col] == 0).iloc[::-1].rolling(window=WINDOW_NOISE, min_periods=1).sum() / WINDOW_NOISE).iloc[::-1]
    df['zeros_pct'] = df[['zeros_pct_trailing', 'zeros_pct_leading']].max(axis=1)

    # 2.2 Compute the SNR
    df['signal'] = df[value_col].rolling(window=WINDOW_NOISE, center=True, min_periods=1).mean()
    df['noise'] = df[value_col] - df['signal']
    df['snr'] = 10 * np.log10(df['signal']**2 / df['noise']**2)

    # 2.3 Define noise flag when SNR & not all zero
    df['noise_flag'] = 0
    df.loc[(df['snr'] <= THRESHOLD_NOISE) & (df['zeros_pct'] <= THRESHOLD_ZEROS), 'noise_flag'] = 1

    # 3. Detect up/down trend. Uses first derivates of savgol filter (like diff). 
    # Savgol filter (rolling avg improvement). Caters for seasonality with tightness to day.
    # Results in signal that's uptrend > 0, else down. As long as its not on a flat or noise.
    df['trend_flag'] = 0
    df.loc[df['flat_flag'] == 1, 'trend_flag'] = -2
    df.loc[df['noise_flag'] == 1, 'trend_flag'] = -3
    df['smoothed_deriv'] = savgol_filter(df[value_col], window_length=WINDOW_SMOOTH, polyorder=1, deriv=1)
    df.loc[(df['smoothed_deriv'] >= THRESHOLD_SMOOTH) & (df['flat_flag'] == 0) & (df['noise_flag'] == 0), 'trend_flag'] = 1
    df.loc[(df['smoothed_deriv'] < -THRESHOLD_SMOOTH) & (df['flat_flag'] == 0) & (df['noise_flag'] == 0), 'trend_flag'] = -1

    return df