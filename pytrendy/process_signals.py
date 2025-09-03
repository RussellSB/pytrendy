import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt

def process_signals(df:pd.DataFrame, value_col: str):
    """Core logic. Uses savgol filter derivative to find uptrend & downtred, robust to flat (std) & noisy (snr) periods"""
    WINDOW_SMOOTH = 15
    WINDOW_FLAT = int(WINDOW_SMOOTH*0.5)
    WINDOW_NOISE = int(WINDOW_SMOOTH*0.5)

    THRESHOLD_FLAT = 0 # Sensitivity to detecting flats (recommended 0-0.5)
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
    threshold_flat = df[value_col].rolling(WINDOW_FLAT, center=True).std().quantile(THRESHOLD_FLAT) # initially set at 2 for series_gradual example
    df.loc[df['smoothed_std'] <= threshold_flat, 'flat_flag'] = 1 # can comment out to not care about flats. Just take flats with up/down

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
    
    ax = df[[value_col, 'zeros_pct_trailing']].plot(figsize=(20,3), secondary_y='zeros_pct_trailing')
    ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    plt.show()   
    
    ax = df[[value_col, 'zeros_pct_leading']].plot(figsize=(20,3), secondary_y='zeros_pct_leading')
    ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    plt.show()

    ax = df[[value_col, 'zeros_pct']].plot(figsize=(20,3), secondary_y='zeros_pct')
    ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    plt.show()

    ax = df[[value_col, 'snr']].plot(figsize=(20,3), secondary_y='snr')
    ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    plt.show()
    
    ax = df[[value_col, 'noise_flag']].plot(figsize=(20,3), secondary_y='noise_flag')
    ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    plt.show()

    # 4. Detect up/down trend. Uses first derivates of savgol filter (like diff). 
    # Results in signal that's uptrend > 0, else down. As long as its not on a flat.
    df['trend_flag'] = 0
    df.loc[df['flat_flag']==1, 'trend_flag'] = -2
    df.loc[df['noise_flag']==1, 'trend_flag'] = -3
    df['smoothed_deriv'] = savgol_filter(df[value_col], window_length=WINDOW_SMOOTH, polyorder=1, deriv=1)
    df.loc[(df['smoothed_deriv'] >= THRESHOLD_SMOOTH) & (df['flat_flag'] == 0) & (df['noise_flag'] == 0), 'trend_flag'] = 1
    df.loc[(df['smoothed_deriv'] < -THRESHOLD_SMOOTH) & (df['flat_flag'] == 0) & (df['noise_flag'] == 0), 'trend_flag'] = -1

    return df