"""**Add Metrics and Rank Trend Segments**"""

import pandas as pd
import numpy as np

def analyse_segments(df: pd.DataFrame, value_col: str, segments: list[dict]) -> list[dict]:
    """
    Enhances trend segments with quantitative metrics and rankings.

    This function compares signal behaviour before and after each trend period to characterize
    the magnitude and clarity of change. 

    It computes descriptors that reflect how the signal
    transitions from a pretreatment state (before the trend) to a post-treatment state (after the trend),
    helping to validate the significance of each detected segment.

    Metrics added include:
    
    - Absolute and percent change (based on start/end values)

    - Duration in days

    - Cumulative total change (sum of diffs)

    - Signal-to-noise ratio (SNR)

    - Change rank (based on total change magnitude)

    These enhancements support downstream filtering, ranking, and visualization.

    Args:
        df (pd.DataFrame): 
            Time series DataFrame containing signal, noise, and smoothed columns.
        value_col (str): 
            Name of the column containing the signal to analyse.
        segments (list): 
            List of segment dictionaries with `'start'`, `'end'`, and `'direction'`.

    Returns:
        list: 
            A list of enhanced segment dictionaries with additional keys:
            - `'change'`, `'pct_change'`, `'days'`, `'total_change'`, `'SNR'`, `'change_rank'`
    """
    segments_enhanced = []
    for segment in segments:
        segment_enhanced = segment.copy()
        df_segment = df.loc[segment['start']:segment['end']]

        # Calculate absolute and relative change from first point to last point of trend.
        val_start = df_segment[value_col].iloc[0]
        val_end = df_segment[value_col].iloc[-1]
        segment_enhanced['change'] = float(val_end - val_start)
        segment_enhanced['pct_change'] = (float(val_end / val_start - 1) if val_start != 0 else np.nan)

        # Calculate days & cumulative total change
        days = segment['end'] - segment['start']
        if days == 0: 
            days = 1 # edge case for 1 day flat between noise spike & trend
        segment_enhanced['days'] = days # set days

        # Calculate cumulative total change
        segment_enhanced['total_change'] = float(df_segment[value_col].diff().sum())

        # Calculate Signal to Noise Ratio
        signal_power = np.mean(df_segment['signal']**2)
        noise_power = np.mean(df_segment['noise']**2)
        segment_enhanced['SNR'] = float(10 * np.log10(signal_power / noise_power)) if noise_power != 0 else np.nan
        segments_enhanced.append(segment_enhanced)

    # Establish time index, earliest to latest
    for i, _ in enumerate(segments_enhanced):
        segments_enhanced[i]['time_index'] = i+1

    # Rank change, by steepest to shallowest change
    sorted_segments = sorted(segments_enhanced, key=lambda x: abs(x.get('total_change', 0)), reverse=True)
    for i, seg in enumerate(sorted_segments):
        j = seg['time_index'] - 1
        segments_enhanced[j]['change_rank'] = int(i+1)

    return segments_enhanced