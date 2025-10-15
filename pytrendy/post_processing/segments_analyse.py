"""**Add Metrics and Rank Trend Segments**"""

import pandas as pd
import numpy as np

def analyse_segments(df:pd.DataFrame, value_col: str, segments: list):
    """
    Enhances trend segments with quantitative metrics and rankings.

    This function compares signal behavior before and after each trend period to characterize
    the magnitude and clarity of change. 

    It computes descriptors that reflect how the signal
    transitions from a pretreatment state (before the trend) to a post-treatment state (after the trend),
    helping to validate the significance of each detected segment.

    Metrics added include:
    
    - Absolute and percent change (based on min/max values)

    - Duration in days

    - Cumulative total change (sum of diffs)

    - Signal-to-noise ratio (SNR)

    - Change rank (based on total change magnitude)

    These enhancements support downstream filtering, ranking, and visualization.

    Args:
        df (pd.DataFrame): 
            Time series DataFrame containing signal, noise, and smoothed columns.
        value_col (str): 
            Name of the column containing the signal to analyze.
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
        # (Using min/max instead of first/last to be more robust to noise.)
        val_min = df_segment[value_col].min()
        val_max = df_segment[value_col].max()
        if segment['direction'] == 'Up':  # max - min
            segment_enhanced['change'] = float(val_max - val_min)
            segment_enhanced['pct_change'] = (
                float(val_max / val_min - 1) if val_min != 0 else np.nan
            )
        elif segment['direction'] == 'Down':  # min - max
            segment_enhanced['change'] = float(val_min - val_max)
            segment_enhanced['pct_change'] = (
                float(val_min / val_max - 1) if val_max != 0 else np.nan
            )

        # Calculate days & cumulative total change
        segment_enhanced['days'] = (pd.to_datetime(segment['end']) - pd.to_datetime(segment['start'])).days
        if segment['direction'] in ['Up', 'Down']:
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
    sorted_trends = [seg for seg in sorted_segments if 'total_change' in seg and abs(seg['total_change']) > 0]
    for i, seg in enumerate(sorted_trends):
        j = seg['time_index'] - 1
        segments_enhanced[j]['change_rank'] = int(i+1)

    return segments_enhanced