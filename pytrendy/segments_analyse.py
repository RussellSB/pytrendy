import pandas as pd
import numpy as np

def analyse_segments(df:pd.DataFrame, value_col: str, segments: list):
    """Add change descriptors of period pretreatment vs posttreatment"""
    segments_enhanced = []
    for segment in segments:
        segment_enhanced = segment.copy()
        df_segment = df.loc[segment['start']:segment['end']]

        # Calculate absolute and relative change from first point to last point of trend.
        # (Using min/max instead of first/last to be more robust to noise.)
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

        # Calculate Signal to Noise Ratio
        signal_power = np.mean(df_segment['signal']**2)
        noise_power = np.mean(df_segment['noise']**2)
        segment_enhanced['SNR'] = float(10 * np.log10(signal_power / noise_power))

        segments_enhanced.append(segment_enhanced)

    # Rank steepest to shallowest change
    sorted_segments = sorted(segments_enhanced, key=lambda x: abs(x.get('total_change', 0)), reverse=True)
    rank_map = {segment.get('time_index'): i + 1 for i, segment in enumerate(sorted_segments)} # Create a mapping from time_index to rank
    for segment in segments_enhanced: # Assign ranks to the original, unsorted segments
        segment['change_rank'] = rank_map.get(segment.get('time_index'))

    return segments_enhanced