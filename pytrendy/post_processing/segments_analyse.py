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