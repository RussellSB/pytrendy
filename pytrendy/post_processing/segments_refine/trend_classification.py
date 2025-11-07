"""**Trend Classification Utilities**

Functions for classifying segments as 'gradual' or 'abrupt' using DTW.
"""

import pandas as pd
import numpy as np
from copy import deepcopy
from ...simpledtw import dtw
from ...io.data_loader import load_data


def classify_trends(df: pd.DataFrame, value_col: str, segments: list[dict]) -> list[dict]:
    """
    Classifies segments as 'gradual' or 'abrupt' using DTW against reference signals.

    Adds a `'trend_class'` key to each segment based on similarity to synthetic patterns.

    Args:
        df (pd.DataFrame): Time series DataFrame.
        value_col (str): Name of the signal column.
        segments (list): List of segment dictionaries.

    Returns:
        list: Segment list with added `'trend_class'` labels.
    """

    segments_classified = deepcopy(segments)

    df_class = load_data('classes_signals')
    df_class.set_index('date', inplace=True)
    df_class = (df_class - df_class.min()) / (df_class.max() - df_class.min())

    for i, segment in enumerate(segments):

        if segment['direction'] not in ['Up', 'Down']: 
            continue

        # Assume some padding for abrupt cases
        start = pd.to_datetime(segment['start']) - pd.Timedelta(days=2)
        end = pd.to_datetime(segment['end']) + pd.Timedelta(days=2)

        df_segment = df.loc[start:end]
        df_segment = (df_segment - df_segment.min()) / (df_segment.max() - df_segment.min())

        if segment['direction'] == 'Up': # using value cleaned to not misclassify as abrupt when padded around noise
            _, cost_gradual_up, _, _, _ = dtw(df_segment['value_cleaned'], df_class['gradual_up'])
            _, cost_abrupt_up, _, _, _ = dtw(df_segment['value_cleaned'], df_class['abrupt_up'])

            # round up, so default to gradual if too close
            cost_gradual_up = round(cost_gradual_up, 1)
            cost_abrupt_up = round(cost_abrupt_up, 1)

            if np.argmin([cost_gradual_up, cost_abrupt_up]) == 0:
                segments_classified[i]['trend_class'] = 'gradual'
            else:
                segments_classified[i]['trend_class'] = 'abrupt'
        
        if segment['direction'] == 'Down': 

            _, cost_gradual_down, _, _, _ = dtw(df_segment['value_cleaned'], df_class['gradual_down'])
            _, cost_abrupt_down, _, _, _ = dtw(df_segment['value_cleaned'], df_class['abrupt_down'])

            # round up, so default to gradual if too close
            cost_gradual_down = round(cost_gradual_down, 1)
            cost_abrupt_down = round(cost_abrupt_down, 1)

            if np.argmin([cost_gradual_down, cost_abrupt_down]) == 0:
                segments_classified[i]['trend_class'] = 'gradual'
            else:
                segments_classified[i]['trend_class'] = 'abrupt'

        # Final condition, hard-classify graduals as abrupt if too short
        segment_length = (pd.to_datetime(segment['end']) - pd.to_datetime(segment['start'])).days
        if segment_length < 3:
            segments_classified[i]['trend_class'] = 'abrupt'

    return segments_classified
