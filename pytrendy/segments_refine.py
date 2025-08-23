import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from .simpledtw import dtw
import numpy as np

THRESHOLD_DISTANCE = 3

def _update_prev_segment(i, new_start, segments, segments_refined):
    """Shift previous segment end if overlapping with updated start (or original start)."""
    if i == 0:
        return
    distance_refined = (pd.to_datetime(new_start) - pd.to_datetime(segments_refined[i - 1]['end'])).days
    distance_orig = (pd.to_datetime(segments[i]['start']) - pd.to_datetime(segments[i - 1]['end'])).days
    if distance_refined <= THRESHOLD_DISTANCE or distance_orig <= THRESHOLD_DISTANCE:
        segments_refined[i - 1]['end'] = (pd.to_datetime(new_start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')


def _update_next_segment(i, new_end, segments, segments_refined):
    """Shift next segment start if overlapping with updated end (or original end)."""
    if i == len(segments_refined) - 1:
        return
    distance_refined = (pd.to_datetime(segments_refined[i + 1]['start']) - pd.to_datetime(new_end)).days
    distance_orig = (pd.to_datetime(segments[i + 1]['start']) - pd.to_datetime(segments[i]['end'])).days
    if distance_refined <= THRESHOLD_DISTANCE or distance_orig <= THRESHOLD_DISTANCE:
        segments_refined[i + 1]['start'] = (pd.to_datetime(new_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')


def expand_contract_segments(df: pd.DataFrame, value_col: str, segments: list):
    """
    Post-process detected segments by assessing their start and end points.
    Adjusts boundaries by looking ±7 days around each boundary for more precision.
    Is there an appropriately higher or lower point worth taking? Take it.
    If it increases the segment - "expand". If it decreases the segment - "contract".
    """
    segments_refined = deepcopy(segments)

    def _get_window_df(center, days=7):
        """Return a slice of df around a center date ±days."""
        pre = (pd.to_datetime(center) - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        post = (pd.to_datetime(center) + pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        return df.loc[pre:post].copy()

    for i, segment in enumerate(segments_refined):

        start_df = _get_window_df(segment['start'])
        end_df = _get_window_df(segment['end'])

        if segment['direction'] == 'Up':
            new_start = start_df[value_col].idxmin() + pd.Timedelta(days=1)
            new_end = end_df[value_col].idxmax()
        elif segment['direction'] == 'Down':
            new_start = start_df[value_col].idxmax() + pd.Timedelta(days=1)
            new_end = end_df[value_col].idxmin()
        else:
            continue

        # refine start
        if new_start != pd.to_datetime(segment['start']):
            segments_refined[i]['start'] = new_start.strftime('%Y-%m-%d')
            _update_prev_segment(i, new_start, segments, segments_refined)

        # refine end
        if new_end != pd.to_datetime(segment['end']):
            segments_refined[i]['end'] = new_end.strftime('%Y-%m-%d')
            _update_next_segment(i, new_end, segments, segments_refined)

    return segments_refined


def classify_trends(df: pd.DataFrame, value_col: str, segments: list):
    """
    Classifies appropriate segments as pre-defined typed of trends; 
    Gradual or Abrupt. Utilises DTW to compare to synthesized signals.
    """
    segments_classified = deepcopy(segments)

    df_class = pd.read_csv('data/classes_trends.csv')
    df_class.set_index('date', inplace=True)
    df_class = (df_class - df_class.min()) / (df_class.max() - df_class.min())

    for i, segment in enumerate(segments):

        if segment['direction'] not in ['Up', 'Down']: 
            continue

        df_segment = df.loc[segment['start']:segment['end']]
        df_segment = (df_segment - df_segment.min()) / (df_segment.max() - df_segment.min())

        if segment['direction'] == 'Up': 
            _, cost_gradual_up, _, _, _ = dtw(df_segment[value_col], df_class['gradual_up'])
            _, cost_abrupt_up, _, _, _ = dtw(df_segment[value_col], df_class['abrupt_up'])
            if np.argmin([cost_gradual_up, cost_abrupt_up]) == 0:
                segments_classified[i]['trend_class'] = 'gradual'
            else:
                segments_classified[i]['trend_class'] = 'abrupt'
        
        if segment['direction'] == 'Down': 
            _, cost_gradual_down, _, _, _ = dtw(df_segment[value_col], df_class['gradual_down'])
            _, cost_abrupt_down, _, _, _ = dtw(df_segment[value_col], df_class['abrupt_down'])
            if np.argmin([cost_gradual_down, cost_abrupt_down]) == 0:
                segments_classified[i]['trend_class'] = 'gradual'
            else:
                segments_classified[i]['trend_class'] = 'abrupt'

    return segments_classified


def shave_abrupt_trends(df: pd.DataFrame, value_col: str, segments: list):
    """
    Handles case of abrupt trends since changepoint detection is missed by rolling statistics
    We analyse the segment for diff outliers, and take the earliest and latest points from here.
    """
    segments_refined = deepcopy(segments)
    for i, segment in enumerate(segments_refined):
        if segment['direction'] not in ['Up', 'Down'] or segment['trend_class'] != 'abrupt': 
            continue

        # Get start end padded for some leniency
        start = pd.to_datetime(segment['start']) - pd.Timedelta(days=7)
        end = pd.to_datetime(segment['end']) + pd.Timedelta(days=7)
        df_segment = df.loc[start:end].copy()

        df_segment['diff'] = df_segment[value_col].diff()
        df_segment = df_segment.iloc[1:]
        
        df_segment['z_score'] = (df_segment['diff'] - df_segment['diff'].mean()) / df_segment['diff'].std()
        df_segment['abrupt_flag'] = 0
        df_segment.loc[df_segment['z_score'].abs() > 2, 'abrupt_flag'] = 1

        # ax = df_segment[[value_col, 'diff']].plot(figsize=(20,3), secondary_y='diff')
        # ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
        # plt.show()

        # print('--- 35th percentile', df_segment['diff'].abs().quantile(0.35))
        # print('% higher:', ((df_segment['diff'].abs() > df_segment['diff'].abs().quantile(0.5)).sum()) / len(df_segment))

        # ax = df_segment[[value_col, 'z_score']].plot(figsize=(20,3), secondary_y='z_score')
        # ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
        # plt.show()
        # ax = df_segment[[value_col, 'abrupt_flag']].plot(figsize=(20,3), secondary_y='abrupt_flag')
        # ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
        # plt.show()

        new_start = df_segment.loc[df_segment['abrupt_flag'] == 1].index[0] - pd.Timedelta(days=1)
        segments_refined[i]['start'] = new_start.strftime('%Y-%m-%d')
        _update_prev_segment(i, new_start, segments, segments_refined)

        new_end = df_segment.loc[df_segment['abrupt_flag'] == 1].index[-1]
        segments_refined[i]['end'] = new_end.strftime('%Y-%m-%d')
        _update_next_segment(i, new_end, segments, segments_refined)

    return segments_refined



def refine_segments(df: pd.DataFrame, value_col: str, segments: list):
    segments_refined = expand_contract_segments(df, value_col, segments)
    segments_refined = classify_trends(df, value_col, segments_refined)
    segments_refined = shave_abrupt_trends(df, value_col, segments_refined)
    return segments_refined