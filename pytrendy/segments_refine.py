import pandas as pd
from copy import deepcopy
from .simpledtw import dtw
from .data_loader import load_data
import numpy as np

NEIGHBOUR_DISTANCE = 3  # Distance for considering a neighbour to readjust in expand_contract_segments 
GROUPING_DISTANCE = 7 # Distance for grouping segments of same type in group_segments

def _update_prev_segment(i, new_start, segments, segments_refined):
    """Shift previous segment end if overlapping with updated start (or original start)."""
    if i == 0:
        return
    distance_refined = (pd.to_datetime(new_start) - pd.to_datetime(segments_refined[i - 1]['end'])).days
    distance_orig = (pd.to_datetime(segments[i]['start']) - pd.to_datetime(segments[i - 1]['end'])).days
    if distance_refined <= NEIGHBOUR_DISTANCE or distance_orig <= NEIGHBOUR_DISTANCE:
        segments_refined[i - 1]['end'] = (pd.to_datetime(new_start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')


def _update_next_segment(i, new_end, segments, segments_refined):
    """Shift next segment start if overlapping with updated end (or original end)."""
    if i == len(segments_refined) - 1:
        return
    distance_refined = (pd.to_datetime(segments_refined[i + 1]['start']) - pd.to_datetime(new_end)).days
    distance_orig = (pd.to_datetime(segments[i + 1]['start']) - pd.to_datetime(segments[i]['end'])).days
    if distance_refined <= NEIGHBOUR_DISTANCE or distance_orig <= NEIGHBOUR_DISTANCE:
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

        # Refine start
        if new_start != pd.to_datetime(segment['start']):
            segments_refined[i]['start'] = new_start.strftime('%Y-%m-%d')
            _update_prev_segment(i, new_start, segments, segments_refined)

        # Refine end
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

    df_class = load_data('classes_trends')
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


def shave_abrupt_trends(df: pd.DataFrame, value_col: str, segments: list, method_params: dict):
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

        # Use z-score on diff, to know when a change is an anomoly in the trend
        df_segment['diff'] = df_segment[value_col].diff()
        df_segment = df_segment.iloc[1:]
        df_segment['z_score'] = (df_segment['diff'] - df_segment['diff'].mean()) / df_segment['diff'].std()
        df_segment['abrupt_flag'] = 0
        df_segment.loc[df_segment['z_score'].abs() > 2, 'abrupt_flag'] = 1

        # Refine start
        new_start = df_segment.loc[df_segment['abrupt_flag'] == 1].index[0] - pd.Timedelta(days=1)
        segments_refined[i]['start'] = new_start.strftime('%Y-%m-%d')
        _update_prev_segment(i, new_start, segments, segments_refined)

        # Refine end, with custom logic for padding if specified
        new_end = df_segment.loc[df_segment['abrupt_flag'] == 1].index[-1]
        segments_refined[i]['end'] = new_end.strftime('%Y-%m-%d')
        _update_next_segment(i, new_end, segments, segments_refined)

    # Second pass to pad segments if specified
    segments_padded = deepcopy(segments_refined)
    if method_params.get('is_abrupt_padded', False):

        df = pd.DataFrame(segments_refined)
        df['start'] = pd.to_datetime(df['start'])
        df['end'] = pd.to_datetime(df['end'])

        for i, segment in enumerate(segments_refined):

            if segment['direction'] not in ['Up', 'Down'] or segment['trend_class'] != 'abrupt': 
                continue

            # Simulate new end with padding and any overlaps it might cayse
            print(method_params)
            new_end = pd.to_datetime(segment['end']) + pd.Timedelta(days=method_params['abrupt_padding'])
            overlaps = df.loc[(df['start'] <= new_end) & (df['end'] >= new_end)]
            overlaps_nonflats = overlaps[overlaps['direction']!='Flat']

            # Adjust padding to be before first nonflat segment that it would overlap
            if not overlaps.empty and not overlaps_nonflats.empty:
                first_notflat_overlap = overlaps_nonflats.iloc[0]
                new_end = pd.to_datetime(first_notflat_overlap['start']) - pd.Timedelta(days=1)

            segments_padded[i]['end'] = new_end.strftime('%Y-%m-%d')
            _update_next_segment(i, new_end, segments_refined, segments_padded) # will always be a flat it adjusts/overwrites

    return segments_padded


def group_segments(segments):
    """
    Groups segments if they have the same direction AND their gap is <= GROUPING_DISTANCE.
    This reduces noisy selections from sporadic short segments.
    """
    def flush_history(segment_history, output):
        """Append either a single or grouped segment to output."""
        if not segment_history:
            return
        if len(segment_history) == 1:
            output.append(segment_history[0])
        else:
            first, last = segment_history[0], segment_history[-1]
            grouped = last.copy()
            grouped['start'] = first['start']
            grouped['end'] = last['end']
            grouped['segment_length'] = (
                pd.to_datetime(last['end']) - pd.to_datetime(first['start'])
            ).days
            output.append(grouped)

    segments_refined = []
    segment_history = []
    direction_prev = None

    for segment in segments:
        direction = segment['direction']

        if (
            direction == direction_prev
            and segment_history
            and (pd.to_datetime(segment['start']) - pd.to_datetime(segment_history[-1]['end'])).days <= GROUPING_DISTANCE
        ):
            # same direction and within allowed distance -> extend history
            segment_history.append(segment)
        else:
            # flush current history before starting a new group
            flush_history(segment_history, segments_refined)
            segment_history = [segment]

        direction_prev = direction

    # flush any remaining history
    flush_history(segment_history, segments_refined)

    return segments_refined


def clean_artifacts(segments):
    """
    Sometimes the neighbour repositioning can create tiny artifacts (eg. for Flats)
    Cleaning to make sure it does not make its way to final indication
    """
    segments_refined = []
    for segment in segments:
        start = pd.to_datetime(segment['start'])
        end =  pd.to_datetime(segment['end'])
        if (end - start).days < 1: # must align with constants in segments_get
            continue
        segments_refined.append(segment)
        
    return segments_refined


def refine_segments(df: pd.DataFrame, value_col: str, segments: list, method_params:dict):
    segments_refined = expand_contract_segments(df, value_col, segments)
    segments_refined = classify_trends(df, value_col, segments_refined)
    segments_refined = shave_abrupt_trends(df, value_col, segments_refined, method_params)
    segments_refined = group_segments(segments_refined)

    segments_refined = clean_artifacts(segments_refined)
    return segments_refined