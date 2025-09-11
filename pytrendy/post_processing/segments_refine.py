import pandas as pd
from copy import deepcopy
from ..simpledtw import dtw
from ..io.data_loader import load_data
import numpy as np

NEIGHBOUR_DISTANCE = 3  # Distance for considering a neighbour to readjust in expand_contract_segments 
GROUPING_DISTANCE = 7 # Distance for grouping segments of same type in group_segments

def _update_prev_segment(i, new_start, segments, segments_refined):
    """
    Shift previous segment end if overlapping with updated start (or original start).
    Don't touch neighbouring segment if an abrupt signal, it should remain precise and sensitive.
    """
    if (i == 0):
        return
    prev_has_class = 'trend_class' in segments_refined[i-1]
    if prev_has_class and segments_refined[i-1]['trend_class'] == 'abrupt':
        return

    distance_refined = (pd.to_datetime(new_start) - pd.to_datetime(segments_refined[i - 1]['end'])).days
    distance_orig = (pd.to_datetime(segments[i]['start']) - pd.to_datetime(segments[i - 1]['end'])).days
    if distance_refined <= NEIGHBOUR_DISTANCE or distance_orig <= NEIGHBOUR_DISTANCE:
        segments_refined[i - 1]['end'] = (pd.to_datetime(new_start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')


def _update_next_segment(i, new_end, segments, segments_refined):
    """Shift next segment start if overlapping with updated end (or original end)."""
    if (i == len(segments_refined) - 1):
        return
    next_has_class = 'trend_class' in segments_refined[i+1]
    if next_has_class and segments_refined[i+1]['trend_class'] == 'abrupt':
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

        if 'trend_class' in segment and segment['trend_class'] == 'abrupt':
            continue # don't expand/contract abrupt trends. Leave precise to shave.

        if segment['direction'] == 'Up':
            new_start = start_df[value_col].iloc[::-1].idxmin() + pd.Timedelta(days=1) # get min, latest if all same
            new_end = end_df[value_col].idxmax()
        elif segment['direction'] == 'Down':
            new_start = start_df[value_col].iloc[::-1].idxmax() + pd.Timedelta(days=1) # get max, latest if all same
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
        start = pd.to_datetime(segment['start']) - pd.Timedelta(days=2)
        end = pd.to_datetime(segment['end']) + pd.Timedelta(days=2)
        df_segment = df.loc[start:end].copy()

        # Use z-score on diff, to know when a change is an anomoly in the trend
        df_segment['diff'] = df_segment[value_col].diff()
        df_segment = df_segment.iloc[1:]
        df_segment['z_score'] = (df_segment['diff'] - df_segment['diff'].mean()) / df_segment['diff'].std()
        df_segment['abrupt_flag'] = 0
        df_segment.loc[df_segment['z_score'].abs() > 1, 'abrupt_flag'] = 1

        # Note: Follows very similar code to process signals 3.4. 
        df_segment['abrupt_flag_diff'] = df_segment['abrupt_flag'].diff()
        abrupt_starts = df_segment.loc[df_segment['abrupt_flag_diff'] == 1].index
        abrupt_ends = df_segment.loc[df_segment['abrupt_flag_diff'] == -1].index

        # Construct abrupt sub-segments list based on flag_diff
        abrupt_subsegs = []
        for abrupt_start in abrupt_starts: # Loops from first start onwards
            after_ends = [end for end in abrupt_ends if end > abrupt_start]

            # Get abrupt end as
            if len(after_ends) > 0:
                abrupt_end = after_ends[0]  # first if aligned
            elif abrupt_start == df.index[-1]: 
                abrupt_end = df.index[-1] # last if unaligned at end
            else:
                continue # neither if not connected

            abrupt_subsegs.append(dict(start=abrupt_start, end=abrupt_end))

        if len(abrupt_ends) > 0: # Adds abrupt end with no start if at beginning
            abrupt_end = abrupt_ends[0]
            early_starts = [start for start in abrupt_starts if start < abrupt_end]
            if len(early_starts) == 0:
                abrupt_start = df.index[0]
                abrupt_subsegs.insert(0, dict(start=abrupt_start, end=abrupt_end))

        import matplotlib.pyplot as plt
        ax = df_segment[[value_col, 'abrupt_flag']].plot(figsize=(20,3), secondary_y='abrupt_flag')
        ax.right_ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
        plt.show()

        # If in right direction shave out abrupt subsegs from abrupt segment & adjust neighbours.
        for j, abrupt_subseg in enumerate(abrupt_subsegs):

            print(abrupt_subseg)

            new_start = abrupt_subseg['start'] - pd.Timedelta(days=1)
            new_end = abrupt_subseg['end'] - pd.Timedelta(days=1)

            print(new_start, new_end)
            # display(df_segment[value_col])

            value_change = df_segment.loc[new_end, value_col] - df_segment.loc[new_start, value_col]
            print(df_segment.loc[new_start, value_col], df_segment.loc[new_end, value_col])
            direction = 'Up' if value_change > 0 else 'Down'

            if direction != segment['direction']:
                print(abrupt_subseg, value_change, direction, segment['direction'], 'Not Matched')
                continue

            if j == 0:

                print('Updating current')

                # Update current segment
                segments_refined[i]['start'] = new_start.strftime('%Y-%m-%d')
                _update_prev_segment(i, new_start, segments, segments_refined)
                
                segments_refined[i]['end'] = new_end.strftime('%Y-%m-%d')
                _update_next_segment(i, new_end, segments, segments_refined)

            elif j > 0:

                print('Wedging in a new one')
                
                # Wedge in a new segment between current and next (needed for edge case of many abrupt near eachother)
                new_index = i + j
                new_seg = segment.copy()
                segments_refined.insert(new_index, new_seg)

                # Update new segment
                segments_refined[new_index]['start'] = new_start.strftime('%Y-%m-%d')
                _update_prev_segment(new_index, new_start, segments, segments_refined)
                
                segments_refined[new_index]['end'] = new_end.strftime('%Y-%m-%d')
                _update_next_segment(new_index, new_end, segments, segments_refined)


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
            and 'trend_class' not in segment or ('trend_class' in segment and segment['trend_class'] == 'graduaul') # dont group up abrupt trends
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
        if (end - start).days < 1: 
            continue
        segments_refined.append(segment)
    return segments_refined


def refine_segments(df: pd.DataFrame, value_col: str, segments: list, method_params:dict):
    segments_refined = deepcopy(segments)
    segments_refined = classify_trends(df, value_col, segments_refined)
    segments_refined = shave_abrupt_trends(df, value_col, segments_refined, method_params) # for abrupt
    segments_refined = expand_contract_segments(df, value_col, segments_refined) # for gradual
    segments_refined = group_segments(segments_refined)
    segments_refined = clean_artifacts(segments_refined)

    return segments_refined