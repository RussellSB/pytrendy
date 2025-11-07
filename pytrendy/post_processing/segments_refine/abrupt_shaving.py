"""**Abrupt Trend Handling Utilities**

Functions for refining abrupt segments by detecting changepoints using z-score outliers.
"""

import pandas as pd
from copy import deepcopy
from .update_neighbours import update_prev_segment, update_next_segment


def shave_abrupt_trends(df: pd.DataFrame, value_col: str, segments: list[dict], method_params: dict, second_pass: bool = False, init_segments: list[dict] | None = None) -> list[dict]:
    """
    Refines abrupt segments by detecting changepoints using z-score outliers.

    This function identifies sharp transitions missed by rolling statistics and
    adjusts segment boundaries based on statistical anomalies in the signal's first differences.
    It also supports multi-abrupt detection within a segment and optional padding to extend abrupt ends.

    Args:
        df (pd.DataFrame): Time series DataFrame.
        value_col (str): Name of the signal column.
        segments (list): List of segment dictionaries with `'trend_class': 'abrupt'`.
        method_params (dict): Optional parameters for padding and control:
            - `'is_abrupt_padded'` (bool): Whether to pad abrupt segments.
            - `'abrupt_padding'` (int): Number of days to pad.

    Returns:
        list: Refined segment list with adjusted abrupt boundaries.
    """

    if init_segments is None:
        init_segments = []
    
    segments_refined = deepcopy(segments)
    new_segments = []
    for i, segment in enumerate(segments_refined):
        if segment['direction'] not in ['Up', 'Down'] or segment['trend_class'] != 'abrupt': 
            continue

        if second_pass:
            init_segment = init_segments[i]
            is_not_prev_trend = 'trend_class' not in init_segment # edge case, in case not trend before
            is_not_reclassified = is_not_prev_trend or segment['trend_class'] == init_segment['trend_class']
            if is_not_reclassified:
                continue # exit if not re-classified for sake of second pass

        # Get start end padded for some leniency
        start = pd.to_datetime(segment['start']) - pd.Timedelta(days=2)
        end = pd.to_datetime(segment['end']) + pd.Timedelta(days=2)
        df_segment = df.loc[start:end].copy()

        # Use z-score on diff, to know when a change is an anomoly in the trend
        df_segment['diff'] = df_segment[value_col].diff()
        df_segment = df_segment.iloc[1:]
        df_segment['z_score'] = (df_segment['diff'] - df_segment['diff'].mean()) / df_segment['diff'].std()
        df_segment['abrupt_flag'] = 0
        df_segment.loc[(df_segment['z_score'].abs() > 1), 'abrupt_flag'] = 1

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
                abrupt_end = min(abrupt_start + pd.Timedelta(days=1), df.index[-1])
            else:
                continue # neither if not connected

            abrupt_subsegs.append(dict(start=abrupt_start, end=abrupt_end))

        if len(abrupt_ends) > 0: # Adds abrupt end with no start if at beginning
            abrupt_end = abrupt_ends[0]
            early_starts = [start for start in abrupt_starts if start < abrupt_end]
            if len(early_starts) == 0:
                abrupt_start = max(abrupt_end - pd.Timedelta(days=1), df.index[0])
                abrupt_subsegs.insert(0, dict(start=abrupt_start, end=abrupt_end))

        # If in right direction shave out abrupt subsegs from abrupt segment & adjust neighbours.
        for j, abrupt_subseg in enumerate(abrupt_subsegs):
            new_start = abrupt_subseg['start'] - pd.Timedelta(days=1)
            new_end = abrupt_subseg['end'] - pd.Timedelta(days=1)

            start_value = df.loc[new_start, value_col] # referencing df, in case outside df_segment scope
            end_value = df.loc[new_end, value_col]
            value_change = end_value - start_value

            direction = 'Up' if value_change > 0 else 'Down'

            if direction != segment['direction']:
                continue

            if j == 0:
                # Update current segment
                segments_refined[i]['start'] = new_start.strftime('%Y-%m-%d')
                update_prev_segment(i, new_start, segments, segments_refined)

                segments_refined[i]['end'] = new_end.strftime('%Y-%m-%d')
                update_next_segment(i, new_end, segments, segments_refined)

            elif j > 0:
                # Wedge in a new segment between current and next (needed for edge case of many abrupt near each other)
                new_seg = segment.copy()
                new_seg['start'] = new_start.strftime('%Y-%m-%d')
                new_seg['end'] = new_end.strftime('%Y-%m-%d')
                new_segments.append((i, new_seg))  # Store with reference index

    # Add to main segments list, then sort.
    for offset, (base_index, new_seg) in enumerate(new_segments):
        insert_index = base_index + offset + 1
        segments_refined.insert(insert_index, new_seg)
        segments.insert(insert_index, new_seg)
        update_prev_segment(insert_index, pd.to_datetime(new_seg['start']), segments, segments_refined)
        update_next_segment(insert_index, pd.to_datetime(new_seg['end']), segments, segments_refined)
    segments_refined = sorted(segments_refined, key=lambda seg: pd.to_datetime(seg['start']))

    # Second pass to pad segments if specified
    segments_padded = deepcopy(segments_refined)
    if method_params.get('is_abrupt_padded', False) == True:

        meta_df = pd.DataFrame(segments_refined) # metadata df, to filter by datetime easily
        meta_df['start'] = pd.to_datetime(meta_df['start'])
        meta_df['end'] = pd.to_datetime(meta_df['end'])

        for i, segment in enumerate(segments_refined):

            if segment['direction'] not in ['Up', 'Down'] or segment['trend_class'] != 'abrupt': 
                continue

            abrupt_start = pd.to_datetime(segment['start'])
            abrupt_end = pd.to_datetime(segment['end'])

            # Simulate new end with padding and cater for any overlaps it might cause
            new_end = abrupt_end + pd.Timedelta(days=method_params['abrupt_padding'])
            overlaps = meta_df.loc[(meta_df['start'] > abrupt_end) & (meta_df['start'] <= new_end)]
            overlaps_nonflats = overlaps[overlaps['direction']!='Flat']

            # Adjust padding to be before first nonflat segment that it would overlap
            if not overlaps_nonflats.empty:
                first_notflat_overlap = overlaps_nonflats.iloc[0]
                new_end = pd.to_datetime(first_notflat_overlap['start']) - pd.Timedelta(days=1)

            new_end = min(new_end, df.index[-1]) # make sure doesnt go out of bounds
            segments_padded[i]['end'] = new_end.strftime('%Y-%m-%d')
            update_next_segment(i, new_end, segments_refined, segments_padded) # will always be a flat it adjusts/overwrites

            # Store meta data that got padded & stretched out
            segments_padded[i]['padded'] = True if new_end != abrupt_end else False

    return segments_padded
