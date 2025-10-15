"""**Adjust Boundaries and Classify Trends**"""

import pandas as pd
from copy import deepcopy
from ..simpledtw import dtw
from ..io.data_loader import load_data
import numpy as np

NEIGHBOUR_DISTANCE = 3  # Distance for considering a neighbour to re-adjust after expand_contract or shave logic
GROUPING_DISTANCE = 7 # Distance for grouping segments of same type in group_segments

def update_prev_segment(i, new_start, segments, segments_refined, df, value_col):
    """
    Adjusts the end of the previous segment if it overlaps with the updated start.

    Skips adjustment if the previous segment is classified as 'abrupt' or 'noise', preserving its precision.
    
    Args:
        i (int): Index of the current segment.
        new_start (str): Updated start date of the current segment.
        segments (list): Original segment list.
        segments_refined (list): Refined segment list being modified.
        df (pd.DataFrame): Main input df with value_col and time index.
        value_col (str): str column of value column in df.
    """

    if (i == 0): return
    old_start = pd.to_datetime(segments[i]['start'])
    prev_segments = reversed(segments_refined[:i])

    for j, prevseg in enumerate(prev_segments):
        prev_start = pd.to_datetime(prevseg['start'])
        prev_end = pd.to_datetime(prevseg['end'])
        i_neighbour = i - (j+1)

        # Edge case 1.1: do not disturb abrupt trends (leave precise)
        if ('trend_class' in prevseg and prevseg['trend_class'] == 'abrupt'):
            continue

        # # Edge case 1.2: do not disturb noise spikes (leave precise)
        if (prevseg['direction'] == 'Noise'):
            diff = abs(df.loc[prev_end, value_col] - df.loc[prev_start, value_col])
            small_value = df.loc[df[value_col] > 0, value_col].quantile(0.05)
            if diff <= small_value:  # only skip if a noise that spikes
                continue

        # Edge case 2: swallow neighbours that get fully overlapped.
        if prev_start >= new_start and prev_start <= old_start:
            segments_refined[i_neighbour]['end'] = new_start - pd.Timedelta(days=1)
            continue

        # Update when a valid neighbour of close enough distance.
        new_dist = (new_start - prev_end).days
        old_dist = (old_start - prev_end).days
        is_neighbour = (new_dist <= NEIGHBOUR_DISTANCE) or (old_dist <= NEIGHBOUR_DISTANCE)
        if is_neighbour:
            neighbour_end = (new_start - pd.Timedelta(days=1))
            segments_refined[i_neighbour]['end'] = neighbour_end.strftime('%Y-%m-%d')
            return
        

def update_next_segment(i, new_end, segments, segments_refined, df, value_col):
    """
    Adjusts the start of the next segment if it overlaps with the updated end.

    Skips adjustment if the next segment is classified as 'abrupt', preserving its precision.

    Args:
        i (int): Index of the current segment.
        new_end (str): Updated end date of the current segment.
        segments (list): Original segment list.
        segments_refined (list): Refined segment list being modified.
        df (pd.DataFrame): Main input df with value_col and time index.
        value_col (str): str column of value column in df.
    """
    if (i == len(segments) - 1): return
    old_end = pd.to_datetime(segments[i]['end'])
    next_segments = segments_refined[i+1:]

    for j, nextseg in enumerate(next_segments):
        next_start = pd.to_datetime(nextseg['start'])
        next_end = pd.to_datetime(nextseg['end'])
        i_neighbour = i + (j+1)

        # Edge case 1: do not disturb abrupt trends (leave precise)
        if ('trend_class' in nextseg and nextseg['trend_class'] == 'abrupt'):
            continue

        # Edge case 1.2: do not disturb noise spikes (leave precise)
        if (nextseg['direction'] == 'Noise'):
            diff = abs(df.loc[next_end, value_col] - df.loc[next_start, value_col])
            small_value = df.loc[df[value_col] > 0, value_col].quantile(0.05)
            if diff <= small_value: # only skip if a noise that spikes
                continue

        # Edge case 2: swallow neighbours that get fully overlapped.
        if next_end >= old_end and next_end <= new_end:
            segments_refined[i_neighbour]['start'] = new_end + pd.Timedelta(days=1)
            continue

        # Update when a valid neighbour of close enough distance.
        new_dist = (next_start - new_end).days
        old_dist = (next_start - old_end).days
        is_neighbour = (new_dist <= NEIGHBOUR_DISTANCE) or (old_dist <= NEIGHBOUR_DISTANCE)
        if is_neighbour:
            segments_refined[i_neighbour]['start'] = (new_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            return


def expand_contract_segments(df: pd.DataFrame, value_col: str, segments: list):
    """
    Refines segment boundaries by expanding or contracting based on local extrema.

    Examines ±7 days around each segment's start and end to find stronger turning points.
    Skips segments classified as 'abrupt' to preserve their precision.

    Args:
        df (pd.DataFrame): Time series DataFrame.
        value_col (str): Name of the signal column.
        segments (list): List of segment dictionaries.

    Returns:
        list: Refined segment list with updated boundaries.
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
        start_changed = new_start != pd.to_datetime(segment['start'])
        if start_changed:
            segments_refined[i]['start'] = new_start.strftime('%Y-%m-%d')
            update_prev_segment(i, new_start, segments, segments_refined, df, value_col)

        # Refine end
        end_changed = new_end != pd.to_datetime(segment['end'])
        if end_changed:
            segments_refined[i]['end'] = new_end.strftime('%Y-%m-%d')
            update_next_segment(i, new_end, segments, segments_refined, df, value_col)

    return segments_refined


def classify_trends(df: pd.DataFrame, value_col: str, segments: list):
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


def shave_abrupt_trends(df: pd.DataFrame, value_col: str, segments: list, method_params: dict, second_pass: bool = False, init_segments: list= []):
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

    segments_refined = deepcopy(segments)

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
                update_prev_segment(i, new_start, segments, segments_refined, df, value_col)
                
                segments_refined[i]['end'] = new_end.strftime('%Y-%m-%d')
                update_next_segment(i, new_end, segments, segments_refined, df, value_col)

            elif j > 0:
                
                # Wedge in a new segment between current and next (needed for edge case of many abrupt near eachother)
                new_index = i + j
                new_seg = segment.copy()
                segments_refined.insert(new_index, new_seg)
                segments.insert(new_index, new_seg)

                # Update new segment
                segments_refined[new_index]['start'] = new_start.strftime('%Y-%m-%d')
                update_prev_segment(new_index, new_start, segments, segments_refined, df, value_col)
                
                segments_refined[new_index]['end'] = new_end.strftime('%Y-%m-%d')
                update_next_segment(new_index, new_end, segments, segments_refined, df, value_col)


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

            segments_padded[i]['end'] = new_end.strftime('%Y-%m-%d')
            update_next_segment(i, new_end, segments_refined, segments_padded, df, value_col) # will always be a flat it adjusts/overwrites

    return segments_padded


def group_segments(segments: list):
    """
    Groups consecutive segments with the same direction if their gap is small.

    Segments are grouped if:

        - They share the same `'direction'`
        - Their gap is ≤ `GROUPING_DISTANCE`
        - They are not classified as `'abrupt'`

    This reduces fragmentation caused by short, noisy segments.

    Args:
        segments (list): List of segment dictionaries.

    Returns:
        list: Grouped segment list.
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
            and ((not 'trend_class' in segment) or ('trend_class' in segment and segment['trend_class'] != 'abrupt')) # dont group up abrupt trends
        ):
            # same direction and within allowed distance -> extend history
            segment_history.append(segment)
        elif (
            direction == direction_prev
            and segment_history
            and (('trend_class' in segment and segment['trend_class'] == 'abrupt')) 
            and (pd.to_datetime(segment['start']) - pd.to_datetime(segment_history[-1]['end'])).days <= 1
        ):
            # same direction and within tight allowed distance for abrupt -> extend history
            segment_history.append(segment)
        else:
            # flush current history before starting a new group
            flush_history(segment_history, segments_refined)
            segment_history = [segment]

        direction_prev = direction

    # flush any remaining history
    flush_history(segment_history, segments_refined)

    return segments_refined


def clean_artifacts(df: pd.DataFrame, value_col:str, segments:list):
    """
    Removes segments that are too short to be meaningful.
    
    This is typically used to clean up artifacts introduced by boundary adjustments.

    Args:
        segments (list): List of segment dictionaries.

    Returns:
        list: Cleaned segment list with only valid-length segments.
    """

    def has_inverse(df, value_col, segment):
        """
        Checks that if end moved before start from neighbour adjustment, removes artifact.
        Also if trend, but total_change is actually in opposing direction, also remove
        """
        start = pd.to_datetime(segment['start'])
        end =  pd.to_datetime(segment['end'])
        if (end - start).days < 1: # inverse if start before end
            return True

        total_change = df.loc[start:end, value_col].diff().sum()
        
        # inverse if tagged direction does not match total change
        if \
            (segment['direction'] == 'Up' and total_change < 0) or \
            (segment['direction'] == 'Down' and total_change >= 0) or \
            (segment['direction'] in ('Up', 'Down') and abs(total_change) <= df.loc[df[value_col] > 0, value_col].quantile(0.05)): # or too flat
            return True
        return False

    def has_overlap_next(segment, segment_next):
        """Checks whether overlap exists between curr & next, and current is more insignificant"""
        dir = segment['direction']
        start =  pd.to_datetime(segment['start'])
        end =  pd.to_datetime(segment['end'])
        width = (end - start).days

        next_dir = segment_next['direction']
        next_start = pd.to_datetime(segment_next['start'])
        next_end = pd.to_datetime(segment_next['end'])
        next_width = (next_end - next_start).days

        # Define conditions
        is_overlap_next = (end >= next_start)
        is_same_dir = (dir == next_dir)
        is_curr_shorter = (width <= next_width)

        is_trend = (dir in ('Up', 'Down'))
        is_next_noise = (next_dir == 'Noise')

        is_next_gradual = ('trend_class' in segment_next and segment_next['trend_class'] == 'gradual')
        is_next_abrupt = ('trend_class' in segment_next and segment_next['trend_class'] == 'abrupt')

        # Trigger edge cases of overlap if satisfied
        if is_overlap_next and is_same_dir: # and not is_trend and is_curr_shorter:
            return True # overlap when same direction, not trend, and curr is shorter
        if is_overlap_next and (is_trend and is_next_noise and is_curr_shorter):
            return True # overlap when curr is trend and next is noise of larger window
        if is_overlap_next and is_same_dir and (is_next_gradual and is_curr_shorter):
            return True # overlap when next is also gradual but larger
        if is_overlap_next and is_same_dir and (is_next_abrupt and not is_curr_shorter):
            return True  # overlap when next is also abrupt but shorter

        return False
    
    def has_overlap_prev(segment, segment_prev):
        """Light checks with overlaps on previous, that wouldnt already be covered by has_overlap_next"""
        dir = segment['direction']
        start =  pd.to_datetime(segment['start'])
        end =  pd.to_datetime(segment['end'])
        width = (end - start).days

        prev_dir = segment_prev['direction']
        prev_start = pd.to_datetime(segment_prev['start'])
        prev_end = pd.to_datetime(segment_prev['end'])
        prev_width = (prev_end - prev_start).days

        # Define conditions
        is_overlap_prev = (start <= prev_end)
        is_curr_shorter = (width <= prev_width)
        is_trend = (dir in ('Up', 'Down'))
        is_prev_noise = (prev_dir == 'Noise')

        if is_overlap_prev and (is_trend and is_prev_noise and is_curr_shorter):
            return True # overlap when curr is trend and prev is noise of larger/equal window
        return False
    
    def has_partial_overlap_next(segment, segment_next):
        """Checks whether overlap exists between curr & next, and current is more insignificant"""
        dir = segment['direction']
        start =  pd.to_datetime(segment['start'])
        end =  pd.to_datetime(segment['end'])
        width = (end - start).days

        next_dir = segment_next['direction']
        next_start = pd.to_datetime(segment_next['start'])
        next_end = pd.to_datetime(segment_next['end'])
        next_width = (next_end - next_start).days

        # Define conditions
        is_overlap_next = (end >= next_start)
        is_curr_shorter = (width <= next_width)
        is_next_noise = (next_dir == 'Noise') 
        is_trend_or_flat = (dir in ('Up', 'Down', 'Flat'))
        is_next_abrupt = ('trend_class' in segment_next and segment_next['trend_class'] == 'abrupt')

        if is_overlap_next and (is_trend_or_flat and (is_next_noise or is_next_abrupt) and not is_curr_shorter):
            return True # overlap when curr is trend and next is noise of larger window

        return False
    
    def has_partial_overlap_prev(segment, segment_prev):
        """Light checks with overlaps on previous, that wouldnt already be covered by has_overlap_next"""
        dir = segment['direction']
        start =  pd.to_datetime(segment['start'])
        end =  pd.to_datetime(segment['end'])
        width = (end - start).days

        prev_dir = segment_prev['direction']
        prev_start = pd.to_datetime(segment_prev['start'])
        prev_end = pd.to_datetime(segment_prev['end'])
        prev_width = (prev_end - prev_start).days

        # Define conditions
        is_overlap_prev = (start <= prev_end)
        is_curr_shorter = (width <= prev_width)
        is_prev_noise = (prev_dir == 'Noise')
        is_trend_or_flat = (dir in ('Up', 'Down', 'Flat'))
        is_prev_abrupt = ('trend_class' in segment_prev and segment_prev['trend_class'] == 'abrupt')

        if is_overlap_prev and (is_trend_or_flat and (is_prev_noise or is_prev_abrupt) and not is_curr_shorter):
            return True # overlap when curr is trend and prev is noise of larger/equal window
        return False

    # Pass 1: Cleans inverse length segments. Artifacts from expansion/contraction
    segments_refined = []
    for i, segment in enumerate(segments):
        if has_inverse(df, value_col, segment): 
            continue # Excludes segment.
        segments_refined.append(segment)

    # Pass 2: Cleans overlaps of same direction. Also artifacts from expansion/contraction & noise detection
    segments = deepcopy(segments_refined)
    segments_refined = [] 
    for i, segment in enumerate(segments):
        if (i < len(segments)-1 and has_overlap_next(segment, segments[i+1])) or \
            (i > 0 and has_overlap_prev(segment, segments[i-1])): 
            continue 
        segments_refined.append(segment)

    # Pass 3: Cleans partial overlaps with noise. Don't filter out completely when partial, adjust outside noise
    segments = deepcopy(segments_refined)
    segments_refined = [] 
    for i, segment in enumerate(segments):
        if (i < len(segments)-1 and has_partial_overlap_next(segment, segments[i+1])):

            shifted_end = (pd.to_datetime(segments[i+1]['start']) - pd.Timedelta(days=1))
            end_df = df.loc[segment['start']:shifted_end]

            # when gradual, follows similar logic to expand/contract seleciton.
            if segments[i]['direction'] == 'Up':
                new_end = end_df[value_col].idxmax()
                segments[i]['end'] = new_end.strftime('%Y-%m-%d')
            
            if segments[i]['direction'] == 'Down':
                new_end = end_df[value_col].idxmin()
                segments[i]['end'] = new_end.strftime('%Y-%m-%d')

            elif segments[i]['direction'] == 'Flat':
                segments[i]['end'] = shifted_end.strftime('%Y-%m-%d')

        if (i > 0 and has_partial_overlap_prev(segment, segments[i-1])): 

            shifted_start = (pd.to_datetime(segments[i-1]['end']) + pd.Timedelta(days=1))
            start_df = df.loc[shifted_start:segment['end']]
            
            # when gradual, follows similar logic to expand/contract seleciton.
            if segments[i]['direction'] == 'Up':
                new_start = start_df[value_col].iloc[::-1].idxmin() + pd.Timedelta(days=1)
                segments[i]['start'] = new_start.strftime('%Y-%m-%d')

            if segments[i]['direction'] == 'Down':
                new_start = start_df[value_col].iloc[::-1].idxmax() + pd.Timedelta(days=1)
                segments[i]['start'] = new_start.strftime('%Y-%m-%d') 

            elif segments[i]['direction'] == 'Flat':
                segments[i]['start'] = shifted_start.strftime('%Y-%m-%d')

        segments_refined.append(segment)

    return segments_refined


def fill_in_flats(segments:list):
    """Assumes remaining gaps between segments are flats (after post-processing). Fills them in."""
    segments_refined = segments.copy()
    j = 0
    for i, curr_seg in enumerate(segments):
        if i >= (len(segments) - 1): continue # skip if cant see next
        next_seg = segments[i+1]

        start = pd.to_datetime(curr_seg['end']) + pd.Timedelta(days=1)
        end = pd.to_datetime(next_seg['start']) - pd.Timedelta(days=1)
        days = (end - start).days

        # Trigger fill in when not exactly neighbouring
        if days >= 0:
            new_flat = dict(
                start = start.strftime('%Y-%m-%d'),
                end = end.strftime('%Y-%m-%d'),
                direction='Flat'
            )
            j += 1 # iterate segments vs segments_refined displacement
            segments_refined.insert(i+j, new_flat)
    return segments_refined


def refine_segments(df: pd.DataFrame, value_col: str, segments: list, method_params:dict):
    """
    Full post-processing pipeline to refine detected trend segments.

    This function applies:
    - Trend classification (`gradual` vs `abrupt`)
    - Abrupt changepoint shaving
    - Gradual boundary expansion/contraction
    - Segment grouping
    - Artifact cleanup

    Args:
        df (pd.DataFrame): Time series DataFrame.
        value_col (str): Name of the signal column.
        segments (list): Initial segment list from detection.
        method_params (dict): Optional parameters for abrupt padding and control.

    Returns:
        list: Final refined segment list.
    """

    segments_refined = deepcopy(segments)
    
    segments_refined = classify_trends(df, value_col, segments_refined)
    segments_refined = group_segments(segments_refined) # grouping 1st pass: sporadic flats & noises

    segments_refined = expand_contract_segments(df, value_col, segments_refined) # for gradual
    segments_refined = shave_abrupt_trends(df, value_col, segments_refined, method_params) # for abrupt

    segments_refined = clean_artifacts(df, value_col, segments_refined) # cleans overlaps etc from expand/contract
    segments_refined = group_segments(segments_refined) # grouping 2nd pass: after trend refine and cleanup

    init_segments = deepcopy(segments_refined)
    segments_refined = classify_trends(df, value_col, segments_refined) # reclassify after artifacts cleaned: some graduals to abrupt
    if segments_refined != init_segments: # only trigger if any re-classifications
        segments_refined = shave_abrupt_trends(df, value_col, segments_refined, method_params
                                            , second_pass=True, init_segments=init_segments) # abrupt shave 2nd pass: newly converted abrupts 
        segments_refined = clean_artifacts(df, value_col, segments_refined) # cleans overlaps etc from shave abrupt (precaution even though second_pass=True handles this)

    segments_refined = fill_in_flats(segments_refined) # fill in flats in case there are gaps (assume remaining gaps are appropriately flats)
    segments_refined = group_segments(segments_refined) # grouping 3rd pass (final): after abrupt shave 2nd pass and/or flat fill in
    return segments_refined