"""**Artifact Cleanup Utilities**

Functions for removing invalid segments and filling in gaps with flat segments.
"""

import pandas as pd
import numpy as np
from copy import deepcopy
from .segment_grouping import GROUPING_DISTANCE


def clean_artifacts(df: pd.DataFrame, value_col: str, segments_refined: list[dict], method_params: dict) -> list[dict]:
    """
    Removes segments any invalid segments, such as inversions or overlaps.
    Typically to clean up after boundary adjustments introduced from noise or trend refinements.

    Args:
        segments_refined (list): List of segment dictionaries potentially with artifacts from post-processing.
        method_params (dict): Referenced to check is_abrupt_padded. If it is, dont check for neighbouring noise to abrupt.

    Returns:
        list: Cleaned segment list with only valid-length segments.
    """

    def has_inverse(df: pd.DataFrame, value_col: str, segment: dict) -> bool:
        """
        Checks that if end moved before start from neighbour adjustment, removes artifact.
        Also if trend, but total_change is actually in opposing direction, also remove
        """
        start = pd.to_datetime(segment['start'])
        end =  pd.to_datetime(segment['end'])
        if (end - start).days < 1: # inverse if start before end
            return True

        # inverse if tagged direction does not match total change
        total_change = df.loc[start:end, value_col].diff().sum()
        if \
            (segment['direction'] == 'Up' and total_change <= 0) or \
            (segment['direction'] == 'Down' and total_change >= 0):
            return True
        return False

    def has_overlap_next(segment: dict, segment_next: dict) -> bool:
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
        is_curr_similar = (next_width <= 1.5 * width) and (next_width >= 0.5 * width)

        is_trend = (dir in ('Up', 'Down'))
        is_next_noise = (next_dir == 'Noise')
        is_next_opposite_trend = (next_dir in ('Up', 'Down') and next_dir != dir)
        is_next_flat = (next_dir == 'Flat')

        is_next_gradual = ('trend_class' in segment_next and segment_next['trend_class'] == 'gradual')
        is_next_abrupt = ('trend_class' in segment_next and segment_next['trend_class'] == 'abrupt')

        # Trigger edge cases of overlap if satisfied
        if is_overlap_next and is_same_dir: # and not is_trend and is_curr_shorter:
            return True # overlap when same direction, not trend, and curr is shorter
        if is_overlap_next and (is_trend and (is_next_noise or is_next_opposite_trend) and is_curr_shorter):
            return True # overlap when curr is trend and next is noise of larger window
        if is_overlap_next and (is_trend and is_next_flat) and is_curr_similar:
            return True # overlap when curr is trend and next is flat (with similar enough size)
        if is_overlap_next and is_same_dir and (is_next_gradual and is_curr_shorter):
            return True # overlap when next is also gradual but larger
        if is_overlap_next and is_same_dir and (is_next_abrupt and not is_curr_shorter):
            return True  # overlap when next is also abrupt but shorter

        return False
    
    def has_overlap_prev(segment: dict, segment_prev: dict) -> bool:
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
        is_curr_similar = (prev_width <= 1.5 * width) and (prev_width >= 0.5 * width)

        is_trend = (dir in ('Up', 'Down'))
        is_prev_noise = (prev_dir == 'Noise')
        is_prev_opposite_trend = (prev_dir in ('Up', 'Down') and prev_dir != dir)
        is_prev_flat = (prev_dir == 'Flat')

        if is_overlap_prev and (is_trend and (is_prev_noise or is_prev_opposite_trend) and is_curr_shorter):
            return True # overlap when curr is trend and prev is noise of larger/equal window
        if is_overlap_prev and (is_trend and is_prev_flat) and is_curr_similar:
            return True # overlap when curr is trend and prev is flat (with similar enough size)
        return False
    
    def has_partial_overlap_next(segment: dict, segment_next: dict) -> bool:
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
    
    def has_partial_overlap_prev(segment: dict, segment_prev: dict) -> bool:
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

    # Pass 1: Cleans inverse length segments in case any artifacts from expand/contract and abrupt shave logic
    segments = deepcopy(segments_refined)
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
            start = pd.to_datetime(segment['start'])
            is_inverted = (shifted_end < start) # In case noise segment is <= 1 day in length
            if is_inverted: 
                continue

            # when gradual, follows similar logic to expand/contract selection.
            end_df = df.loc[start:shifted_end]
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
            end = pd.to_datetime(segment['end'])
            is_inverted = (end < shifted_start) # In case noise segment is <= 1 day in length
            if is_inverted: 
                continue
            
            # when gradual, follows similar logic to expand/contract selection.
            start_df = df.loc[shifted_start:end]

            if segments[i]['direction'] == 'Up':
                new_start = start_df[value_col].iloc[::-1].idxmin() + pd.Timedelta(days=1)
                segments[i]['start'] = new_start.strftime('%Y-%m-%d')

            if segments[i]['direction'] == 'Down':
                new_start = start_df[value_col].iloc[::-1].idxmax() + pd.Timedelta(days=1)
                segments[i]['start'] = new_start.strftime('%Y-%m-%d') 

            elif segments[i]['direction'] == 'Flat':
                segments[i]['start'] = shifted_start.strftime('%Y-%m-%d')

        segments_refined.append(segment)

    # Pass 4: Cleans inverse AGAIN: in case any artifacts from overlap adjustments
    segments = deepcopy(segments_refined)
    segments_refined = []
    for i, segment in enumerate(segments):
        if has_inverse(df, value_col, segment): 
            continue # Excludes segment.
        segments_refined.append(segment)

    # Pass 5: 
    # - Sets trends to noise when they have too low an SNR, too susceptible to noise, or not trendy enough
    # - Sets trends to flat when too flat.
    segments = deepcopy(segments_refined)
    segments_refined = [] 
    for i, segment in enumerate(segments):
        start = pd.to_datetime(segment['start'])
        end = pd.to_datetime(segment['end'])
        df_segment = df.loc[start:end].copy()

        # Conditions for edge cases
        left_is_noise = any(( # Consider segments within neighbour distance on left
                0 <= (start - pd.to_datetime(prev_seg['end'])).days <= GROUPING_DISTANCE
                and prev_seg.get('direction') == 'Noise'
            ) for k, prev_seg in enumerate(segments) if k != i)
        right_is_noise = any(( # Consider segments within neighbour distance on right
                0 <= (pd.to_datetime(next_seg['start']) - end).days <= GROUPING_DISTANCE
                and next_seg.get('direction') == 'Noise'
            ) for k, next_seg in enumerate(segments) if k != i)
        
        is_flat = segment['direction'] == 'Flat'
        is_gradual = ('trend_class' in segment and segment['trend_class'] == 'gradual')
        is_abrupt = ('trend_class' in segment and segment['trend_class'] == 'abrupt')
        is_padded = is_abrupt and ('padded' in segment) and (segment['padded'] == True)
        is_small = len(df_segment) <= 5

        # Edge case 1: Check SNR for trend but noise
        signal_power = np.mean(df_segment['signal']**2)
        noise_power = np.mean(df_segment['noise']**2)
        snr = float(10 * np.log10(signal_power / noise_power)) if noise_power != 0 else np.nan
        threshold_noise = 2.5 
        if is_gradual: threshold_noise = 5
        if is_flat: threshold_noise = 0
        too_noisy = (snr < threshold_noise)

        # Edge case 2.1: Check if abrupt segment near noise
        is_abrupt_near_noise = is_abrupt and (left_is_noise or right_is_noise)
        if is_padded: is_abrupt_near_noise = False # overwrite to False if segment got abrupt padded
        
        # Edge case 2.2: Check if gradual segment encapsulated by noise
        is_small_gradual_in_noise = is_gradual and (left_is_noise and right_is_noise) and is_small

        # Edge case 3.1: Check if value of end is too close to value of start
        value_start = df.loc[start, value_col]
        value_end = df.loc[end, value_col]
        diff = abs(value_end - value_start)
        threshold_diff = float(df['value_cleaned'].abs().max()) * 0.01
        trend_ends_too_close = (is_gradual or is_abrupt) and (diff <= threshold_diff)

        # Edge case 3.2: Check if total change too small, because noise puts it closer to 0
        total_change = abs(df_segment[value_col].diff().sum())
        threshold_diff = float(df['value_cleaned'].abs().max()) * 0.01
        trend_too_small = (is_gradual or is_abrupt) and (total_change <= threshold_diff)

        # Edge case 3.3: If max is not at end, or min is not at end for Up/Down trends - too flat for trend, consider as noise
        trend_too_flat = False
        if is_gradual and len(df_segment) >= 3:
            # Allow max/min to be in the last 30% of the segment instead of only at end
            segment_length = len(df_segment)
            last_30pct_start = int(segment_length * 0.7)
            last_section = df_segment.iloc[last_30pct_start:]
            
            if segment['direction'] == 'Up':
                max_date = df_segment[value_col].idxmax()
                max_in_last_section = (max_date in last_section.index)
                trend_too_flat = not max_in_last_section
                
            elif segment['direction'] == 'Down':
                min_date = df_segment[value_col].idxmin()
                min_in_last_section = (min_date in last_section.index)
                trend_too_flat = not min_in_last_section

        # Reclassify as noise if either edge cases met
        if too_noisy or is_abrupt_near_noise or is_small_gradual_in_noise:
            segment['direction'] = 'Noise' 
            if 'trend_class' in segment: del segment['trend_class']

        if trend_ends_too_close or trend_too_small or trend_too_flat:
            segment['direction'] = 'Flat' 
            if 'trend_class' in segment: del segment['trend_class']
        
        segments_refined.append(segment)

    return segments_refined


def fill_in_flats(df: pd.DataFrame, segments: list[dict]) -> list[dict]:
    """Fill uncovered time gaps with Flat segments using df's DateTimeIndex.

    Adds Flat segments for:
    - Internal gaps between consecutive segments.
    - Leading gap before the first segment if df index starts earlier.
    - Trailing gap after the last segment if df index ends later.
    """
    if not segments:
        if not df.empty:
            start, end = df.index.min(), df.index.max()
            return [dict(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), direction='Flat')]
        return []

    # Ensure df has a DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for fill_in_flats.")

    segments_refined = segments.copy()

    # Leading gap
    data_start = df.index.min()
    first_start = pd.to_datetime(segments_refined[0]['start'])
    if data_start < first_start:
        lead_end = first_start - pd.Timedelta(days=1)
        if lead_end >= data_start:
            segments_refined.insert(0, dict(
                start=data_start.strftime('%Y-%m-%d'),
                end=lead_end.strftime('%Y-%m-%d'),
                direction='Flat'
            ))

    # Internal gaps (work on snapshot to avoid index shift confusion)
    j = 0
    original = segments_refined.copy()
    for i, curr_seg in enumerate(original):
        mapped = i + j
        if mapped >= len(segments_refined) - 1:
            continue
        next_seg = segments_refined[mapped + 1]
        gap_start = pd.to_datetime(curr_seg['end']) + pd.Timedelta(days=1)
        gap_end = pd.to_datetime(next_seg['start']) - pd.Timedelta(days=1)
        if gap_end >= gap_start:
            segments_refined.insert(mapped + 1, dict(
                start=gap_start.strftime('%Y-%m-%d'),
                end=gap_end.strftime('%Y-%m-%d'),
                direction='Flat'
            ))
            j += 1

    # Trailing gap
    data_end = df.index.max()
    last_end = pd.to_datetime(segments_refined[-1]['end'])
    if data_end > last_end:
        trail_start = last_end + pd.Timedelta(days=1)
        if data_end >= trail_start:
            segments_refined.append(dict(
                start=trail_start.strftime('%Y-%m-%d'),
                end=data_end.strftime('%Y-%m-%d'),
                direction='Flat'
            ))

    return segments_refined
