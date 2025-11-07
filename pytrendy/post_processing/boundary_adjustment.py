"""**Boundary Adjustment Utilities**

Functions for adjusting segment boundaries when neighboring segments are updated.
"""

import pandas as pd

NEIGHBOUR_DISTANCE = 3  # Distance for considering a neighbour to re-adjust after expand_contract or shave logic


def update_prev_segment(i: int, new_start: pd.Timestamp, segments: list[dict], segments_refined: list[dict]) -> None:
    """
    Adjusts the end of the previous segment if it overlaps with the updated start.

    Skips adjustment if the previous segment is classified as 'abrupt' or 'noise', preserving its precision.
    
    Args:
        i (int): Index of the current segment.
        new_start (str): Updated start date of the current segment.
        segments (list): Original segment list.
        segments_refined (list): Refined segment list being modified.
    """

    if (i == 0): return
    old_start = pd.to_datetime(segments[i]['start'])
    prev_segments = reversed(segments_refined[:i])

    for j, prevseg in enumerate(prev_segments):
        prev_start = pd.to_datetime(prevseg['start'])
        prev_end = pd.to_datetime(prevseg['end'])
        i_neighbour = i - (j+1)

        # Edge case 1.1: do not disturb previous trends if abrupt. Update them if gradual however.
        if (prevseg['direction'] in ['Up', 'Down'] and prevseg['trend_class'] == 'abrupt'):
            continue

        # # Edge case 1.2: do not disturb noise spikes (leave precise)
        if (prevseg['direction'] == 'Noise'):
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
        

def update_next_segment(i: int, new_end: pd.Timestamp, segments: list[dict], segments_refined: list[dict]) -> None:
    """
    Adjusts the start of the next segment if it overlaps with the updated end.

    Skips adjustment if the next segment is classified as 'abrupt', preserving its precision.

    Args:
        i (int): Index of the current segment.
        new_end (str): Updated end date of the current segment.
        segments (list): Original segment list.
        segments_refined (list): Refined segment list being modified.
    """
    if (i == len(segments) - 1): return
    old_end = pd.to_datetime(segments[i]['end'])
    next_segments = segments_refined[i+1:]

    for j, nextseg in enumerate(next_segments):
        next_start = pd.to_datetime(nextseg['start'])
        next_end = pd.to_datetime(nextseg['end'])
        i_neighbour = i + (j+1)

        # Edge case 1: do not disturb next trends if abrupt or gradual. They will refine themselves in next iteration.
        if (nextseg['direction'] in ['Up', 'Down']):
            continue

        # Edge case 1.2: do not disturb noise spikes (leave precise)
        if (nextseg['direction'] == 'Noise'):
            continue

        # Edge case 2: swallow neighbours that get fully overlapped.
        if next_end >= old_end and next_end <= new_end:
            segments_refined[i_neighbour]['start'] = (new_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            continue

        # Update when a valid neighbour of close enough distance.
        new_dist = (next_start - new_end).days
        old_dist = (next_start - old_end).days
        is_neighbour = (new_dist <= NEIGHBOUR_DISTANCE) or (old_dist <= NEIGHBOUR_DISTANCE)
        if is_neighbour:
            segments_refined[i_neighbour]['start'] = (new_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            return
