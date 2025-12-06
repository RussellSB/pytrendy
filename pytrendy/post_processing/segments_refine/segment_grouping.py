"""**Segment Grouping Utilities**

Functions for grouping consecutive segments with the same direction.
"""

import pandas as pd

GROUPING_DISTANCE = 7  # Distance for grouping segments of same type in group_segments


def group_segments(segments: list[dict]) -> list[dict]:
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
    # TODO: simplify with new grouping method written in process_signals for noise segments
    def flush_history(segment_history: list[dict], output: list[dict]) -> None:
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
