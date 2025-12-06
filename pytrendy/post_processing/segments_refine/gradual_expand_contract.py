"""**Expansion and Contraction Utilities**

Functions for refining segment boundaries by expanding or contracting based on local extrema.
"""

import pandas as pd
from copy import deepcopy
from .update_neighbours import update_prev_segment, update_next_segment


def expand_contract_segments(df: pd.DataFrame, value_col: str, segments: list[dict]) -> list[dict]:
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

    def _get_window_df(center: str, days: int = 7) -> pd.DataFrame:
        """Return a slice of df around a center date ±days."""
        pre = (pd.to_datetime(center) - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        post = (pd.to_datetime(center) + pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        return df.loc[pre:post].copy()

    for i, segment in enumerate(segments_refined):

        start_df = _get_window_df(segment['start'])
        end_df = _get_window_df(segment['end'])

        # Pre-crop local windows to avoid overlapping neighbouring NOISE segments
        # This ensures the extrema search doesn't pull from a noise neighbour region
        # and reduces the need for later conflict corrections.
        if i > 0:
            prev_seg = segments_refined[i - 1]
            if prev_seg.get('direction') == 'Noise':
                prev_end = pd.to_datetime(prev_seg['end'])
                # Exclude days that belong to the previous noise segment
                crop_from = (prev_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                cropped = start_df.loc[crop_from:]
                if not cropped.empty:
                    start_df = cropped

        if i < len(segments_refined) - 1:
            next_seg = segments_refined[i + 1]
            if next_seg.get('direction') == 'Noise':
                next_start = pd.to_datetime(next_seg['start'])
                # Exclude days that belong to the next noise segment
                crop_to = (next_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                cropped = end_df.loc[:crop_to]
                if not cropped.empty:
                    end_df = cropped

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

        # Check for any inversions
        start_inverted = (new_start >= pd.to_datetime(segment['end']))
        end_inverted = (new_end <= pd.to_datetime(segment['start']))

        # Refine start provided valid to update
        start_changed = (new_start != pd.to_datetime(segment['start']))
        if start_changed and not start_inverted:
            segments_refined[i]['start'] = new_start.strftime('%Y-%m-%d')
            update_prev_segment(i, new_start, segments, segments_refined)

        # Refine end provided valid to update
        end_changed = (new_end != pd.to_datetime(segment['end']))
        if end_changed and not end_inverted:
            segments_refined[i]['end'] = new_end.strftime('%Y-%m-%d')
            update_next_segment(i, new_end, segments, segments_refined)

    return segments_refined
