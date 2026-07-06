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
        df (pd.DataFrame): Series DataFrame.
        value_col (str): Name of the signal column.
        segments (list): List of segment dictionaries.

    Returns:
        list: Refined segment list with updated boundaries.
    """

    segments_refined = deepcopy(segments)

    def _get_window_df(center: int, days: int = 7) -> pd.DataFrame:
        """Return a slice of df around a center date ±days."""
        pre = center - days
        post = center + days
        return df.loc[pre:post].copy()

    for i, segment in enumerate(segments_refined):

        start_df = _get_window_df(segment['start'])
        end_df = _get_window_df(segment['end'])

        # Pre-crop local windows to avoid overlapping neighbouring NOISE segments
        # This ensures the extrema search doesn't pull from a noise neighbour region
        # and reduces the need for later conflict corrections.
        if i > 0: # handles right of noise
            prev_seg = segments_refined[i - 1]
            if prev_seg.get('direction') == 'Noise':
                prev_end = prev_seg['end']
                # Exclude days that belong to the previous noise segment
                crop_from = prev_end + 1
                cropped = start_df.loc[crop_from:]
                if not cropped.empty:
                    start_df = cropped

        if i < len(segments_refined) - 1: # handles left of noise
            next_seg = segments_refined[i + 1]
            if next_seg.get('direction') == 'Noise':
                next_start = next_seg['start']
                # Exclude days that belong to the next noise segment
                crop_to = next_start - 1
                cropped = end_df.loc[:crop_to]
                if not cropped.empty:
                    end_df = cropped

        if 'trend_class' in segment and segment['trend_class'] == 'abrupt':
            continue # don't expand/contract abrupt trends. Leave precise to shave.
        if segment['direction'] == 'Up':
            new_start = start_df[value_col].iloc[::-1].idxmin() + 1 # get min, latest if all same
            new_end = end_df[value_col].idxmax()
        elif segment['direction'] == 'Down':
            new_start = start_df[value_col].iloc[::-1].idxmax() + 1 # get max, latest if all same
            new_end = end_df[value_col].idxmin()
        else:
            continue

        # Avoid orphaning a peak/trough that no adjacent segment covers.
        # The +1 day pushes start past the extremum, assuming it belongs to
        # the neighbour. When no neighbour is near and the extremum holds a
        # significantly different value, the +1 day skips a genuine drop/rise.
        # Skip this when the previous segment is Noise — noise boundaries are
        # deliberately fuzzy and shouldn't anchor the orphan test.
        prev_seg = segments_refined[i - 1] if i > 0 else None
        prev_is_noise = prev_seg is not None and prev_seg.get('direction') == 'Noise'
        if segment['direction'] in ('Up', 'Down') and i > 0:
            prev_end = pd.to_datetime(segments_refined[i - 1]['end'])
            extremum = pd.to_datetime(new_start) - pd.Timedelta(days=1)
            distance = (extremum - prev_end).days
            # Skip orphan check when previous segment is Noise AND the Noise
            # is close (within 3 days) to the extremum — noise boundaries are
            # deliberately fuzzy in that case.  When Noise is far away, the
            # extremum is genuinely orphaned and should be captured.
            # For non-Noise neighbours, keep the original distance > 1 gate.
            if prev_is_noise and distance <= 3:
                pass
            elif (not prev_is_noise and distance > 1) or (prev_is_noise and distance > 3):
                if extremum in df.index and new_start in df.index:
                    extremum_val = df.loc[extremum, value_col]
                    start_val = df.loc[new_start, value_col]
                    max_abs = df[value_col].abs().max()
                    if max_abs > 0 and abs(extremum_val - start_val) > 0.2 * max_abs:
                        new_start -= pd.Timedelta(days=1)

        # Check for any inversions
        start_inverted = (new_start >= segment['end'])
        end_inverted = (new_end <= segment['start'])

        # Refine start provided valid to update
        start_changed = (new_start != segment['start'])
        if start_changed and not start_inverted:
            segments_refined[i]['start'] = new_start
            update_prev_segment(i, new_start, segments, segments_refined)

        # Refine end provided valid to update
        end_changed = (new_end != segment['end'])
        if end_changed and not end_inverted:
            segments_refined[i]['end'] = new_end
            update_next_segment(i, new_end, segments, segments_refined)

    return segments_refined
