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
        if i > 0: # handles right of noise
            prev_seg = segments_refined[i - 1]
            if prev_seg.get('direction') == 'Noise':
                prev_end = pd.to_datetime(prev_seg['end'])
                # Exclude days that belong to the previous noise segment
                crop_from = (prev_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                cropped = start_df.loc[crop_from:]
                if not cropped.empty:
                    start_df = cropped

        if i < len(segments_refined) - 1: # handles left of noise
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


def pad_gradual_trends(df: pd.DataFrame, value_col: str, segments: list[dict], method_params: dict) -> list[dict]:
    """
    Extends gradual segment end dates by a specified number of days.

    Mirrors the padding behaviour for abrupt segments: extends the end date forward,
    truncating before any non-Flat segment that would be overlapped, and clamping
    to the last index date. Sets a ``padded`` flag on modified segments.

    Args:
        df (pd.DataFrame): Time series DataFrame.
        value_col (str): Name of the signal column.
        segments (list): List of segment dictionaries with ``'trend_class': 'gradual'``.
        method_params (dict): Supported keys:

            - **gradual_padding** (`int`): Number of days to pad. Defaults to ``0``.

    Returns:
        list: Segment list with padded gradual boundaries.
    """

    gradual_padding = method_params.get('gradual_padding', 0)
    if gradual_padding <= 0:
        return segments

    segments_padded = deepcopy(segments)

    meta_df = pd.DataFrame(segments)
    meta_df['start'] = pd.to_datetime(meta_df['start'])
    meta_df['end'] = pd.to_datetime(meta_df['end'])

    for i, segment in enumerate(segments):

        if segment['direction'] not in ['Up', 'Down'] or segment.get('trend_class') != 'gradual':
            continue

        gradual_end = pd.to_datetime(segment['end'])

        new_end = gradual_end + pd.Timedelta(days=gradual_padding)
        overlaps = meta_df.loc[(meta_df['start'] > gradual_end) & (meta_df['start'] <= new_end)]
        overlaps_nonflats = overlaps[overlaps['direction'] != 'Flat']

        if not overlaps_nonflats.empty:
            first_notflat_overlap = overlaps_nonflats.iloc[0]
            new_end = pd.to_datetime(first_notflat_overlap['start']) - pd.Timedelta(days=1)

        new_end = min(new_end, df.index[-1])
        segments_padded[i]['end'] = new_end.strftime('%Y-%m-%d')
        update_next_segment(i, new_end, segments, segments_padded)

        segments_padded[i]['padded'] = True if new_end != gradual_end else False

    return segments_padded
