import pandas as pd
from copy import deepcopy

def refine_segments(df: pd.DataFrame, value_col: str, segments: list):
    """
    Post-process detected segments by refining their start and end points.
    Adjusts boundaries by looking ±7 days around each boundary for more precision.
    Is there an appropriately higher or lower point worth taking? Take it.
    """
    THRESHOLD_DISTANCE = 3

    segments_refined = deepcopy(segments)

    def _get_window_df(center, days=7):
        """Return a slice of df around a center date ±days."""
        pre = (pd.to_datetime(center) - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        post = (pd.to_datetime(center) + pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        return df.loc[pre:post].copy()

    def _update_prev_segment(i, new_start):
        """Shift previous segment end if overlapping with updated start (or original start)."""
        if i == 0:
            return
        distance_refined = (pd.to_datetime(new_start) - pd.to_datetime(segments_refined[i - 1]['end'])).days
        distance_orig = (pd.to_datetime(segments[i]['start']) - pd.to_datetime(segments[i - 1]['end'])).days
        if distance_refined <= THRESHOLD_DISTANCE or distance_orig <= THRESHOLD_DISTANCE:
            segments_refined[i - 1]['end'] = (pd.to_datetime(new_start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    def _update_next_segment(i, new_end):
        """Shift next segment start if overlapping with updated end (or original end)."""
        if i == len(segments_refined) - 1:
            return
        distance_refined = (pd.to_datetime(segments_refined[i + 1]['start']) - pd.to_datetime(new_end)).days
        distance_orig = (pd.to_datetime(segments[i + 1]['start']) - pd.to_datetime(segments[i]['end'])).days
        if distance_refined <= THRESHOLD_DISTANCE or distance_orig <= THRESHOLD_DISTANCE:
            segments_refined[i + 1]['start'] = (pd.to_datetime(new_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

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

        print(i, segment['direction'])
        # refine start
        if new_start != pd.to_datetime(segment['start']):
            segments_refined[i]['start'] = new_start.strftime('%Y-%m-%d')
            _update_prev_segment(i, new_start)

        # refine end
        if new_end != pd.to_datetime(segment['end']):
            segments_refined[i]['end'] = new_end.strftime('%Y-%m-%d')
            _update_next_segment(i, new_end)

        print('reasy')

    print(segments[5])
    print(segments_refined[5])
    return segments_refined