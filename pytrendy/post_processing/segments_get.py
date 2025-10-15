"""**Extract Contiguous Segments from Signal Flags**"""

import pandas as pd

def get_segments(df: pd.DataFrame):
    """
    Extracts contiguous segments from a flagged time series.

    This function scans the `trend_flag` column in the input DataFrame and groups
    consecutive values into segments based on direction. It applies minimum length
    constraints to ensure meaningful segments are retained.

    Segment directions are mapped as follows:

    - `1`: Up
    - `-1`: Down
    - `-2`: Flat
    - `-3`: Noise


    Only segments meeting the following minimum lengths are retained:

    - Up/Down: ≥ 3 days
    - Flat/Noise: ≥ 1 days

    Args:
        df (pd.DataFrame): 
            Time series DataFrame containing a `trend_flag` column.

    Returns:
        list: 
            A list of dictionaries, each representing a segment with keys:

            - `'direction'`: Segment type (e.g., `'Up'`, `'Down'`)
            - `'start'`: Start date of the segment
            - `'end'`: End date of the segment
            - `'segmenth_length'`: Duration in days
            - `'time_index'`: Sequential index of the segment
    """
    map_direction = {
        0: 'Unknown'
        , 1: 'Up'
        , -1: 'Down'
        , -2: 'Flat'
        , -3: 'Noise'
    }

    segment_length = 0
    segment_length_prev = 0
    direction_prev = map_direction[0]
    segments = []

    for index, value in df[['trend_flag']].itertuples():
        direction = map_direction[value]
        if index == df.index.max(): direction = 'Done'

        if direction == direction_prev:
            segment_length += 1
        elif direction != direction_prev: 
            if (    # Save only when satisfies min window for up/down or flat respectively.
                    (direction_prev in ['Up', 'Down'] and (segment_length_prev >= 3)) \
                    or (direction_prev == 'Noise' and (segment_length_prev >= 1)) \
                    or (direction_prev == 'Flat' and (segment_length_prev >= 1)) \
                ):
                start = (pd.to_datetime(index) - pd.Timedelta(days=segment_length_prev+1))
                end = (pd.to_datetime(index) - pd.Timedelta(days=1))

                # Save the segment
                segments.append({
                    'direction': direction_prev
                    , 'start': start.strftime('%Y-%m-%d')
                    , 'end': end.strftime('%Y-%m-%d')
                })
                segment_length=0

        direction_prev = direction
        segment_length_prev = segment_length

    return segments # main result
