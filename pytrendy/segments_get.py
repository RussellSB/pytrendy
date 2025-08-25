import pandas as pd

def get_segments(df: pd.DataFrame):
    """Chisels out continuous segments from signals that indicate the areas."""
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
                    (direction_prev in ['Up', 'Down'] and (segment_length_prev >= 7)) \
                    or (direction_prev == 'Flat' and (segment_length_prev >= 3)) \
                    or (direction_prev == 'Noise' and (segment_length_prev >= 3)) \
                ):
                start = (pd.to_datetime(index) - pd.Timedelta(days=segment_length_prev+1))
                end = (pd.to_datetime(index) - pd.Timedelta(days=1))

                # Save the segment
                segments.append({
                    'direction': direction_prev
                    , 'segmenth_length': segment_length_prev
                    , 'start': start.strftime('%Y-%m-%d')
                    , 'end': end.strftime('%Y-%m-%d')
                })
                segment_length=0

        direction_prev = direction
        segment_length_prev = segment_length

    for i, _ in enumerate(segments):
        segments[i]['time_index'] = i+1

    return segments # main result
