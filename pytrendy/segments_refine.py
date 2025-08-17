
import pandas as pd

def refine_segments(df:pd.DataFrame, value_col: str, segments: list):
    """Post processing the segments. Slight tweak of segment starts & ends for more precision. Most useful in abrupt case."""
    segments_refined = segments.copy()
    for i in range(len(segments)):

        segment = segments[i]
        segment_prev = segments[i-1] if i != 0 else None
        segment_next = segments[i+1] if i != len(segments)-1 else None

        prev_distance = (pd.to_datetime(segment['start']) - pd.to_datetime(segment_prev['end'])).days if segment_prev else None
        next_distance = (pd.to_datetime(segment_next['start']) - pd.to_datetime(segment['end'])).days if segment_next else None

        prev_exists_and_touching = (segment_prev and prev_distance <= 1)
        next_exists_and_touching = (segment_next and next_distance <= 1)

        if segment['direction'] == 'Up':

            ### EXPANSION

            # Refine uptrend's start date to be lower if possible
            if segment['start'] != df.index[0].strftime('%Y-%m-%d'):

                # Using diff, find closest low and closest high
                temp = df.loc[:segment['start']].copy()
                temp['diff'] = temp[value_col].diff()
                temp = temp[:-2]

                closestlow = temp.index[(temp["diff"] <= 0)][-1]
                closesthigh = temp.index[(temp["diff"] > 0)][-1]
                
                start_value = df.loc[segment['start'], value_col]
                closestlow_value = df.loc[closestlow, value_col]

                # Edge cases
                found_continuous = closestlow > closesthigh
                found_lower = closestlow_value < start_value
                
                if found_continuous and found_lower: 
                    # Select new candidate if it passes edge cases
                    betterstart = closestlow.strftime('%Y-%m-%d')
                    segments_refined[i]['start'] = betterstart

                    # Update previous segment if touching/overlap
                    prev_distance_refined = (pd.to_datetime(segments_refined[i]['start']) - pd.to_datetime(segment_prev['end'])).days if segment_prev else None
                    prev_exists_and_touching_refined = (segment_prev and prev_distance_refined <= 1)
                    if prev_exists_and_touching_refined: segments_refined[i-1]['end'] = (closestlow - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            
            # Refine uptrend's end date to be higher if possible
            if segment['end'] != df.index[-1].strftime('%Y-%m-%d'):

                # Using diff, check perspective of after end forwards
                temp = df.loc[segment['end']:].copy()
                temp['diff'] = temp[value_col].diff()
                temp = temp[2:]
                
                closestlow = temp.index[(temp["diff"] <= 0)][0]
                closesthigh = temp.index[(temp["diff"] > 0)][0]

                end_value = df.loc[segment['end'], value_col]
                closesthigh_value = df.loc[closesthigh, value_col]
                
                # Edge cases
                found_continuous = closesthigh < closestlow
                found_higher = closesthigh_value > end_value

                if found_continuous and found_higher:
                    # Select new candidate if it passes edge cases
                    betterend = closesthigh.strftime('%Y-%m-%d')
                    segments_refined[i]['end'] = betterend
                    
                    # Update next segment if touching/overlap
                    next_distance_refined = (pd.to_datetime(segment_next['start']) - pd.to_datetime(segments_refined[i]['end'])).days if segment_next else None
                    next_exists_and_touching_refined = (segment_next and next_distance_refined <= 1)
                    if next_exists_and_touching_refined: segments_refined[i+1]['start'] = (closesthigh + pd.Timedelta(days=1)).strftime('%Y-%m-%d')


        elif segment['direction'] == 'Down': pass

    return segments_refined