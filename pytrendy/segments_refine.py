
import pandas as pd
from copy import deepcopy

def refine_segments(df:pd.DataFrame, value_col: str, segments: list):
    """Post processing the segments. Slight tweak of segment starts & ends for more precision. Most useful in abrupt case."""
    segments_refined = deepcopy(segments)
    for i in range(len(segments)):

        segment = segments_refined[i]
        segment_prev = segments_refined[i-1] if i != 0 else None
        segment_next = segments_refined[i+1] if i != len(segments_refined)-1 else None

        pre_start = (pd.to_datetime(segment['start']) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        post_start = (pd.to_datetime(segment['start']) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        df_start = df.loc[pre_start:post_start].copy()
        
        pre_end = (pd.to_datetime(segment['end']) - pd.Timedelta(days=7)).strftime('%Y-%m-%d') 
        post_end = (pd.to_datetime(segment['end']) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        df_end = df.loc[pre_end:post_end].copy()
            
        if segment['direction'] == 'Up':

            lower_start = df_start.loc[:, value_col].idxmin() + pd.Timedelta(days=1)
            if lower_start != pd.to_datetime(segment['start']):
                segments_refined[i]['start'] = lower_start.strftime('%Y-%m-%d')

                # Update previous segment if overlaps (or used to overlap)
                prev_distance_refined = (pd.to_datetime(segments_refined[i]['start']) - pd.to_datetime(segment_prev['end'])).days if segment_prev else None
                prev_distance = (pd.to_datetime(segments[i]['start']) - pd.to_datetime(segment_prev['end'])).days if segment_prev else None
                prev_exists_and_touching = (segment_prev and prev_distance_refined <= 1) or (segment_prev and prev_distance <= 1)
                if prev_exists_and_touching: 
                    segments_refined[i-1]['end'] = (lower_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            higher_end = df_end.loc[:, value_col].idxmax()
            if higher_end != pd.to_datetime(segment['end']):
                segments_refined[i]['end'] = higher_end.strftime('%Y-%m-%d')

                # Update next segment if overlaps (or used to overlap)
                next_distance_refined = (pd.to_datetime(segment_next['start']) - pd.to_datetime(segments_refined[i]['end'])).days if segment_next else None
                next_distance = (pd.to_datetime(segment_next['start']) - pd.to_datetime(segments[i]['end'])).days if segment_next else None
                next_exists_and_touching = (segment_next and next_distance <= 1) or (segment_next and next_distance_refined <= 1)
                if next_exists_and_touching: 
                    segments_refined[i+1]['start'] = (higher_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        if segment['direction'] == 'Down':

            higher_start = df_start.loc[:, value_col].idxmax() + pd.Timedelta(days=1)
            if higher_start != pd.to_datetime(segment['start']):
                segments_refined[i]['start'] = higher_start.strftime('%Y-%m-%d')

                # Update previous segment if overlaps (or used to overlap)
                prev_distance_refined = (pd.to_datetime(segments_refined[i]['start']) - pd.to_datetime(segment_prev['end'])).days if segment_prev else None
                prev_distance = (pd.to_datetime(segments[i]['start']) - pd.to_datetime(segment_prev['end'])).days if segment_prev else None
                prev_exists_and_touching = (segment_prev and prev_distance_refined <= 1) or (segment_prev and prev_distance <= 1)

                if prev_exists_and_touching: 
                    segments_refined[i-1]['end'] = (higher_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            lower_end = df_end.loc[:, value_col].idxmin()
            if lower_end != pd.to_datetime(segment['end']):
                segments_refined[i]['end'] = lower_end.strftime('%Y-%m-%d')

                # Update next segment if touching/overlap (or used to overlap)
                next_distance_refined = (pd.to_datetime(segment_next['start']) - pd.to_datetime(segments_refined[i]['end'])).days if segment_next else None
                next_distance = (pd.to_datetime(segment_next['start']) - pd.to_datetime(segments[i]['end'])).days if segment_next else None
                next_exists_and_touching = (segment_next and next_distance <= 1) or (segment_next and next_distance_refined <= 1)

                if next_exists_and_touching: 
                    segments_refined[i+1]['start'] = (lower_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    return segments_refined