"""**Adjust Boundaries and Classify Trends**

Main orchestration function for the full post-processing pipeline to refine detected trend segments.
"""

import pandas as pd
from copy import deepcopy
from .trend_classify import classify_trends
from .gradual_expand_contract import expand_contract_segments
from .abrupt_shaving import shave_abrupt_trends
from .segment_grouping import group_segments
from .artifact_cleanup import clean_artifacts, fill_in_flats


def refine_segments(df: pd.DataFrame, value_col: str, segments: list[dict], method_params: dict) -> list[dict]:
    """
    Full post-processing pipeline to refine detected trend segments.

    This function applies:
    - Trend classification (`gradual` vs `abrupt`)
    - Abrupt changepoint shaving
    - Gradual boundary expansion/contraction
    - Segment grouping
    - Artifact cleanup

    Args:
        df (pd.DataFrame): Time series DataFrame.
        value_col (str): Name of the signal column.
        segments (list): Initial segment list from detection.
        method_params (dict): Optional parameters for abrupt padding and control.

    Returns:
        list: Final refined segment list.
    """

    segments_refined = deepcopy(segments)
    
    segments_refined = classify_trends(df, value_col, segments_refined)
    segments_refined = group_segments(segments_refined) # grouping 1st pass: sporadic flats & noises

    segments_refined = expand_contract_segments(df, value_col, segments_refined) # for gradual
    segments_refined = shave_abrupt_trends(df, value_col, segments_refined, method_params) # for abrupt

    segments_refined = clean_artifacts(df, value_col, segments_refined, method_params) # cleans overlaps etc from expand/contract
    segments_refined = group_segments(segments_refined) # grouping 2nd pass: after trend refine and cleanup
    segments_refined = clean_artifacts(df, value_col, segments_refined, method_params) # cleans overlaps again after grouping

    init_segments = deepcopy(segments_refined)
    segments_refined = classify_trends(df, value_col, segments_refined) # reclassify after artifacts cleaned: some graduals to abrupt
    if segments_refined != init_segments: # only trigger if any re-classifications
        segments_refined = shave_abrupt_trends(df, value_col, segments_refined, method_params
                                            , second_pass=True, init_segments=init_segments) # abrupt shave 2nd pass: newly converted abrupts 
        segments_refined = group_segments(segments_refined) # make sure re-classifications are grouped to build strong enough cases for gradual -> abrupts
        segments_refined = clean_artifacts(df, value_col, segments_refined, method_params) # cleans overlaps etc from shave abrupt (precaution even though second_pass=True handles this)

    segments_refined = fill_in_flats(df, segments_refined) # fill uncovered gaps (leading, internal, trailing) with flats
    segments_refined = group_segments(segments_refined) # grouping 3rd pass (final): after abrupt shave 2nd pass and/or flat fill in
    segments_refined = clean_artifacts(df, value_col, segments_refined, method_params) # cleans 1-day flats if any leading/trailing
    
    return segments_refined