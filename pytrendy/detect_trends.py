"""**End-to-End Trend Detection**"""

import warnings
import pandas as pd
import numpy as np
import warnings
from difflib import get_close_matches
from .process_signals import process_signals
from .post_processing.segments_get import get_segments
from .post_processing.segments_refine import refine_segments
from .post_processing.segments_analyse import analyse_segments
from .io.plot_pytrendy import plot_pytrendy
from .io.results_pytrendy import PyTrendyResults
from datetime import date

def detect_trends(df: pd.DataFrame, 
                  value_col: str,
                  date_col: str|None=None,
                  plot: bool=True, 
                  method_params: dict|None=None, 
                  debug: bool=False
                  ) -> PyTrendyResults:
    """
    This is the main function that runs trend detection end-to-end.
    
    It runs the full PyTrendy pipeline in five stages: signal smoothing, segment extraction, boundary refinement, metric analysis, and optional visualization. 
    It returns a `PyTrendyResults` object containing ranked, classified, and trend segments, ready for filtering, plotting, or export. 
    Furthermore, it identifies patterns such as uptrends, downtrends, flat regions, and noise by applying rolling statistics, segmentation heuristics, and post-processing refinements.
    It optionally visualizes the results and returns a structured object containing segment metadata.

    The pipeline includes:
    
    1. **Signal Processing**: Applies Savitzky-Golay smoothing and computes flags for flat and noisy regions.
    2. **Segmentation**: Extracts contiguous segments based on signal classification.
    3. **Refinement**: Adjusts segment boundaries and classifies trends as gradual or abrupt.
    4. **Analysis**: Computes metrics like total change, percent change, and signal-to-noise ratio.
    5. **Visualization (optional)**: Plots the original signal with annotated segments.

    Args:
        df (pd.DataFrame):
            Input time series data containing at least the specified `date_col` and `value_col`.
            The `date_col` must contain datetime-like values (daily frequency recommended).
        value_col (str):
            Name of the column containing the primary signal to analyse for trend detection.
        date_col (str|None):
            Historically, this represents the name of the column containing timestamps, but pytrendy now allows for indexes of any type to be used. In general, this column represents a human readable reference to the x-position of the sequence. Normally this would be a date or timestamp, but any unique set of values could be used. Default is 'None', in which case an integer sequence will be generated and used to idenify segmenets.
        plot (bool, optional):
            If `True`, generates a matplotlib plot showing the detected trend segments over the original signal.
            Defaults to `True`.
        method_params (dict, optional):
            Optional parameters to customize detection heuristics. Supported keys:

            - **abrupt_padding** (`int`): Number of days to pad around abrupt transitions. Defaults to `0`.
            - **avoid_noise** (`bool`): Whether to avoid noisy segments in trend detection. Defaults to `True`.
        debug (bool, optional):
            If `True` will run in debug mode, outputting various additional plots and print statements. Only recommended for developers of pytrendy.
            Defaults to `False`.
            
    Returns:
        PyTrendyResults:
            An object encapsulating the detected segments and associated metadata.
            Use this object to access segment statistics, rankings, and export utilities.
    """
    df = df.copy()
    index_type = 'integer'

    if date_col is not None:
        s = df[date_col]
        external_index = df[date_col].copy()
        if pd.api.types.is_string_dtype(df[date_col]):

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Could not infer format.*"
                )
                parsed = pd.to_datetime(df[date_col], errors="coerce")

            if parsed.notna().all():
                df[date_col] = parsed
                index_type = "date"
            else:
                print(
                    f"Attempting to cast {date_col} to date failed, "
                    "treating as string lookup."
                    )
                index_type = "string"
        elif pd.api.types.is_datetime64_any_dtype(df[date_col]):
            index_type = "datetime64"
        #elif s.map(lambda x: isinstance(x, date) or pd.isna(x)).all():
        #    index_type = "datetimePd"
        elif pd.api.types.is_integer_dtype(df[date_col]):
                pass
        elif pd.api.types.is_float_dtype(df[date_col]):
                index_type = "float"
        else:
            raise NotImplementedError(f"date_col has unimplimented dtype {df[date_col].dtype}") 
    else:
        external_index = np.arange(len(df))

    internal_index = np.arange(len(df))
    index_lookup = {internal_index[i] : external_index[i] for i in range(len(internal_index))}
    
    df[date_col] = internal_index.copy()
    df.set_index(date_col, inplace=True)
    df = df[[value_col]]
    
    if method_params is None:
        method_params = {} # Avoid mutable default argument by accepting None and constructing a new dict here

    # Trigger deprecation warning if old parameter is used    
    if 'is_abrupt_padded' in method_params:
        warnings.warn(
            "'is_abrupt_padded' in method_params is deprecated. "
            "Use 'abrupt_padding' only instead, which is 0 by default. Set to the number of days to pad by (e.g. 28).",
            DeprecationWarning,
            stacklevel=2,
        )

    # Configures trend detection heuristics
    method_params = {
        'abrupt_padding': method_params.get('abrupt_padding', 0),
        'avoid_noise': method_params.get('avoid_noise', True),
    }

    # Core 5-step pipeline
    df = process_signals(df, value_col, method_params, debug)
    segments = get_segments(df)
    segments = refine_segments(df, value_col, segments, method_params)
    segments = analyse_segments(df, value_col, segments)

    for segment in segments:
        segment['start'] = index_lookup[segment['start']]
        segment['end'] = index_lookup[segment['end']]

    if index_type == 'date':
        external_index = pd.to_datetime(external_index)

    if plot: 
        df[date_col] = external_index
        df.set_index(date_col, inplace=True)
        plot_pytrendy(df=df, value_col=value_col, segments_enhanced=segments, index_type=index_type)

    results = PyTrendyResults(segments=segments, index_type=index_type)
    return results