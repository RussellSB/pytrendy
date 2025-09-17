"""**End-to-End Trend Detection**"""

import pandas as pd
from .process_signals import process_signals
from .post_processing.segments_get import get_segments
from .post_processing.segments_refine import refine_segments
from .post_processing.segments_analyse import analyse_segments
from .io.plot_pytrendy import plot_pytrendy
from .io.results_pytrendy import PyTrendyResults

def detect_trends(df:pd.DataFrame, date_col:str, value_col: str, plot=True, method_params:dict={}) -> PyTrendyResults:
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
        date_col (str):
            Name of the column representing timestamps. This column is converted to datetime and set as the index.
        value_col (str):
            Name of the column containing the primary signal to analyze for trend detection.
        plot (bool, optional):
            If `True`, generates a matplotlib plot showing the detected trend segments over the original signal.
            Defaults to `True`.
        method_params (dict, optional):
            Dictionary of optional parameters to customize detection heuristics. Supported keys:

                - `'is_abrupt_padded'` (`bool`, default: `False`): Whether to pad abrupt transitions between segments.
                - `'abrupt_padding'` (`int`, default: `28`): Number of days to pad around abrupt transitions.

    Returns:
        PyTrendyResults:
            An object encapsulating the detected segments and associated metadata.
            Use this object to access segment statistics, rankings, and export utilities.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df[[value_col]]

    # Configures trend detection heuristics
    method_params = {
        'is_abrupt_padded': method_params.get('is_abrupt_padded', False)
        , 'abrupt_padding': method_params.get('abrupt_padding', 28)
    }

    # Core 5-step pipeline
    df = process_signals(df, value_col)
    segments = get_segments(df)
    segments = refine_segments(df, value_col, segments, method_params)
    segments = analyse_segments(df, value_col, segments)
    if plot: plot_pytrendy(df, value_col, segments)

    results = PyTrendyResults(segments)
    return results