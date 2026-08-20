"""**End-to-End Trend Detection**"""

import warnings
import pandas as pd
from .process_signals import process_signals
from .post_processing.segments_get import get_segments
from .post_processing.segments_refine import refine_segments
from .post_processing.segments_analyse import analyse_segments
from .io.plot_pytrendy import plot_pytrendy
from .io.results_pytrendy import PyTrendyResults
from .io import prepare_index


def detect_trends(df: pd.DataFrame, 
                  value_col: str,
                  date_col: str|None=None,
                  plot: bool=True, 
                  method_params: dict|None=None, 
                  plot_params: dict|None=None,
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
            Historically, this represents the name of the column containing dates, but pytrendy now allows for indexes of any type to be used. In general, this column represents a human readable reference to the x-position of the sequence. Normally this would be a date or timestamp, but any unique set of values could be used. Default is 'None', in which case an integer sequence will be generated and used to identify segments.
        plot (bool, optional):
            If `True`, generates a matplotlib plot showing the detected trend segments over the original signal.
            Defaults to `True`.
        method_params (dict, optional):
            Optional parameters to customize detection heuristics. Supported keys:

            - **abrupt_padding** (`int`): Number of days to pad around abrupt transitions. Defaults to `0`.
            - **avoid_noise** (`bool`): Whether to avoid noisy segments in trend detection. Defaults to `True`.
        plot_params (dict, optional):
            Optional dict to customise plot appearance. Only used when `plot` is `True`. Supported keys:

            - **figsize** (`tuple`): Figure size as (width, height). Defaults to (20, 5).
            - **title** (`str`): Plot title. Defaults to "PyTrendy Detection".
            - **xlabel** (`str`): X-axis label. Defaults to "Date".
            - **ylabel** (`str`): Y-axis label. Defaults to "Value".
            - **colors** (`dict`): Dictionary mapping direction ('Up', 'Down', 'Flat', 'Noise') to matplotlib colors. Defaults to light variants.
            - **alpha** (`float`): Transparency level for shaded regions. Defaults to 0.4.
            - **grid** (`dict`): Grid configuration with keys 'visible' (bool), 'which' (str), 'color' (str), 'alpha' (float).
            - **legend_loc** (`str`): Legend location. Defaults to "upper right".
            - **legend_bbox_to_anchor** (`tuple`): Legend box anchor position. Defaults to (1, 1.15).
        debug (bool, optional):
            If `True` will run in debug mode, outputting various additional plots and print statements. Only recommended for developers of pytrendy.
            Defaults to `False`.
            
    Returns:
        PyTrendyResults:
            An object encapsulating the detected segments and associated metadata.
            Use this object to access segment statistics, rankings, and export utilities.
    """
    # Stage the DataFrame on an internal integer index, keeping the external index
    # values and a lookup so boundaries can be remapped back later.
    df, external_index, index_lookup, index_type = prepare_index.prepare_index(df, date_col, value_col)

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

    # Translate internal segment boundaries back to the user's external index values.
    segments = prepare_index.remap_boundaries(segments, index_lookup)

    if plot:
        # Restore the external index for plotting before rendering the segments.
        plot_df = prepare_index.prepare_plot_frame(df, date_col, external_index, index_type)
        plot_pytrendy(df=plot_df, value_col=value_col, segments_enhanced=segments, index_type=index_type, plot_params=plot_params)

    results = PyTrendyResults(segments=segments, index_type=index_type)
    return results
