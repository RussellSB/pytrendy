import pandas as pd
from .process_signals import process_signals
from .segments_get import get_segments
from .segments_refine import refine_segments
from .segments_analyse import analyse_segments
from .plot_pytrendy import plot_pytrendy
from .results_pytrendy import PyTrendyResults

def detect_trends(df:pd.DataFrame, date_col:str, value_col: str, plot=True, method_params:dict={}):
    """
    Detects trends through a 5-step pipeline.
    1. Process Signals; Rolling statistics for detecting segments of flat, noise, and uptrend/downtrend
    2. Get Segments; Partitions the areas into segments provided long enough
    3. Refine Segments: Post processes the data to cater for rolling statistic edge cases.
    4. Analyse Segments: Provides change stats and ranking for each trend.
    5. Plot Trends (optional): Plots results with matplotlib.

    Returns: PyTrendyResults (obj)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

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