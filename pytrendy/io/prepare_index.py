"""**Index Preparation for the Detection Pipeline**

PyTrendy's pipeline operates on a positional integer index internally, regardless of
the index the user supplies. This decouples the detection logic from any particular
time axis (daily, weekly, or otherwise) and lets the same pipeline accept datetime,
integer, float, or string indexes.

The preparation is a two-way translation:

1. **Inbound** — the user's index column (``date_col``) is inspected to determine its
   type, its values are captured, and the working DataFrame is re-staged on an internal
   integer index (``0..n-1``). A lookup table maps internal positions back to the
   original external index values.

2. **Outbound** — once segments are detected on the internal index, their boundaries are
   remapped back to the external index values before the results are returned or plotted.

The functions in this module encapsulate that translation so ``detect_trends`` stays
focused on orchestrating the pipeline.
"""

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd


def detect_index_type(df: pd.DataFrame, date_col: str) -> str:
    """
    Detect the index type from the date column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Name of the date column.

    Returns:
        str: Index type (``'date'``, ``'datetime64'``, ``'integer'``, ``'float'``, or ``'string'``).
    """
    if pd.api.types.is_string_dtype(df[date_col]):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*"
            )
            parsed = pd.to_datetime(df[date_col], errors="coerce")

        if parsed.notna().all():
            return "date"
        else:
            return "string"
    elif pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return "datetime64"
    elif pd.api.types.is_integer_dtype(df[date_col]):
        return "integer"
    elif pd.api.types.is_float_dtype(df[date_col]):
        return "float"
    else:
        raise NotImplementedError(f"date_col has unimplemented dtype {df[date_col].dtype}")


def build_index_lookup(external_index) -> dict:
    """
    Build a lookup mapping internal integer positions to external index values.

    Args:
        external_index: The original index values captured before staging.

    Returns:
        dict: Mapping from internal position (``int``) to external index value.
    """
    internal_index = np.arange(len(external_index))
    return dict(zip(internal_index, np.asarray(external_index)))


def prepare_index(df: pd.DataFrame, date_col: str | None, value_col: str) -> tuple:
    """
    Prepare the internal index framework used by the pipeline.

    Detects the index type, captures the external index values, builds the
    internal integer index and its lookup, and stages the working DataFrame
    on a dedicated scratch column so the user's columns are never clobbered.

    Args:
        df (pd.DataFrame): Input time series DataFrame.
        date_col (str|None): Name of the column representing the external index.
        value_col (str): Name of the signal column.

    Returns:
        tuple: ``(df, external_index, index_lookup, index_type)`` where ``df`` is
        the internal-indexed working copy, ``external_index`` holds the original
        index values, ``index_lookup`` maps internal to external index values, and
        ``index_type`` is the detected index type.
    """
    df = df.copy()
    index_type = 'integer'

    if date_col is not None:
        index_type = detect_index_type(df, date_col)
        external_index = df[date_col].copy()

        if index_type == 'date':
            df[date_col] = pd.to_datetime(df[date_col])
        elif index_type == 'string':
            warnings.warn(
                f"Attempting to cast {date_col} to date failed, "
                "treating as string lookup.",
                UserWarning,
                stacklevel=2,
            )
    else:
        external_index = np.arange(len(df))

    index_lookup = build_index_lookup(external_index)

    # Use a dedicated scratch column name to avoid clobbering user's columns
    _pytrendy_idx = '_pytrendy_idx'
    df[_pytrendy_idx] = np.arange(len(df))
    df.set_index(_pytrendy_idx, inplace=True)
    df = df[[value_col]]

    return df, external_index, index_lookup, index_type


def remap_boundaries(segments: list[dict], index_lookup: dict) -> list[dict]:
    """
    Remap internal segment boundaries back to external index values.

    Args:
        segments (list): Segment list with internal index boundaries.
        index_lookup (dict): Mapping from internal to external index values.

    Returns:
        list: A new segment list with boundaries expressed in external index values.
    """
    remapped = deepcopy(segments)
    for segment in remapped:
        segment['start'] = index_lookup[segment['start']]
        segment['end'] = index_lookup[segment['end']]
    return remapped


def prepare_plot_frame(df: pd.DataFrame, date_col: str | None, external_index, index_type: str) -> pd.DataFrame:
    """
    Restore the external index onto the working DataFrame for plotting.

    Args:
        df (pd.DataFrame): Internal-indexed working DataFrame.
        date_col (str|None): Name of the external index column.
        external_index: External index values captured before staging.
        index_type (str): Detected index type.

    Returns:
        pd.DataFrame: DataFrame with the external index restored for plotting.
    """
    if index_type == 'date':
        external_index = pd.to_datetime(external_index)

    df[date_col] = external_index
    df.set_index(date_col, inplace=True)
    return df
