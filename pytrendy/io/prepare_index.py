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


def detect_index_type(values) -> str:
    """
    Detect the index type from a Series or Index of values.

    Args:
        values: A pandas Series or Index of index values.

    Returns:
        str: Index type (``'date'``, ``'datetime64'``, ``'integer'``, ``'float'``, or ``'string'``).
    """
    if pd.api.types.is_string_dtype(values):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*"
            )
            parsed = pd.to_datetime(values, errors="coerce")

        if parsed.notna().all():
            return "date"
        else:
            return "string"
    elif pd.api.types.is_datetime64_any_dtype(values):
        return "datetime64"
    elif pd.api.types.is_integer_dtype(values):
        return "integer"
    elif pd.api.types.is_float_dtype(values):
        return "float"
    else:
        raise NotImplementedError(f"unimplemented dtype {values.dtype}")


def is_legacy_positional_order(df: pd.DataFrame, value_col: str, date_col: str) -> bool:
    """
    Detect the deprecated ``detect_trends(df, date_col, value_col)`` positional order.

    Under the old API the first positional column was always date-like (it was passed
    through ``pd.to_datetime``) and the second was the numeric value column. So if the
    column currently bound as ``value_col`` is date-like while ``date_col`` is numeric,
    the caller almost certainly used the legacy order.

    Args:
        df (pd.DataFrame): Input DataFrame.
        value_col (str): Column currently bound as the value/signal column.
        date_col (str): Column currently bound as the date/index column.

    Returns:
        bool: True if the arguments appear to be in the legacy (date_col, value_col) order.
    """
    value_dtype = df[value_col].dtype
    date_dtype = df[date_col].dtype

    value_is_datelike = pd.api.types.is_datetime64_any_dtype(value_dtype) or (
        pd.api.types.is_string_dtype(value_dtype)
        and pd.to_datetime(df[value_col], errors="coerce").notna().all()
    )
    date_is_numeric = pd.api.types.is_numeric_dtype(date_dtype)

    return value_is_datelike and date_is_numeric


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

    if date_col is not None:
        index_type = detect_index_type(df[date_col])
        index_values = df[date_col]

        # Sort sortable index types ascending so the internal positional index
        # reflects chronological (date) / numeric order. Non-sortable types
        # ('string') keep their given order.
        if index_type in ('date', 'datetime64', 'integer', 'float'):
            sort_key = pd.to_datetime(index_values) if index_type == 'date' else index_values
            df = df.iloc[np.asarray(sort_key).argsort(kind='stable')].reset_index(drop=True)
        external_index = df[date_col].copy()

        if index_type == 'string':
            warnings.warn(
                f"Attempting to cast {date_col} to date failed, "
                "treating as string lookup.",
                UserWarning,
                stacklevel=2,
            )
    else:
        # No date column: fall back to the DataFrame's own index.
        index_type = detect_index_type(df.index)
        index_values = df.index

        # Sort sortable index types ascending; non-sortable ('string') and the
        # default integer index keep their given order.
        if index_type in ('datetime64', 'integer', 'float'):
            df = df.sort_index(kind='stable')
        elif index_type == 'date':  # string-date index: sort chronologically, keep labels
            order = np.asarray(pd.to_datetime(df.index)).argsort(kind='stable')
            df = df.iloc[order]
        external_index = np.asarray(df.index)

    if index_type == 'float' and pd.isna(index_values).any():
        warnings.warn(
            "float index contains NaN values; they sort to the end and may "
            "produce unexpected segment boundaries.",
            UserWarning,
            stacklevel=2,
        )

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

    # Use a sentinel name when no date column was supplied, so the restored
    # index has a meaningful label rather than a None-named column.
    index_name = date_col if date_col is not None else '_index'
    df[index_name] = np.asarray(external_index)
    df.set_index(index_name, inplace=True)
    return df
