"""**Duration-Preserving Detection Window Scaling**

PyTrendy's detection windows are expressed as point counts, but they are
calibrated in days for daily-sampled data. When the same series is aggregated
to a coarser cadence (weekly, fortnightly, monthly, ...) a fixed point count
spans a much longer real-time duration, so the smoothing window swallows whole
cycles and everything collapses to Flat or Noise.

This module derives point counts from the sampling cadence so that a window
always spans approximately the same real-time duration regardless of how
densely the series is sampled. At daily spacing the derived counts equal the
historical constants exactly, so daily behaviour is unchanged.
"""

import numpy as np
import pandas as pd

DATE_INDEX_TYPES = ('date', 'datetime64')


def compute_gap_days(external_index, index_type: str) -> float:
    """Median spacing of the external index, in days.

    Returns ``1.0`` for non-date index types or series with fewer than two
    points. Those inputs have no meaningful "day" unit, and falling back to
    ``1.0`` preserves the historical point-count behaviour for them.

    Args:
        external_index: The external index values captured by ``prep_index``.
        index_type (str): Detected index type from ``detect_index_type``.

    Returns:
        float: Median gap between consecutive index values in days.
    """
    if index_type not in DATE_INDEX_TYPES or len(external_index) <= 1:
        return 1.0

    values = pd.to_datetime(pd.Series(np.asarray(external_index)))
    diffs = values.diff().dropna().dt.total_seconds() / 86400.0
    diffs = diffs[diffs > 0]  # ignore duplicate timestamps
    if diffs.empty:
        return 1.0
    return float(diffs.median())


def scale_window(base_days: int, gap_days: float, minimum: int = 1) -> int:
    """Scale a duration base (in days) to a point count for a sampling gap.

    Args:
        base_days (int): Real-time window length the default was calibrated for.
        gap_days (float): Median sampling gap in days.
        minimum (int): Lower bound for the returned point count.

    Returns:
        int: Point count spanning ``base_days`` at the given cadence, floored at
        ``minimum``.
    """
    if not np.isfinite(gap_days) or gap_days <= 0:
        gap_days = 1.0
    return max(int(base_days / gap_days), minimum)
