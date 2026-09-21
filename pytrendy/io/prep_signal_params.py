"""**Signal-Parameter Preparation**

The signal-processing stages share a common set of constants — smoothing and
rolling windows, grouping distance, and the noise/flat thresholds. ``detect_trends``
is the single public entry point, so this module is the one place those constants
are defaulted before the pipeline runs.

Defaults are calibrated as *durations* and divided by the observed sampling gap, so a
window spans approximately the same real-time duration regardless of how densely the
series is sampled; at daily spacing the derived counts equal the historical constants
exactly. They are then merged with any user-supplied ``signal_params`` overrides. The
derived windows ``window_flat`` and ``window_noise`` are not top-level defaults: unless
explicitly overridden they derive from the cadence-scaled ``window_smooth`` via
``smooth_factor`` and ``noise_factor``. Every stage downstream assumes
:func:`prep_signal_params` returns a fully-populated dict.
"""

from .._spacing import scale_window

# Duration ladder: the real-time span (days) the smoothing window targets for a
# cadence, selected by the series' median sampling gap. Sub-daily series smooth
# across ~24 h (24 steps at 1 h, 48 at 30 min); daily and coarser keep the
# historical 15-day window, which shrinks to proportionally fewer points as the
# cadence coarsens. Every other window and minimum length is a fraction of the
# same span, so all of them stay proportional to the cadence.
_WINDOW_LADDER = (
    (1.0, 1.0),             # intraday:    ~24 h
    (7.0, 15.0),            # daily:       15 d
    (14.0, 15.0),           # weekly:      15 d
    (28.0, 15.0),           # fortnightly: 15 d
    (200.0, 15.0),          # monthly:     15 d
    (float('inf'), 15.0),   # yearly+:     15 d
)
_GROUPING_FRACTION = 7 / 15
_MIN_TREND_FRACTION = 3 / 15
_MIN_FLAT_NOISE_FRACTION = 1 / 15


def _smooth_window_days(gap_days: float) -> float:
    """Real-time span (days) the smoothing window targets for a sampling gap."""
    for max_gap_days, duration_days in _WINDOW_LADDER:
        if gap_days < max_gap_days:
            return duration_days
    return _WINDOW_LADDER[-1][1]


# The remaining constants are dimensionless, so they need no cadence conversion.
_SIGNAL_PARAMS_DEFAULTS = {
    'smooth_factor': 0.5,       # Fraction of the cadence-derived window_smooth spanned by window_flat: raise to widen it (smoother flat baseline, fewer flags), lower to narrow it (more sensitive); applied to the scaled window so it stays proportional.
    'noise_factor': 0.5,        # Fraction of the cadence-derived window_smooth spanned by window_noise: raise to widen it (smoother SNR estimate, fewer flags), lower to narrow it (more sensitive); applied to the scaled window so it stays proportional.
    'threshold_noise': 2.5,     # SNR threshold (dB) below which a region is classified as noise.
    'threshold_smooth': 0.001,  # Derivative threshold as a fraction of the signal IQR, below which motion counts as flat.
    'threshold_flat': 0.835,    # Flat sensitivity as a fraction of the minimum non-zero rolling std.
}

# Savitzky-Golay requires window_length > polyorder (1). Five is the smallest floor
# that keeps the derived rolling gates alive: window_flat/window_noise = int(5 * 0.5) = 2,
# so their rolling std is defined; a 2-point window gave a 1-point (NaN) std, leaving
# flat_flag/noise_flag stuck at 0 and the savgol derivative a raw one-step diff sign.
_MIN_SMOOTH_WINDOW = 5


def prep_signal_params(signal_params: dict|None=None, index_gap_days: float=1.0, n_obs: int|None=None) -> dict:
    """
    Merge user-supplied `signal_params` over cadence-derived defaults.

    Every point-count default is a real-time duration from `_WINDOW_LADDER`
    divided by the index's median sampling gap, so a window spans the same
    duration at any cadence: ~24 h for sub-daily data and 15 days for daily and
    coarser, with grouping distance and minimum lengths proportional to the same
    span. Daily inputs reproduce the historical constants exactly. Explicit
    overrides are raw point counts and bypass the cadence scaling.

    `window_flat` and `window_noise` are not top-level defaults: unless explicitly
    overridden they derive from the cadence-scaled `window_smooth` via
    `smooth_factor` and `noise_factor`. Every stage downstream assumes this returns
    a fully-populated dict.

    Args:
        signal_params (dict, optional): User overrides. Unknown keys are forwarded silently.
        index_gap_days (float): Median sampling gap of the prepared index, in days.
        n_obs (int, optional): Number of observations; caps `window_smooth` at a
            quarter of the frame (and never above the series length) so a window
            cannot consume most of a short series. Defaults to no clamp.

    Returns:
        dict: Fully-populated signal_params with every key the stages read.
    """
    smooth_span_days = _smooth_window_days(index_gap_days)
    if n_obs is not None and 0 < index_gap_days < 1.0:
        # Intraday only: keep the target frame-relative as well as duration-relative.
        # At a 1-day sub-daily span the ~24 h target equals the whole frame (24 steps
        # at 1 h spacing), which smooths every leg away. Cap the target at a quarter
        # of the observations; window_flat/noise, grouping and the minimum lengths all
        # derive from this span, so they shrink with it. Daily and coarser (and
        # non-date indexes, gap=1.0) keep the historical duration target.
        smooth_span_days = min(smooth_span_days, max(3, n_obs // 4) * index_gap_days)
    window_smooth = scale_window(smooth_span_days, index_gap_days, minimum=_MIN_SMOOTH_WINDOW)
    if n_obs is not None:
        # The floor yields to short frames: never request a window longer than the series.
        window_smooth = min(window_smooth, n_obs)
    resolved = {
        'window_smooth': window_smooth,
        'grouping_distance': scale_window(smooth_span_days * _GROUPING_FRACTION, index_gap_days),
        'min_trend_length': scale_window(smooth_span_days * _MIN_TREND_FRACTION, index_gap_days),
        'min_flat_noise_length': scale_window(smooth_span_days * _MIN_FLAT_NOISE_FRACTION, index_gap_days),
        'smooth_factor': _SIGNAL_PARAMS_DEFAULTS['smooth_factor'],
        'noise_factor': _SIGNAL_PARAMS_DEFAULTS['noise_factor'],
        'threshold_noise': _SIGNAL_PARAMS_DEFAULTS['threshold_noise'],
        'threshold_smooth': _SIGNAL_PARAMS_DEFAULTS['threshold_smooth'],
        'threshold_flat': _SIGNAL_PARAMS_DEFAULTS['threshold_flat'],
    }
    resolved.update(signal_params or {})
    resolved.setdefault('window_flat', int(resolved['window_smooth'] * resolved['smooth_factor']))
    resolved.setdefault('window_noise', int(resolved['window_smooth'] * resolved['noise_factor']))
    return resolved
