"""**Signal-Parameter Preparation**

The signal-processing stages share a common set of constants — smoothing and
rolling windows, grouping distance, and the noise/flat thresholds. ``detect_trends``
is the single public entry point, so this module is the one place those constants
are defaulted before the pipeline runs.

Defaults are calibrated in *days* and scaled to the observed sampling cadence, so a
window spans approximately the same real-time duration regardless of how densely the
series is sampled; at daily spacing the derived counts equal the historical constants
exactly. They are then merged with any user-supplied ``signal_params`` overrides. The
derived windows ``window_flat`` and ``window_noise`` are not top-level defaults: unless
explicitly overridden they derive from the cadence-scaled ``window_smooth`` via
``smooth_factor`` and ``noise_factor``. Every stage downstream assumes
:func:`prep_signal_params` returns a fully-populated dict.
"""

from .._spacing import scale_window

_SIGNAL_PARAMS_DEFAULTS = {
    'window_smooth': 15,        # days: Savitzky-Golay smoothing window.
    'smooth_factor': 0.5,       # Fraction of the cadence-derived window_smooth spanned by window_flat: raise to widen it (smoother flat baseline, fewer flags), lower to narrow it (more sensitive); applied to the scaled window so it stays proportional.
    'noise_factor': 0.5,        # Fraction of the cadence-derived window_smooth spanned by window_noise: raise to widen it (smoother SNR estimate, fewer flags), lower to narrow it (more sensitive); applied to the scaled window so it stays proportional.
    'grouping_distance': 7,     # days: maximum gap for grouping nearby segments.
    'min_trend_length': 3,      # days: minimum length for an Up/Down segment to be retained.
    'min_flat_noise_length': 1, # days: minimum length for a Flat/Noise segment to be retained.
    'threshold_noise': 2.5,     # SNR threshold (dB) below which a region is classified as noise.
    'threshold_smooth': 0.001,  # Derivative threshold as a fraction of the signal IQR, below which motion counts as flat.
    'threshold_flat': 0.835,    # Flat sensitivity as a fraction of the minimum non-zero rolling std.
}

# Savitzky-Golay requires window_length > polyorder (1), so scaling never floors below 2 points.
_MIN_SMOOTH_WINDOW = 2


def prep_signal_params(signal_params: dict|None=None, index_gap_days: float=1.0) -> dict:
    """
    Merge user-supplied `signal_params` over cadence-derived defaults.

    Point-count defaults are the historical day-calibrated constants scaled to
    the index's median sampling gap, so daily inputs reproduce the historical
    values exactly and coarser cadences span the same real-time duration.
    Explicit overrides are raw point counts and bypass the cadence scaling.

    `window_flat` and `window_noise` are not top-level defaults: unless explicitly
    overridden they derive from the cadence-scaled `window_smooth` via
    `smooth_factor` and `noise_factor`. Every stage downstream assumes this returns
    a fully-populated dict.

    Args:
        signal_params (dict, optional): User overrides. Unknown keys are forwarded silently.
        index_gap_days (float): Median sampling gap of the prepared index, in days.

    Returns:
        dict: Fully-populated signal_params with every key the stages read.
    """
    resolved = {
        'window_smooth': scale_window(_SIGNAL_PARAMS_DEFAULTS['window_smooth'], index_gap_days, minimum=_MIN_SMOOTH_WINDOW),
        'grouping_distance': scale_window(_SIGNAL_PARAMS_DEFAULTS['grouping_distance'], index_gap_days),
        'min_trend_length': scale_window(_SIGNAL_PARAMS_DEFAULTS['min_trend_length'], index_gap_days),
        'min_flat_noise_length': scale_window(_SIGNAL_PARAMS_DEFAULTS['min_flat_noise_length'], index_gap_days),
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
