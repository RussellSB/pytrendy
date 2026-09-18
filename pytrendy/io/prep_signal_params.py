"""**Signal-Parameter Preparation**

The signal-processing stages share a common set of constants — smoothing and
rolling windows, grouping distance, and the noise/flat thresholds. ``detect_trends``
is the single public entry point, so this module is the one place those constants
are defaulted before the pipeline runs.

Defaults are merged with any user-supplied ``signal_params`` overrides. The derived
windows ``window_flat`` and ``window_noise`` are not top-level defaults: unless
explicitly overridden they derive from ``window_smooth``. Every stage downstream
assumes :func:`prep_signal_params` returns a fully-populated dict.
"""

_SIGNAL_PARAMS_DEFAULTS = {
    'window_smooth': 15,        # Savitzky-Golay smoothing window, in points.
    'smooth_factor': 0.5,       # Fraction of window_smooth spanned by window_flat: raise to widen it (smoother flat baseline, fewer flags), lower to narrow it (more sensitive); scaled to window_smooth so it stays proportional.
    'noise_factor': 0.5,        # Fraction of window_smooth spanned by window_noise: raise to widen it (smoother SNR estimate, fewer flags), lower to narrow it (more sensitive); scaled to window_smooth so it stays proportional.
    'grouping_distance': 7,     # Maximum gap, in index steps, for grouping nearby segments.
    'min_trend_length': 3,      # Minimum length, in steps, for an Up/Down segment to be retained.
    'min_flat_noise_length': 1, # Minimum length, in steps, for a Flat/Noise segment to be retained.
    'threshold_noise': 2.5,     # SNR threshold (dB) below which a region is classified as noise.
    'threshold_smooth': 0.001,  # Derivative threshold as a fraction of the signal IQR, below which motion counts as flat.
    'threshold_flat': 0.835,    # Flat sensitivity as a fraction of the minimum non-zero rolling std.
}


def prep_signal_params(signal_params: dict|None=None) -> dict:
    """
    Merge user-supplied `signal_params` over the defaults and derive window sizes.

    `window_flat` and `window_noise` are not top-level defaults: unless explicitly
    overridden they derive from `window_smooth`. Every stage downstream assumes
    this returns a fully-populated dict.

    Args:
        signal_params (dict, optional): User overrides. Unknown keys are forwarded silently.

    Returns:
        dict: Fully-populated signal_params with every key the stages read.
    """
    resolved = {**_SIGNAL_PARAMS_DEFAULTS, **(signal_params or {})}
    resolved.setdefault('window_flat', int(resolved['window_smooth'] * resolved['smooth_factor']))
    resolved.setdefault('window_noise', int(resolved['window_smooth'] * resolved['noise_factor']))
    return resolved
